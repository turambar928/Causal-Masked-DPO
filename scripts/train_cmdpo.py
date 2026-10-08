#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import hashlib
import importlib.metadata
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmdpo.collator import CMDPOCollator
from cmdpo.data import read_jsonl
from cmdpo.trainer import CMDPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--normalize-rejected", action="store_true")
    parser.add_argument("--process-positive-weight", type=float, default=0.0)
    parser.add_argument("--chosen-nll-weight", type=float, default=0.0)
    parser.add_argument("--objective", choices=["dpo", "sft"], default="dpo")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--strict-length", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--reference-cache")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(4)
    output_dir = Path(args.output_dir)
    if (output_dir / "adapter_model.safetensors").exists():
        raise FileExistsError(f"Refusing to overwrite completed adapter: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {**vars(args), "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                  "versions": {n: importlib.metadata.version(n) for n in ["torch", "transformers", "peft"]}}
    (output_dir / "run_config.json").write_text(json.dumps(provenance, indent=2) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    policy = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    ref = None if args.reference_cache or args.objective == "sft" else AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if args.use_lora:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        policy = get_peft_model(policy, lora_config)

    rows = read_jsonl(args.data)
    collator = CMDPOCollator(tokenizer=tokenizer, max_length=args.max_length,
                            use_chat_template=args.use_chat_template, strict_length=args.strict_length)
    if args.reference_cache:
        from cmdpo.reference_cache import attach_reference_cache
        rows = attach_reference_cache(rows, collator, args.reference_cache, args.model, args.normalize_rejected)
    dataset = Dataset.from_list(rows)
    # Fail BEFORE training if a mask/error is lost to truncation.
    if args.strict_length:
        for row in rows:
            collator([row])

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=10,
        save_strategy="no",
        max_steps=args.max_steps,
        disable_tqdm=True,
        bf16=use_bf16,
        fp16=use_fp16,
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        report_to=[],
    )
    trainer = CMDPOTrainer(
        model=policy,
        ref_model=ref,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
        beta=args.beta,
        normalize_rejected=args.normalize_rejected,
        process_positive_weight=args.process_positive_weight,
        chosen_nll_weight=args.chosen_nll_weight,
        objective=args.objective,
    )
    result = trainer.train()
    trainer.save_model(args.output_dir)
    trainer.save_state()
    (output_dir / "train_metrics.json").write_text(json.dumps(result.metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
