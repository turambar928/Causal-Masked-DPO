#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmdpo.data import read_jsonl, write_jsonl
from cmdpo.verifier import extract_answer, normalize_answer, verify_answer
from cmdpo.prompting import math_instruction, format_prompt


def load_model(
    model_path: str,
    adapter_path: str | None,
    dtype: torch.dtype,
    device_map: str | None,
) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.eval()
    return model


def build_eval_prompt(prompt: str) -> str:
    return math_instruction(prompt)


def generate_one(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool,
) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    use_chat_template: bool,
    return_metadata: bool = False,
) -> list:
    formatted = [format_prompt(tokenizer, prompt, use_chat_template) for prompt in prompts]
    encoded = tokenizer(formatted, return_tensors="pt", add_special_tokens=False, padding=True)
    device = next(model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    input_length = encoded["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completions = []
    for row in output_ids:
        tokens = row[input_length:]
        ends = (tokens == tokenizer.eos_token_id).nonzero().flatten()
        stopped = len(ends) > 0
        length = int(ends[0]) + 1 if stopped else len(tokens)
        text = tokenizer.decode(tokens[:length], skip_special_tokens=True)
        completions.append({"text": text, "generated_tokens": length, "stopped_on_eos": stopped,
                            "hit_token_limit": not stopped and length >= max_new_tokens}
                           if return_metadata else text)
    return completions


def evaluate_model(
    name: str,
    model: torch.nn.Module,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_new_tokens: int,
    use_chat_template: bool,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details = []
    correct = 0
    for start in range(0, len(rows), batch_size):
        end = min(len(rows), start + batch_size)
        print(f"[{name}] samples {start + 1}-{end}/{len(rows)}", flush=True)
        batch_rows = rows[start:end]
        eval_prompts = [build_eval_prompt(row["prompt"]) for row in batch_rows]
        completions = generate_batch(model, tokenizer, eval_prompts, max_new_tokens, use_chat_template, return_metadata=True)
        for offset, (row, generation) in enumerate(zip(batch_rows, completions, strict=True)):
            completion = generation["text"]
            idx = start + offset
            is_correct = verify_answer(completion, row["answer"])
            correct += int(is_correct)
            details.append(
                {
                    "model": name,
                    "index": idx,
                    "prompt": row["prompt"],
                    "gold": row["answer"],
                    "prediction": completion,
                    "pred_answer": normalize_answer(extract_answer(completion)),
                    "correct": is_correct,
                    "prediction_tokens": generation["generated_tokens"],
                    "stopped_on_eos": generation["stopped_on_eos"],
                    "hit_token_limit": generation["hit_token_limit"],
                    "has_explicit_answer": "####" in completion or "\\boxed{" in completion,
                    "metadata": row.get("metadata", {}),
                }
            )
    total = len(rows)
    return (
        {
            "model": name,
            "accuracy": correct / max(1, total),
            "correct": correct,
            "total": total,
        },
        details,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--details-output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--adapter", action="append", default=[], help="name=path")
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--limit", type=int, help="Diagnostic subset only; do not use for final benchmark tables")
    args = parser.parse_args()
    torch.set_num_threads(4)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
    device_map = "auto" if torch.cuda.is_available() else None

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[:args.limit]

    runs: list[tuple[str, str | None]] = [] if args.skip_base else [("base", None)]
    for adapter_arg in args.adapter:
        if "=" not in adapter_arg:
            raise ValueError("--adapter must use name=path")
        name, path = adapter_arg.split("=", 1)
        runs.append((name, path))

    summaries = []
    all_details = []
    for name, adapter_path in runs:
        print(f"Loading {name}", flush=True)
        model = load_model(args.model, adapter_path, dtype=dtype, device_map=device_map)
        summary, details = evaluate_model(
            name,
            model,
            tokenizer,
            rows,
            args.max_new_tokens,
            args.use_chat_template,
            args.batch_size,
        )
        summaries.append(summary)
        all_details.extend(details)
        write_jsonl(args.summary_output, summaries)
        write_jsonl(args.details_output, all_details)
        print(
            f"Finished {name}: accuracy={summary['accuracy']:.4f} "
            f"correct={summary['correct']}/{summary['total']}",
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("model,accuracy,correct,total")
    for row in summaries:
        print(f"{row['model']},{row['accuracy']:.4f},{row['correct']},{row['total']}")


if __name__ == "__main__":
    main()
