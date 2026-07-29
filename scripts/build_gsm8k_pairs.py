#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmdpo.data import write_jsonl
from cmdpo.verifier import verify_answer


def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    num_return_sequences: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    use_chat_template: bool,
) -> list[str]:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
    else:
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **encoded,
        do_sample=True,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_len = encoded["input_ids"].shape[-1]
    return tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    model.eval()

    dataset = load_dataset("gsm8k", "main", split=args.split)
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(tqdm(dataset.select(range(min(args.limit, len(dataset)))))):
        question = item["question"]
        answer = item["answer"]
        prompt = (
            f"Question: {question}\n"
            "Answer with 3 to 6 numbered steps. "
            "Make each step short and include any arithmetic explicitly. "
            "End with a final line in the form '#### <answer>'."
        )
        candidates = generate_responses(
            model,
            tokenizer,
            prompt,
            args.num_candidates,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.use_chat_template,
        )
        correct = [c for c in candidates if verify_answer(c, answer)]
        incorrect = [c for c in candidates if not verify_answer(c, answer)]
        if not incorrect:
            continue
        chosen = correct[0] if correct else item["answer"]
        rows.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": incorrect[0],
                "answer": answer,
                "metadata": {
                    "source": "gsm8k",
                    "row_id": idx,
                    "num_candidates": len(candidates),
                    "has_sampled_chosen": bool(correct),
                },
            }
        )

    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} preference pairs to {args.output}")


if __name__ == "__main__":
    main()
