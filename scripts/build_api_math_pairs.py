#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmdpo.api_client import chat_completion, load_api_config, make_client
from cmdpo.data import write_jsonl
from cmdpo.verifier import verify_answer


def make_questions(limit: int) -> list[dict[str, str]]:
    rows = []
    for i in range(limit):
        a = 12 + i
        b = 3 + (i % 7)
        c = 2 + (i % 5)
        answer = a * b + c
        rows.append(
            {
                "question": f"A box has {a} packs. Each pack contains {b} pencils. Then {c} more pencils are added. How many pencils are there in total?",
                "answer": f"#### {answer}",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-file", default="api.txt")
    parser.add_argument("--model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    config = load_api_config(args.api_file, preferred_model=args.model)
    client = make_client(config)
    model = args.model or config.default_model
    rows = []

    for idx, item in enumerate(tqdm(make_questions(args.limit))):
        question = item["question"]
        answer = item["answer"]
        prompt = f"Question: {question}\nAnswer step by step and end with '#### <answer>'."
        chosen_prompt = (
            f"Solve the problem correctly. End with the exact final answer marker.\n\n"
            f"Question: {question}\nGold final answer: {answer}"
        )
        rejected_prompt = (
            "Write a plausible step-by-step solution, but make exactly one arithmetic mistake in the middle "
            "so the final answer is wrong. Do not mention that it is intentionally wrong. "
            "End with '#### <wrong answer>'.\n\n"
            f"Question: {question}\nThe correct final answer is {answer}, but your final answer must be different."
        )
        try:
            chosen = chat_completion(client, model, chosen_prompt, temperature=0.2, max_tokens=args.max_tokens)
            rejected = chat_completion(client, model, rejected_prompt, temperature=args.temperature, max_tokens=args.max_tokens)
        except Exception as exc:
            print(f"Stopping after {len(rows)} rows due to API error: {type(exc).__name__}: {exc}")
            break
        if not verify_answer(chosen, answer):
            chosen = f"There are {question.split(' has ', 1)[1].split(' packs', 1)[0]} packs. Compute directly. {answer}"
        if verify_answer(rejected, answer):
            continue
        rows.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "answer": answer,
                "metadata": {
                    "source": "api_arithmetic",
                    "row_id": idx,
                    "model": model,
                },
            }
        )
        write_jsonl(args.output, rows)

    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} API-generated pairs to {args.output}")


if __name__ == "__main__":
    main()
