#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmdpo.data import write_jsonl


def gsm8k_to_pairs(split: str, limit: int | None = None) -> list[dict[str, Any]]:
    dataset = load_dataset("gsm8k", "main", split=split)
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(dataset):
        if limit is not None and idx >= limit:
            break
        question = item["question"]
        answer = item["answer"]
        rows.append(
            {
                "prompt": f"Question: {question}\nAnswer step by step.",
                "chosen": answer,
                "rejected": answer,
                "answer": answer,
                "metadata": {
                    "source": "gsm8k",
                    "row_id": idx,
                    "note": "Placeholder pair. Replace rejected with sampled incorrect responses for real DPO.",
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gsm8k"], default="gsm8k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.dataset == "gsm8k":
        rows = gsm8k_to_pairs(args.split, args.limit)
    else:
        raise ValueError(args.dataset)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    print("Note: prepare_data.py creates placeholder pairs; sample incorrect rejected responses before training.")


if __name__ == "__main__":
    main()
