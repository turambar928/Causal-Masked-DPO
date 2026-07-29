#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmdpo.data import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    dataset = load_dataset("gsm8k", "main", split=args.split)
    rows = []
    for idx, item in enumerate(dataset.select(range(min(args.limit, len(dataset))))):
        rows.append(
            {
                "prompt": f"Question: {item['question']}",
                "answer": item["answer"],
                "metadata": {
                    "source": "gsm8k",
                    "split": args.split,
                    "row_id": idx,
                },
            }
        )

    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} eval rows to {args.output}")


if __name__ == "__main__":
    main()
