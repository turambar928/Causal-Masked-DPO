#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmdpo.data import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--test-output", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if not rows:
        raise ValueError("input dataset is empty")
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be in (0, 1)")

    rng = random.Random(args.seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)

    split_idx = max(1, min(len(rows) - 1, int(round(len(rows) * args.train_ratio))))
    train_rows = [rows[i] for i in indices[:split_idx]]
    test_rows = [rows[i] for i in indices[split_idx:]]

    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.test_output, test_rows)
    print(f"Wrote {len(train_rows)} train rows to {args.train_output}")
    print(f"Wrote {len(test_rows)} test rows to {args.test_output}")


if __name__ == "__main__":
    main()
