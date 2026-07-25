#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmdpo.data import read_jsonl, write_jsonl
from cmdpo.segmentation import segment_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-step-chars", type=int, default=8)
    parser.add_argument("--max-step-chars", type=int, default=500)
    args = parser.parse_args()

    rows = []
    for row in read_jsonl(args.input):
        steps = segment_steps(
            row["rejected"],
            min_chars=args.min_step_chars,
            max_chars=args.max_step_chars,
        )
        row["rejected_steps"] = steps
        rows.append(row)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} segmented rows to {args.output}")


if __name__ == "__main__":
    main()
