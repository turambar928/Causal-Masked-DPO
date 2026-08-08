#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmdpo.data import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-prefix-steps", type=int, default=1)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    output: list[dict[str, Any]] = []
    for row in rows:
        m = int(row["first_error_step"])
        if m < args.min_prefix_steps:
            continue
        item = dict(row)
        item["positive_prefix"] = "\n".join(item["rejected_steps"][:m]).strip()
        item["metadata"] = {
            **item.get("metadata", {}),
            "process_positive": True,
            "positive_prefix_steps": m,
        }
        output.append(item)

    write_jsonl(args.output, output)
    print(f"Wrote {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()
