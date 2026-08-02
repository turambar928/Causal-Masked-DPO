#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmdpo.data import read_jsonl, write_jsonl


def cmdpo_weights(num_steps: int, first_error: int, gamma: float) -> list[float]:
    return [
        0.0 if k < first_error else (1.0 if k == first_error else gamma ** (k - first_error))
        for k in range(num_steps)
    ]


def shifted_error(row: dict[str, Any], shift: int) -> dict[str, Any]:
    item = dict(row)
    steps = item["rejected_steps"]
    original = int(item["first_error_step"])
    shifted = min(max(original + shift, 0), len(steps) - 1)
    item["first_error_step"] = shifted
    item["step_weights"] = cmdpo_weights(len(steps), shifted, float(item.get("metadata", {}).get("gamma", 0.25)))
    item["metadata"] = {
        **item.get("metadata", {}),
        "reviewer_variant": f"shift_error_{shift:+d}",
        "original_first_error_step": original,
        "shifted_first_error_step": shifted,
    }
    return item


def truncated_rejected(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    m = int(item["first_error_step"])
    kept_steps = item["rejected_steps"][: m + 1]
    item["rejected_steps"] = kept_steps
    item["rejected"] = "\n".join(kept_steps)
    item["first_error_step"] = len(kept_steps) - 1
    item["step_weights"] = [1.0] * len(kept_steps)
    item["metadata"] = {
        **item.get("metadata", {}),
        "reviewer_variant": "truncated_rejected_through_first_error",
        "original_num_rejected_steps": len(row["rejected_steps"]),
        "original_first_error_step": m,
    }
    return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=["shift_plus_one", "shift_minus_one", "truncated"], required=True)
    parser.add_argument("--gamma", type=float, default=0.25)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    output = []
    for row in rows:
        row = dict(row)
        row["metadata"] = {**row.get("metadata", {}), "gamma": args.gamma}
        if args.variant == "shift_plus_one":
            output.append(shifted_error(row, 1))
        elif args.variant == "shift_minus_one":
            output.append(shifted_error(row, -1))
        elif args.variant == "truncated":
            output.append(truncated_rejected(row))
        else:
            raise ValueError(args.variant)

    write_jsonl(args.output, output)
    print(f"Wrote {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()
