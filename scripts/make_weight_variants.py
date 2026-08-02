#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmdpo.data import read_jsonl, write_jsonl


def weights_for_variant(num_steps: int, first_error: int, variant: str, gamma: float, uniform_weight: float) -> list[float]:
    if variant == "vanilla":
        return [1.0] * num_steps
    if variant == "uniform_downweight":
        return [uniform_weight] * num_steps
    if variant == "prefix_masked":
        return [0.0 if k < first_error else 1.0 for k in range(num_steps)]
    if variant == "first_error_only":
        return [1.0 if k == first_error else 0.0 for k in range(num_steps)]
    if variant == "cmdpo":
        return [
            0.0 if k < first_error else (1.0 if k == first_error else gamma ** (k - first_error))
            for k in range(num_steps)
        ]
    raise ValueError(f"Unknown variant: {variant}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="api_math_pairs_20")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--uniform-weight", type=float, default=0.25)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant in ["vanilla", "uniform_downweight", "prefix_masked", "first_error_only", "cmdpo"]:
        variant_rows = []
        for row in rows:
            item = dict(row)
            steps = item["rejected_steps"]
            first_error = int(item["first_error_step"])
            item["step_weights"] = weights_for_variant(
                len(steps),
                first_error,
                variant,
                args.gamma,
                args.uniform_weight,
            )
            item["metadata"] = {**item.get("metadata", {}), "weight_variant": variant, "gamma": args.gamma}
            if variant == "uniform_downweight":
                item["metadata"]["uniform_weight"] = args.uniform_weight
            variant_rows.append(item)
        output = out_dir / f"{args.prefix}_{variant}.jsonl"
        write_jsonl(output, variant_rows)
        print(f"Wrote {len(variant_rows)} rows to {output}")


if __name__ == "__main__":
    main()
