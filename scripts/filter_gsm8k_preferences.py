#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmdpo.data import read_jsonl, write_jsonl


def looks_truncated(text: str) -> bool:
    stripped = text.rstrip()
    if len(stripped) < 80:
        return True
    if stripped.endswith(("=", "+", "-", "*", "/", ":", ",")):
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.9)
    parser.add_argument("--require-not-last", action="store_true")
    parser.add_argument("--require-sampled-chosen", action="store_true")
    parser.add_argument("--drop-truncated", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    kept = []
    stats = {
        "low_confidence": 0,
        "last_error": 0,
        "gold_fallback_chosen": 0,
        "truncated": 0,
    }
    for row in rows:
        if float(row.get("localization_confidence", 0.0)) < args.min_confidence:
            stats["low_confidence"] += 1
            continue
        if args.require_not_last and int(row["first_error_step"]) >= len(row["rejected_steps"]) - 1:
            stats["last_error"] += 1
            continue
        if args.require_sampled_chosen and not bool(row.get("metadata", {}).get("has_sampled_chosen")):
            stats["gold_fallback_chosen"] += 1
            continue
        if args.drop_truncated and looks_truncated(row["rejected"]):
            stats["truncated"] += 1
            continue
        kept.append(row)

    write_jsonl(args.output, kept)
    print(f"Wrote {len(kept)} / {len(rows)} rows to {args.output}")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
