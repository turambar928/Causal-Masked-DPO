#!/usr/bin/env python
"""Describe objective scales without changing the frozen training experiment."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from transformers import AutoTokenizer

from cmdpo.collator import CMDPOCollator
from cmdpo.data import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_v1.json")
    parser.add_argument("--output-dir", default="outputs/stage1_v1/eval512")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
    collator = CMDPOCollator(tokenizer, cfg["max_length"], use_chat_template=True, strict_length=True)
    results = []
    for variant in cfg["variants"]:
        rows = read_jsonl(Path(cfg["data_dir"]) / f"train_{variant}.jsonl")
        masses = {"chosen": [], "rejected": [], "positive": []}
        for start in range(0, len(rows), 64):
            batch = collator(rows[start:start + 64])
            for side in masses:
                key = f"{side}_response_mask"
                if key in batch:
                    masses[side].extend(batch[key][:, 1:].sum(-1).tolist())
        record = {"variant": variant, "examples": len(rows)}
        for side, values in masses.items():
            if values:
                record[f"{side}_mask_mass"] = {"mean": float(np.mean(values)),
                                                "min": float(np.min(values)), "max": float(np.max(values))}
        if variant == "normalized":
            record["note"] = "Rejected sum is divided by its mask mass (clamped at 1); chosen remains an unnormalized sum."
        elif variant == "sft":
            record["note"] = "Only chosen per-token NLL is optimized; rejected masks are unused."
        results.append(record)
    output = Path(args.output_dir) / "mask_mass_summary.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
