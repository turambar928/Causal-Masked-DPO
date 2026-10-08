#!/usr/bin/env python
"""Quantify the paired base-model 256/512-token preflight difference."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.summarize_stage1 import read, mcnemar_exact, paired_bootstrap


def main():
    root = Path("outputs/stage1_v1")
    results = []
    for split in ["id", "ood"]:
        short = read(root / f"base_{split}_details.jsonl")
        long = read(root / "eval512" / f"base_{split}_details.jsonl")
        if [r["prompt"] for r in short] != [r["prompt"] for r in long]:
            raise ValueError("Preflight questions differ; cannot compare")
        a = np.array([r["correct"] for r in short], dtype=int)
        b = np.array([r["correct"] for r in long], dtype=int)
        results.append({"split": split, "questions": len(a), "accuracy_256": float(a.mean()),
                        "accuracy_512": float(b.mean()), "difference": float((b-a).mean()),
                        "paired_question_bootstrap_95ci": paired_bootstrap((b-a)[None, :]),
                        "mcnemar_exact_p": mcnemar_exact(a, b),
                        "wrong_to_correct": int(np.sum((a == 0) & (b == 1))),
                        "correct_to_wrong": int(np.sum((a == 1) & (b == 0))),
                        "hit_token_limit_512": sum(r["hit_token_limit"] for r in long),
                        "interpretation": "Base model only, identical prompts and greedy batch32; generation budget preflight, not a masking-method comparison"})
    path = root / "eval512" / "generation_budget_control.json"
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
