#!/usr/bin/env python
"""Build a deduplicated, frozen template-disjoint mechanism experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmdpo.data import write_jsonl
from cmdpo.localization import build_cm_weights
from scripts.build_harder_math_pairs import TEMPLATES
from scripts.make_weight_variants import weights_for_variant


class IndependentParameters:
    """Supply independent uniform draws to the legacy templates' `seed % n` slots.

    Reusing a single integer in all slots restricts some templates to only 120
    distinct questions. The text, arithmetic, and injected errors are unchanged.
    """
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def __mod__(self, upper):
        return self.rng.randrange(upper)


def build_splits(train_size=2000, eval_size=500):
    rng = random.Random(20260918)
    seen = set()
    groups = {"train": TEMPLATES[:3] + TEMPLATES[5:],
              "id": TEMPLATES[:3] + TEMPLATES[5:], "ood": TEMPLATES[3:5]}
    result = {}
    for split, templates in groups.items():
        rows = []
        attempts = 0
        target = train_size if split == "train" else eval_size
        while len(rows) < target:
            attempts += 1
            if attempts > 1000000:
                raise RuntimeError("Requested more unique questions than the templates provide")
            pair = templates[len(rows) % len(templates)](IndependentParameters(rng.randrange(10**12)))
            if pair.question in seen:
                continue
            seen.add(pair.question)
            prompt = f"Question: {pair.question}\nAnswer step by step and end with '#### <answer>'."
            rows.append({"prompt": prompt, "chosen": "\n".join(pair.chosen_steps),
                         "rejected": "\n".join(pair.rejected_steps), "answer": f"#### {pair.answer}",
                         "rejected_steps": pair.rejected_steps, "first_error_step": pair.first_error_step,
                         "step_weights": build_cm_weights(len(pair.rejected_steps), pair.first_error_step, 0.25),
                         "metadata": {"source": "stage1_template_oracle", "split": split,
                                      "template": pair.template, "row_id": len(rows),
                                      "sample_id": hashlib.sha256(prompt.encode()).hexdigest()}})
        result[split] = rows
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/processed/stage1_v1")
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--eval-size", type=int, default=500)
    args = parser.parse_args()
    out = Path(args.output_dir)
    if (out / "manifest.json").exists():
        raise FileExistsError("Dataset already frozen; use a new output directory")
    splits = build_splits(args.train_size, args.eval_size)
    manifest = {"seed": 20260918, "gamma": 0.25, "parameter_sampling": "independent_uniform_per_slot",
                "splits": {}, "variants": {}}
    for name, rows in splits.items():
        path = out / f"{name}.jsonl"
        write_jsonl(path, rows)
        manifest["splits"][name] = {"rows": len(rows), "templates": sorted({r["metadata"]["template"] for r in rows}),
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for variant in ["sft", "vanilla", "dpo_nll", "prefix_masked", "first_error_only", "cmdpo",
                    "process_positive", "normalized", "uniform_downweight"]:
        mask = "vanilla" if variant in ["sft", "dpo_nll"] else variant
        if variant in ["process_positive", "normalized"]:
            mask = "cmdpo"
        rows = []
        for source in splits["train"]:
            row = dict(source)
            row["step_weights"] = weights_for_variant(len(row["rejected_steps"]), row["first_error_step"], mask, .25, .25)
            if variant == "process_positive":
                row["positive_prefix"] = "\n".join(row["rejected_steps"][:row["first_error_step"]])
            rows.append(row)
        path = out / f"train_{variant}.jsonl"
        write_jsonl(path, rows)
        manifest["variants"][variant] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
