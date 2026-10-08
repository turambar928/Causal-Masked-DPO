#!/usr/bin/env python
"""Produce reviewable tables and paired statistics from per-question predictions."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def length_metrics(rows):
    return {"mean_tokens": float(np.mean([r["prediction_tokens"] for r in rows])),
            "hit_token_limit_rate": float(np.mean([r.get("hit_token_limit", False) for r in rows])),
            "boxed_or_hash_marker_rate": float(np.mean(["####" in r["prediction"] or "\\boxed{" in r["prediction"] for r in rows]))}


def mcnemar_exact(a, b):
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n = n01 + n10
    if n == 0:
        return 1.
    # Python arbitrary-precision integers keep the exact binomial sum stable.
    return min(1., 2 * (sum(math.comb(n, k) for k in range(min(n01, n10) + 1)) / (2**n)))


def paired_bootstrap(delta, seed=20260918, replicates=10000):
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        # Resample paired training seeds AND shared test questions.
        s = rng.integers(delta.shape[0], size=delta.shape[0])
        q = rng.integers(delta.shape[1], size=delta.shape[1])
        draws.append(float(delta[s][:, q].mean()))
    return list(map(float, np.quantile(draws, [.025, .975])))


def holm_adjust(pvalues):
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    maximum = 0.
    for rank, index in enumerate(order):
        maximum = max(maximum, min(1., (len(values) - rank) * values[index]))
        adjusted[index] = maximum
    return adjusted.tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stage1_v1.json")
    p.add_argument("--output-dir", default="outputs/stage1_v1/eval512")
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.output_dir)
    summaries, comparisons, matrices, template_rows = [], [], {}, []

    def record_templates(split, variant, seed, rows):
        for template in sorted({r["metadata"]["template"] for r in rows}):
            selected = [r for r in rows if r["metadata"]["template"] == template]
            template_rows.append({"split": split, "variant": variant, "seed": seed, "template": template,
                                  "correct": sum(r["correct"] for r in selected), "n": len(selected),
                                  "accuracy": float(np.mean([r["correct"] for r in selected]))})
    for split in ["id", "ood", "gsm8k_test"]:
        basepath = out / f"base_{split}_details.jsonl"
        if not basepath.exists():
            continue
        base = read(basepath)
        if split != "gsm8k_test":
            record_templates(split, "base", None, base)
        expected = 1319 if split == "gsm8k_test" else cfg["eval_rows_per_split"]
        if len(base) != expected:
            raise ValueError(f"Incomplete baseline: {split}")
        prompts = [r["prompt"] for r in base]
        matrices[(split, "base")] = np.tile([r["correct"] for r in base], (len(cfg["seeds"]), 1)).astype(float)
        summaries.append({"split": split, "variant": "base", "seeds": 0,
                          "accuracy": np.mean([r["correct"] for r in base]), "std": None, "n": len(base), **length_metrics(base)})
        for variant in cfg["variants"]:
            details = []
            for seed in cfg["seeds"]:
                path = out / f"{variant}_seed{seed}_{split}_details.jsonl"
                if not path.exists():
                    continue
                rows = read(path)
                if [r["prompt"] for r in rows] != prompts:
                    raise ValueError(f"Unpaired evaluation file: {path}")
                details.append(rows)
                if split != "gsm8k_test":
                    record_templates(split, variant, seed, rows)
            if not details:
                continue
            matrix = np.asarray([[r["correct"] for r in rows] for rows in details], dtype=float)
            matrices[(split, variant)] = matrix
            summaries.append({"split": split, "variant": variant, "seeds": len(details),
                              "accuracy": float(matrix.mean()),
                              "std": float(matrix.mean(1).std(ddof=1)) if len(details) > 1 else None,
                              "n": matrix.shape[1], **length_metrics([r for rows in details for r in rows])})
        for left, right in [("cmdpo", "vanilla"), ("cmdpo", "first_error_only"),
                            ("cmdpo", "prefix_masked"), ("cmdpo", "uniform_downweight"),
                            ("cmdpo", "normalized"),
                            ("prefix_masked", "vanilla"), ("process_positive", "cmdpo"),
                            ("cmdpo", "dpo_nll"), ("cmdpo", "sft"),
                            ("vanilla", "base"), ("cmdpo", "base"), ("sft", "base")]:
            x, y = matrices.get((split, left)), matrices.get((split, right))
            if x is None or y is None or x.shape[0] != 3 or y.shape[0] != 3:
                continue
            comparisons.append({"split": split, "left": left, "right": right,
                                "mean_accuracy_difference": float((x-y).mean()),
                                "paired_seed_question_bootstrap_95ci": paired_bootstrap(x-y),
                                "mcnemar_exact_per_seed_unadjusted": [mcnemar_exact(xx, yy) for xx, yy in zip(x, y)],
                                "caution": "Only three training seeds; bootstrap CIs are marginal/exploratory, not simultaneous confidence intervals"})
    for split in ["id", "ood", "gsm8k_test"]:
        group = [r for r in comparisons if r["split"] == split]
        adjusted = holm_adjust([p for row in group for p in row["mcnemar_exact_per_seed_unadjusted"]])
        for i, row in enumerate(group):
            row["mcnemar_per_seed_holm_adjusted_within_split"] = adjusted[i * 3:(i + 1) * 3]
            row["holm_family_size"] = len(adjusted)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "accuracy.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "variant", "seeds", "accuracy", "std", "n", "mean_tokens", "hit_token_limit_rate", "boxed_or_hash_marker_rate"])
        writer.writeheader()
        writer.writerows(summaries)
    with (out / "template_accuracy.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "variant", "seed", "template", "correct", "n", "accuracy"])
        writer.writeheader()
        writer.writerows(template_rows)
    (out / "paired_statistics.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    lines = ["# Stage-one results", "", "Generated from saved per-question predictions. Missing jobs are not counted as completed.", "",
             "| Split | Variant | Completed seeds | Accuracy | Seed SD | Questions | Mean tokens | Length limit |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        sd = "—" if r["std"] is None else f"{r['std']:.4f}"
        lines.append(f"| {r['split']} | {r['variant']} | {r['seeds']} | {r['accuracy']:.4f} | {sd} | {r['n']} | {r['mean_tokens']:.1f} | {r['hit_token_limit_rate']:.1%} |")
    lines += ["", "## Paired contrasts", ""]
    for r in comparisons:
        lo, hi = r["paired_seed_question_bootstrap_95ci"]
        lines.append(f"- {r['split']}: {r['left']} minus {r['right']} = {r['mean_accuracy_difference']:.4f}, paired bootstrap 95% CI [{lo:.4f}, {hi:.4f}].")
    lines += ["", "Three seeds remain a limited uncertainty estimate. Bootstrap intervals are marginal; exact per-seed McNemar p-values and Holm correction across all reported contrasts within each split are saved in paired_statistics.json. For base contrasts, the same fixed base predictions are paired with each seed, not counted as independent base runs. Full-test GSM8K results, if present, measure transfer from arithmetic training, not GSM8K preference training."]
    probe_summaries = []
    lines += ["", "## Held-out per-token likelihood changes", "",
              "| Split | Variant | Seeds | Prefix delta | Error delta | Suffix delta |",
              "|---|---|---:|---:|---:|---:|"]
    for split in ["id", "ood"]:
        for variant in cfg["variants"]:
            probes = [read(path)[0] for seed in cfg["seeds"]
                      if (path := out / f"{variant}_seed{seed}_{split}_probe.jsonl").exists()]
            if not probes:
                continue
            row = {"split": split, "variant": variant, "seeds": len(probes)}
            for metric in ["prefix_delta", "error_delta", "suffix_delta", "cmdpo_delta"]:
                row[metric] = float(np.mean([p[metric] for p in probes]))
                row[metric + "_std"] = float(np.std([p[metric] for p in probes], ddof=1)) if len(probes) > 1 else None
            probe_summaries.append(row)
            lines.append(f"| {split} | {variant} | {len(probes)} | {row['prefix_delta']:.4f} | {row['error_delta']:.4f} | {row['suffix_delta']:.4f} |")
    (out / "probe_summary.json").write_text(json.dumps(probe_summaries, indent=2) + "\n")
    lines += ["", "Probe sample: first 128 held-out examples per split; empty spans excluded (OOD prefix: 64 classroom examples only; recipe first errors occur at step 0 and have no prefix). Positive delta means higher likelihood; negative error delta means error suppression under the supplied rejected context, not necessarily fewer errors in free generation."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(summaries)} result rows and {len(comparisons)} paired contrasts")


if __name__ == "__main__":
    main()
