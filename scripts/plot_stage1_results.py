#!/usr/bin/env python
"""Export complete three-seed results; error bars are across-seed standard deviations."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/stage1_v1/eval512")
    a = p.parse_args()
    out = Path(a.output_dir)
    rows = list(csv.DictReader((out / "accuracy.csv").open()))
    variants = ["vanilla", "prefix_masked", "first_error_only", "cmdpo", "sft", "dpo_nll",
                "process_positive", "normalized", "uniform_downweight"]
    labels = ["DPO", "Prefix\nmask", "First\nerror", "CM-DPO", "SFT", "DPO\n+NLL",
              "CM+proc.\npositive", "Norm.\nCM", "Uniform\n0.25"]
    colors = ["#8f969f", "#5c91bb", "#5baf9b", "#d68541", "#7b69a7", "#b7749c",
              "#ae8561", "#8a9d52", "#6988a0"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True, constrained_layout=True)
    for ax, split, title in zip(axes, ["id", "ood"], ["Held-out questions: seen templates", "Held-out questions: unseen templates"]):
        selected = {r["variant"]: r for r in rows if r["split"] == split}
        if any(int(selected[v]["seeds"]) != 3 for v in variants):
            raise ValueError("All three seeds must finish before exporting the final figure")
        means = [100 * float(selected[v]["accuracy"]) for v in variants]
        errors = [100 * float(selected[v]["std"]) for v in variants]
        ax.bar(range(len(variants)), means, color=colors, alpha=.85, width=.7)
        ax.errorbar(range(len(variants)), means, yerr=errors, fmt="none", ecolor="#333333", capsize=3, linewidth=1)
        base = 100 * float(selected["base"]["accuracy"])
        ax.axhline(base, linestyle="--", color="#333333", linewidth=1, label=f"Base: {base:.1f}%")
        for i, v in enumerate(variants):
            per_seed = [json.loads((out / f"{v}_seed{s}_{split}_summary.jsonl").read_text())["accuracy"] * 100 for s in [1, 2, 3]]
            ax.scatter(i + np.array([-.12, 0, .12]), per_seed, s=10, c="#333333", zorder=3)
        ax.set_xticks(range(len(variants)), labels, fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper left", fontsize=9)
    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle("Qwen2.5-0.5B-Instruct · 2,000 training pairs · 3 seeds · 512-token greedy evaluation\nBars: mean; error bars: seed SD; dots: individual seeds; 500 questions per split", fontsize=10)
    fig.savefig(out / "accuracy.png", dpi=200)
    fig.savefig(out / "accuracy.pdf")
    plt.close(fig)
    print(f"Saved {out / 'accuracy.png'} and accuracy.pdf")


if __name__ == "__main__":
    main()
