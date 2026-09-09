#!/usr/bin/env python
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "iclr2027" / "figures"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 180,
            "savefig.dpi": 240,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"Wrote {path}")


def plot_gamma_sweep() -> None:
    rows = read_jsonl(ROOT / "outputs/qwen0_5b_harder2000_cmdpo_gamma_ablation_likelihood_deltas.jsonl")
    gamma_map = {
        "gamma0_first_error_only": 0.0,
        "gamma025": 0.25,
        "gamma05": 0.5,
        "gamma075": 0.75,
        "gamma1_prefix_masked": 1.0,
    }
    rows = sorted(rows, key=lambda r: gamma_map[r["variant"]])
    gammas = [gamma_map[r["variant"]] for r in rows]
    prefix = [r["prefix_delta"] for r in rows]
    error = [r["error_delta"] for r in rows]
    suffix = [r["suffix_delta"] for r in rows]

    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.plot(gammas, prefix, marker="o", linewidth=2, label="Prefix delta")
    ax.plot(gammas, error, marker="o", linewidth=2, label="Error delta")
    ax.plot(gammas, suffix, marker="o", linewidth=2, label="Suffix delta")
    ax.set_xlabel(r"Suffix decay $\gamma$")
    ax.set_ylabel("Masked log-likelihood change")
    ax.set_xticks(gammas)
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.set_title("Causal masking tradeoff across suffix schedules")
    save(fig, "gamma_sweep.png")


def grouped_barh(ax: plt.Axes, rows: list[dict[str, Any]], title: str, show_legend: bool = False) -> None:
    metrics = [
        ("prefix_delta", "Prefix", "#4C78A8"),
        ("error_delta", "Error", "#F58518"),
        ("suffix_delta", "Suffix", "#54A24B"),
    ]
    labels = [r["variant"] for r in rows]
    y = np.arange(len(rows))
    h = 0.22
    offsets = [-h, 0.0, h]
    for offset, (key, label, color) in zip(offsets, metrics, strict=True):
        values = [r[key] for r in rows]
        ax.barh(y + offset, values, height=h, color=color, label=label)
    ax.axvline(0, color="0.5", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel("Masked log-likelihood change")
    if show_legend:
        ax.legend(frameon=False, ncol=3, loc="lower right")


def plot_control_panels() -> None:
    reviewer = read_jsonl(ROOT / "outputs/qwen0_5b_harder2000_reviewer_baselines_likelihood_deltas.jsonl")
    normalization = read_jsonl(ROOT / "outputs/qwen0_5b_harder2000_normalization_likelihood_deltas.jsonl")
    localization = read_jsonl(ROOT / "outputs/qwen0_5b_harder2000_localization_noise_likelihood_deltas.jsonl")
    process_positive = read_jsonl(ROOT / "outputs/qwen0_5b_harder2000_process_positive_likelihood_deltas.jsonl")

    order_maps = {
        "reviewer": ["truncated_rejected", "clean_cmdpo", "shift_plus_one"],
        "normalization": ["clean_cmdpo", "cmdpo_normalized", "uniform_downweight025"],
        "localization": ["shift_minus_one", "clean_cmdpo", "shift_plus_one", "random25"],
        "process": ["clean_cmdpo", "process_positive02"],
    }
    label_maps = {
        "truncated_rejected": "Truncated rejected",
        "clean_cmdpo": "Clean CM-DPO",
        "shift_plus_one": "Shift +1",
        "cmdpo_normalized": "Normalized CM-DPO",
        "uniform_downweight025": "Uniform downweight",
        "shift_minus_one": "Shift -1",
        "random25": "Random 25%",
        "process_positive02": "Process-positive",
    }

    def reorder(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        by_variant = {r["variant"]: r for r in rows}
        out = []
        for variant in order_maps[key]:
            item = dict(by_variant[variant])
            item["variant"] = label_maps.get(variant, variant)
            out.append(item)
        return out

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0))
    panels = [
        (axes[0, 0], reorder(reviewer, "reviewer"), "Reviewer baselines"),
        (axes[0, 1], reorder(normalization, "normalization"), "Normalization controls"),
        (axes[1, 0], reorder(localization, "localization"), "Localization noise"),
        (axes[1, 1], reorder(process_positive, "process"), "Process-positive"),
    ]
    for ax, rows, title in panels:
        grouped_barh(ax, rows, title, show_legend=(title == "Reviewer baselines"))
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Diagnostics that isolate causal masking from alternative explanations", y=1.02)
    save(fig, "control_panels.png")


def plot_scale_accuracy() -> None:
    rows_05 = read_jsonl(ROOT / "outputs/qwen0_5b_harder2000_gamma025_multiseed_eval500_accuracy.jsonl")
    rows_15 = read_jsonl(ROOT / "outputs/qwen1_5b_harder_eval500_accuracy.jsonl")

    def mean_std(rows: list[dict[str, Any]], prefix: str, models: list[str]) -> tuple[list[float], list[float]]:
        values = {m: [] for m in models}
        for row in rows:
            name = row["model"]
            for model in models:
                if name == model or name.endswith(f"_{model}"):
                    values[model].append(row["accuracy"])
        means = [float(np.mean(values[m])) if values[m] else np.nan for m in models]
        stds = [float(np.std(values[m], ddof=1)) if len(values[m]) > 1 else 0.0 for m in models]
        return means, stds

    models_05 = ["base", "vanilla", "prefix_masked", "first_error_only", "cmdpo"]
    means_05, stds_05 = mean_std(rows_05, "seed", models_05)
    rows_15_map = {r["model"]: r for r in rows_15}
    models_15 = ["base", "vanilla", "cmdpo"]
    means_15 = [rows_15_map[m]["accuracy"] for m in models_15]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=False)

    def panel(ax: plt.Axes, models: list[str], means: list[float], stds: list[float] | None, title: str) -> None:
        y = np.arange(len(models))
        ax.barh(y, means, xerr=stds, color="#4C78A8", alpha=0.9, capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels([m.replace("_", " ") for m in models])
        ax.set_xlim(0, 0.55)
        ax.set_xlabel("Accuracy")
        ax.set_title(title)
        ax.grid(axis="x")
        for yi, val in zip(y, means, strict=True):
            ax.text(min(val + 0.01, 0.53), yi, f"{val:.3f}", va="center", fontsize=8)

    panel(axes[0], models_05, means_05, stds_05, "Qwen2.5-0.5B-Instruct (mean over 3 seeds)")
    panel(axes[1], models_15, means_15, None, "Qwen2.5-1.5B-Instruct")
    fig.suptitle("Generation accuracy by scale")
    save(fig, "scale_accuracy.png")


def main() -> None:
    setup_style()
    plot_gamma_sweep()
    plot_control_panels()
    plot_scale_accuracy()


if __name__ == "__main__":
    main()
