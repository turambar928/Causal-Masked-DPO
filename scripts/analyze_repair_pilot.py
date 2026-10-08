#!/usr/bin/env python
"""Complete-matrix validation and frozen inference for the repair pilot."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import rankdata
from cmdpo.data import read_jsonl
from scripts.repair_pilot import (build_data, check_frozen, completed_rows, continuation_ids,
                                 jobs_for, save, sha, write_rows)

CONTRASTS = {"C-B": (0, 1), "R-B": (2, 1), "R-S": (2, 3)}


def resampled_questions(templates, rng):
    return np.concatenate([rng.choice(indices, size=len(indices), replace=True)
                           for label in sorted(set(templates))
                           for indices in [np.flatnonzero(np.asarray(templates) == label)]])


def interval(values, level):
    alpha = (1-level)/2
    return list(map(float, np.quantile(values, [alpha, 1-alpha])))


def paired_bootstrap(delta, templates, level, seed, replicates):
    # One row per question; both positions and every within-question pairing stay together.
    delta = np.asarray(delta, dtype=float)
    assert delta.shape == (len(templates), 2)
    rng = np.random.default_rng(seed)
    means = np.asarray([delta[resampled_questions(templates, rng)].mean() for _ in range(replicates)])
    return interval(means, level)


def residual_rank_correlation(gains, templates):
    # gains: question x position x independent sampling bank.
    centered = np.asarray(gains, dtype=float).copy()
    for label in sorted(set(templates)):
        indices = np.flatnonzero(np.asarray(templates) == label)
        centered[indices] -= centered[indices].mean(axis=0, keepdims=True)
    left, right = [rankdata(centered[:, :, bank].flatten(), method="average") for bank in (0, 1)]
    if left.std() == 0 or right.std() == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def reproducibility(gains, templates, seed, replicates):
    rho = residual_rank_correlation(gains, templates)
    if rho is None:
        return {"rho": None, "ci": None, "valid_bootstraps": 0}
    rng, draws = np.random.default_rng(seed), []
    for _ in range(replicates):
        indices = resampled_questions(templates, rng)
        value = residual_rank_correlation(gains[indices], np.asarray(templates)[indices])
        if value is not None:
            draws.append(value)
    # Constant bootstrap draws cannot manufacture a passing gate.
    ci = interval(draws, .95) if len(draws) == replicates else None
    return {"rho": rho, "ci": ci, "valid_bootstraps": len(draws)}


def decide(cfg, rates, truncation, comparisons):
    if max(truncation.values()) > cfg["max_truncation_rate"]:
        return "protocol_uninterpretable"
    if rates["C"] < cfg["min_clean_success"]:
        return "small_model_capability_insufficient"
    gate = comparisons[cfg["gate_reproducibility_contrast"]]["reproducibility"]
    gains_supported = all(row["ci"][0] > 0 for row in comparisons.values())
    magnitude = comparisons["R-S"]["difference"] >= cfg["min_repair_vs_sham_gain"]
    stable = gate["ci"] is not None and gate["ci"][0] > 0
    if gains_supported and magnitude and stable:
        return "worth_followup_not_training_evidence"
    if gains_supported and magnitude:
        return "repair_gain_without_stable_heterogeneity"
    return "no_reliable_full_mechanism_support"


def csv_rows(path, rows):
    with Path(path).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(rows):
    return {"n": len(rows), "success": float(np.mean([r["correct"] for r in rows])),
            "mean_tokens": float(np.mean([r["generated_tokens"] for r in rows])),
            "truncation": float(np.mean([r["hit_token_limit"] for r in rows])),
            "explicit_answer": float(np.mean([r["has_explicit_answer"] for r in rows]))}


def analyze(cfg, config_path):
    from transformers import AutoTokenizer
    out, data = Path(cfg["output_dir"]), Path(cfg["data_dir"])
    provenance = check_frozen(cfg, config_path)
    questions = read_jsonl(data / "questions.jsonl")
    cases = read_jsonl(data / "cases.jsonl")
    excluded = {r["prompt"] for p in cfg["excluded_data"] for r in read_jsonl(p)}
    assert (questions, cases) == build_data(cfg, excluded), "Dataset is not reproducible"
    assert len({q["prompt"] for q in questions}) == 96
    assert not ({q["prompt"] for q in questions} & excluded)
    manifest = json.loads((out / "rollout_manifest.json").read_text())
    assert manifest["config_sha256"] == provenance["config_sha256"]
    expected_jobs = jobs_for(questions, cases, cfg)
    assert len(expected_jobs) == len(manifest["jobs"]) == 432
    assert set(manifest["jobs"]) == {k for k, _ in expected_jobs}
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
    rows, artifacts = [], {}
    for job_id, specs in expected_jobs:
        entry = manifest["jobs"][job_id]
        batch = completed_rows(entry, specs, cfg, job_id)
        artifacts[entry["path"]] = entry["sha256"]
        for row in batch:
            tokens = continuation_ids(tokenizer, row["prompt"], row["prefix"])
            assert row["input_ids_sha256"] == hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
            ids = row["generated_ids"]
            assert len(ids) == row["generated_tokens"]
            assert tokenizer.decode(ids, skip_special_tokens=True) == row["prediction"]
            assert row["stopped_on_eos"] == (ids[-1] == tokenizer.eos_token_id)
            assert tokenizer.eos_token_id not in ids[:-1]
            assert row["hit_token_limit"] == (not row["stopped_on_eos"] and len(ids) == cfg["max_new_tokens"])
        rows.extend(batch)
    assert len(rows) == 13824
    qindex = {q["question_id"]: i for i, q in enumerate(questions)}
    cindex = {c: i for i, c in enumerate(cfg["conditions"])}
    matrix = np.full((96, 2, 4, 2, 8), np.nan)
    for row in rows:
        if row["condition"] == "F":
            continue
        key = (qindex[row["question_id"]], row["position"], cindex[row["condition"]], row["bank"], row["sample"])
        assert np.isnan(matrix[key]), "Duplicate sample"
        matrix[key] = row["correct"]
    assert np.isfinite(matrix).all()
    summaries = [{"condition": c, **metrics([r for r in rows if r["condition"] == c])}
                 for c in [*cfg["conditions"], "F"]]
    templates = np.asarray([q["template"] for q in questions])
    grouped, scenario_rows = [], []
    for t in sorted(set(templates)):
        for p in [0, 1]:
            for c in cfg["conditions"]:
                selected = [r for r in rows if r["template"] == t and r["position"] == p and r["condition"] == c]
                grouped.append({"template": t, "position": p+1, "condition": c, **metrics(selected)})
    probabilities = matrix.mean(axis=-1)  # question x position x condition x bank
    comparisons = {}
    for name, (left, right) in CONTRASTS.items():
        gain_banks = probabilities[:, :, left] - probabilities[:, :, right]
        delta = gain_banks.mean(axis=-1)
        comparisons[name] = {"difference": float(delta.mean()), "ci_level": cfg["primary_ci_level"],
                             "ci": paired_bootstrap(delta, templates, cfg["primary_ci_level"], cfg["seed"], cfg["bootstrap_replicates"]),
                             "bank_differences": gain_banks.mean(axis=(0, 1)).tolist(),
                             "reproducibility": reproducibility(gain_banks, templates, cfg["seed"], cfg["bootstrap_replicates"])}
    for i, q in enumerate(questions):
        for pos in (0, 1):
            result = {"question_id": q["question_id"], "template": q["template"], "position": pos+1}
            for c, ci in cindex.items():
                for bank in (0, 1):
                    result[f"{c}_bank{bank}"] = float(probabilities[i, pos, ci, bank])
            scenario_rows.append(result)
    rates = {r["condition"]: r["success"] for r in summaries}
    truncation = {r["condition"]: r["truncation"] for r in summaries}
    decision = decide(cfg, rates, truncation, comparisons)
    result = {"decision": decision, "comparisons": comparisons,
              "assay": {"rates": rates, "truncation": truncation},
              "gate_reproducibility_contrast": cfg["gate_reproducibility_contrast"],
              "caution": "Question-cluster bootstrap stratified by template; 3 Bonferroni-adjusted primary intervals. Correlation CI is an exploratory investment gate, not a simultaneous test. R supplies a correct repair; it is not autonomous self-correction. S is an active sham. No training or novelty claim."}
    save(out / "statistics.json", result)
    csv_rows(out / "conditions.csv", summaries)
    csv_rows(out / "template_position.csv", grouped)
    csv_rows(out / "scenario_probabilities.csv", scenario_rows)
    write_rows(out / "predictions.jsonl", rows)
    artifacts[str(out / "predictions.jsonl")] = sha(out / "predictions.jsonl")
    validation = {"status": "passed", "jobs": 432, "questions": 96, "cases": 192, "predictions": len(rows),
                  "excluded_questions": len(excluded), "artifact_sha256": artifacts,
                  "config_sha256": sha(config_path), "analysis_sha256": sha(__file__),
                  "decision": decision, "note": "Integrity passing is separate from the scientific gate."}
    save(out / "validation.json", validation)
    plot(out, summaries, probabilities, templates, comparisons)
    lines = ["# Repair intervention pilot", "", f"Decision: `{decision}`.", "",
             "96 questions, 192 error scenarios, 432 batches, 13,824 continuations. Local frozen Qwen; no API or training.", "",
             "| Condition | Success | Mean tokens | Truncation | Explicit answer |", "|---|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['condition']} | {r['success']:.2%} | {r['mean_tokens']:.1f} | {r['truncation']:.2%} | {r['explicit_answer']:.2%} |")
    lines.extend(["", "## Frozen contrasts", ""])
    for name, row in comparisons.items():
        lo, hi = row["ci"]
        rep = row["reproducibility"]
        lines.append(f"- {name}: {100*row['difference']:+.2f} pp; {100*row['ci_level']:.3f}% CI [{100*lo:+.2f}, {100*hi:+.2f}]. Residual cross-bank Spearman: {rep['rho']}; 95% CI {rep['ci']}.")
    lines.extend(["", result["caution"], "", "R-S reproducibility is the only heterogeneity gate. Template/position centering does not remove all difficulty confounds. These continuations do not show whether learning a repair improves free generation."])
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"validation": "passed", "decision": decision, "conditions": summaries, "comparisons": comparisons}, indent=2))


def plot(out, summaries, probabilities, templates, comparisons):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    labels = ["C: clean", "B: wrong", "R: repair", "S: sham", "F: free"]
    axes[0].bar(labels, [r["success"]*100 for r in summaries], color=["#579e91", "#bd8576", "#6e9fc2", "#999999", "#8b76ab"])
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Continuation success (%)")
    axes[0].tick_params(axis="x", labelrotation=20)
    gains = probabilities[:, :, 2] - probabilities[:, :, 3]
    for label in sorted(set(templates)):
        ix = np.flatnonzero(templates == label)
        gains[ix] -= gains[ix].mean(axis=0, keepdims=True)
    axes[1].scatter(gains[:, :, 0].flatten(), gains[:, :, 1].flatten(), s=15, alpha=.45)
    axes[1].axhline(0, color="gray", lw=.7)
    axes[1].axvline(0, color="gray", lw=.7)
    axes[1].set_xlabel("R-S residual gain: independent bank 0")
    axes[1].set_ylabel("R-S residual gain: independent bank 1")
    rho = comparisons["R-S"]["reproducibility"]["rho"]
    axes[1].set_title(f"Template/position-centered Spearman: {rho:.3f}" if rho is not None else "Undefined rank correlation")
    fig.suptitle("Frozen Qwen2.5-0.5B · repair pilot · no training\n96 questions; 2 error positions; 16 continuations per condition", fontsize=11)
    fig.savefig(out / "diagnostic.png", dpi=180)
    fig.savefig(out / "diagnostic.pdf")
    plt.close(fig)
