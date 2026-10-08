#!/usr/bin/env python
"""Validate the entire frozen stage-two matrix before reporting final statistics."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from cmdpo.data import read_jsonl
from cmdpo.verifier import verify_answer
from scripts.stage2 import check_frozen, save, sha
from scripts.summarize_stage1 import mcnemar_exact, holm_adjust, length_metrics


def bootstrap(delta, level, seed=20260930, replicates=10000):
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 2 or not delta.size or not 0 < level < 1:
        raise ValueError("Expected seed-by-question paired differences and a valid CI level")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        seeds = rng.integers(delta.shape[0], size=delta.shape[0])
        questions = rng.integers(delta.shape[1], size=delta.shape[1])
        draws.append(float(delta[seeds][:, questions].mean()))
    tail = (1 - level) / 2
    return list(map(float, np.quantile(draws, [tail, 1-tail])))


def write_csv(path, rows):
    with Path(path).open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_v1.json")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg["output_dir"])
    out = root / "eval512"
    provenance = check_frozen(cfg, args.config)
    previous_validation = Path(cfg["stage1_dir"]) / "eval512/validation.json"
    assert sha(previous_validation) == provenance["stage1_validation_sha256"]
    assert sha(Path(cfg["stage1_dir"]) / "eval512/reproducibility_source.tar.gz") == provenance["stage1_snapshot_sha256"]
    assert sha(Path(cfg["stage1_dir"]) / "reference.pt") == provenance["reference_cache_sha256"]
    assert sha(Path(cfg["model"]) / "model.safetensors") == provenance["model_sha256"]
    training = json.loads((root / "training_manifest.json").read_text())
    assert training["config_sha256"] == provenance["config_sha256"]
    artifacts = {}
    for variant in cfg["variants"]:
        for seed in cfg["seeds"]:
            name = f"{variant}_seed{seed}"
            if variant in cfg["new_variants"]:
                job = training["jobs"][name]
                assert job["status"] == "complete", name
                folder = root / "adapters" / name
                entry = job
            else:
                entry = provenance["reused"][name]
                folder = Path(entry["path"])
            for path, checksum in entry["files"].items():
                assert sha(path) == checksum, path
                artifacts[path] = checksum
            run = json.loads((folder / "run_config.json").read_text())
            state = json.loads((folder / "trainer_state.json").read_text())
            metrics = json.loads((folder / "train_metrics.json").read_text())
            assert run["seed"] == seed and run["use_chat_template"] and run["strict_length"], name
            assert state["global_step"] == 250 and metrics["epoch"] == cfg["epochs"] == 1, name
            for actual, expected in [("learning_rate", "learning_rate"), ("beta", "beta"), ("max_length", "max_length"),
                                     ("per_device_train_batch_size", "train_batch_size"), ("gradient_accumulation_steps", "gradient_accumulation_steps")]:
                assert run[actual] == cfg[expected], (name, actual)
            assert run["objective"] == ("sft" if variant == "sft" else "dpo")
            assert not run["normalize_rejected"] and run["chosen_nll_weight"] == run["process_positive_weight"] == 0
            if variant in cfg["new_variants"]:
                assert run["data_sha256"] == sha(Path(cfg["data_dir"]) / f"train_{variant}.jsonl")
            lora = json.loads((folder / "adapter_config.json").read_text())
            assert lora["r"] == 16 and lora["lora_alpha"] == 32 and lora["lora_dropout"] == .05
            for event in state["log_history"]:
                for key in ["loss", "grad_norm"]:
                    if key in event:
                        assert math.isfinite(event[key]), (name, key)
    jobs = {}
    for path in out.glob("evaluation*manifest.json"):
        manifest = json.loads(path.read_text())
        assert manifest["config_sha256"] == provenance["config_sha256"]
        for name, job in manifest["jobs"].items():
            assert job["status"] == "complete", (name, job["status"])
            if name in jobs:
                assert jobs[name]["files"] == job["files"], name
            jobs[name] = job
    names = ["base"] + [f"{v}_seed{s}" for v in cfg["variants"] for s in cfg["seeds"]]
    assert len(jobs) == len(names) * 2 == 44
    old_questions = {r["prompt"] for split in ["train", "id", "ood"] for r in read_jsonl(Path(cfg["stage1_data_dir"]) / f"{split}.jsonl")}
    new_questions = set()
    matrices, summaries, per_seed, per_template = {}, [], [], []
    predictions = 0
    for split in ["id", "ood"]:
        gold = read_jsonl(Path(cfg["data_dir"]) / f"{split}.jsonl")
        assert len(gold) == cfg["eval_rows_per_split"]
        for row in gold:
            assert row["prompt"] not in old_questions | new_questions
            new_questions.add(row["prompt"])
        details = {}
        for name in names:
            job = jobs[f"{name}_{split}"]
            for path, checksum in job["files"].items():
                assert sha(path) == checksum, path
                artifacts[path] = checksum
            rows = read_jsonl(out / f"{name}_{split}_details.jsonl")
            assert len(rows) == len(gold)
            for i, (row, expected) in enumerate(zip(rows, gold)):
                assert row["model"] == name and row["index"] == i
                assert row["prompt"] == expected["prompt"] and row["gold"] == expected["answer"]
                assert row["metadata"] == expected["metadata"]
                assert row["correct"] == verify_answer(row["prediction"], row["gold"])
                assert 0 < row["prediction_tokens"] <= cfg["max_new_tokens"]
                assert row["stopped_on_eos"] != row["hit_token_limit"]
            summary = read_jsonl(out / f"{name}_{split}_summary.jsonl")[0]
            assert summary["correct"] == sum(r["correct"] for r in rows)
            assert summary["total"] == len(rows) and summary["accuracy"] == summary["correct"] / len(rows)
            details[name] = rows
            predictions += len(rows)
            variant, seed = ("base", None) if name == "base" else (name.rsplit("_seed", 1)[0], int(name.rsplit("_seed", 1)[1]))
            per_seed.append({"split": split, "variant": variant, "seed": seed, "accuracy": summary["accuracy"], **length_metrics(rows)})
            for template in sorted({r["metadata"]["template"] for r in rows}):
                selected = [r for r in rows if r["metadata"]["template"] == template]
                per_template.append({"split": split, "variant": variant, "seed": seed, "template": template,
                                     "n": len(selected), "accuracy": float(np.mean([r["correct"] for r in selected]))})
        for variant in ["base", *cfg["variants"]]:
            selected = [details["base"]] if variant == "base" else [details[f"{variant}_seed{s}"] for s in cfg["seeds"]]
            matrix = np.asarray([[r["correct"] for r in rows] for rows in selected], dtype=float)
            matrices[(split, variant)] = matrix
            summaries.append({"split": split, "variant": variant, "seeds": 0 if variant == "base" else len(selected),
                              "n": len(gold), "accuracy": float(matrix.mean()),
                              "std": None if variant == "base" else float(matrix.mean(1).std(ddof=1)),
                              **length_metrics([r for rows in selected for r in rows])})
    comparisons = []
    for split in ["ood", "id"]:
        for left, right in cfg["primary_contrasts"]:
            x, y = matrices[(split, left)], matrices[(split, right)]
            level = cfg["primary_ci_level"] if split == cfg["primary_split"] else .95
            comparisons.append({"split": split, "left": left, "right": right, "primary": split == cfg["primary_split"],
                                "difference": float((x-y).mean()), "ci_level": level,
                                "ci": bootstrap(x-y, level, cfg["bootstrap_seed"], cfg["bootstrap_replicates"]),
                                "mcnemar_per_seed": [mcnemar_exact(xx, yy) for xx, yy in zip(x, y)]})
    primary = [r for r in comparisons if r["primary"]]
    adjusted = holm_adjust([p for r in primary for p in r["mcnemar_per_seed"]])
    for i, row in enumerate(primary):
        row["mcnemar_holm_primary_six_tests"] = adjusted[3*i:3*(i+1)]
    wins = sum(r["ci"][0] > 0 for r in primary)
    decision = "both_position_comparisons_supported" if wins == 2 else ("only_one_supported" if wins == 1 else "insufficient_advantage_evidence")
    save(out / "paired_statistics.json", {"comparisons": comparisons, "decision": decision,
         "caution": "Two primary OOD comparisons: 97.5% marginal bootstrap CIs for Bonferroni family adjustment; approximate inference with only three seeds. ID 95% intervals are secondary. No CI crossing zero establishes equivalence."})
    write_csv(out / "accuracy.csv", summaries)
    write_csv(out / "seed_accuracy.csv", per_seed)
    write_csv(out / "template_accuracy.csv", per_template)
    result = {"status": "passed", "new_training_runs": 6, "reused_adapters": 15, "generation_runs": 44,
              "predictions": predictions, "fresh_questions": len(new_questions), "stage1_questions_excluded": len(old_questions),
              "artifact_sha256": artifacts, "config_sha256": sha(args.config), "analysis_source_sha256": sha(__file__),
              "scope": "Local Qwen; no API; no new model scale, GSM8K, or hyperparameter search"}
    assert predictions == 22000
    save(out / "validation.json", result)
    lines = ["# Stage-two position-versus-mass diagnostic", "", "All 6 new runs, 15 reused adapters and 44 generation jobs validated (22,000 predictions).", "",
             "| Split | Method | Accuracy (%) | Seed SD (pp) | Mean tokens | Length limit |", "|---|---|---:|---:|---:|---:|"]
    for row in summaries:
        sd = "—" if row["std"] is None else f"{row['std']*100:.2f}"
        lines.append(f"| {row['split']} | {row['variant']} | {row['accuracy']*100:.2f} | {sd} | {row['mean_tokens']:.1f} | {row['hit_token_limit_rate']:.2%} |")
    lines += ["", "## Predeclared paired comparisons", ""]
    for row in comparisons:
        lo, hi = row["ci"]
        lines.append(f"- {row['split']}: {row['left']} minus {row['right']} = {row['difference']*100:+.2f} pp; {row['ci_level']*100:g}% CI [{lo*100:+.2f}, {hi*100:+.2f}].")
    lines += ["", f"Decision: `{decision}`.", "", "Intervals are approximate with three seeds. Weight mass is not gradient-norm matching. Shared-prefix and fixed-error-position limitations remain; no real-task generalization is established. ID intervals are secondary; absence of evidence is not evidence of equivalence."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["DPO", "Prefix\nmask", "CM-DPO", "SFT", "Uniform\n0.25", "Mass\nmatched", "Prefix-safe\nshuffle"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True, constrained_layout=True)
    for ax, split in zip(axes, ["id", "ood"]):
        selected = {r["variant"]: r for r in summaries if r["split"] == split}
        vals = [100*selected[v]["accuracy"] for v in cfg["variants"]]
        stds = [100*selected[v]["std"] for v in cfg["variants"]]
        ax.bar(range(7), vals, color=["#999999", "#6e9fc2", "#d68541", "#8b76ab", "#7797aa", "#579e91", "#bd8576"])
        ax.errorbar(range(7), vals, yerr=stds, fmt="none", color="#333333", capsize=3)
        for i, variant in enumerate(cfg["variants"]):
            ax.scatter(i + np.array([-.12, 0, .12]), matrices[(split, variant)].mean(1)*100, s=13, color="#333333", zorder=3)
        ax.axhline(100*selected["base"]["accuracy"], color="#333333", linestyle="--", label=f"Base {selected['base']['accuracy']:.1%}")
        ax.set_xticks(range(7), labels, fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title("Seen templates (fresh ID)" if split == "id" else "Unseen templates (fresh OOD)")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle("Qwen2.5-0.5B-Instruct · position vs. weight mass · 3 seeds\n500 fresh questions per split · bars: mean; error bars: seed SD; dots: seeds", fontsize=10)
    fig.savefig(out / "accuracy.png", dpi=200)
    fig.savefig(out / "accuracy.pdf")
    plt.close(fig)
    print({k:v for k,v in result.items() if k != "artifact_sha256"})
    print("Primary decision:", decision)


if __name__ == "__main__":
    main()
