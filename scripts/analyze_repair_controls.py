#!/usr/bin/env python
"""Frozen matched-control contrasts and question-held-out incremental prediction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from cmdpo.data import read_jsonl
from scripts.repair_pilot import completed_rows, continuation_ids, grade, sha, save, write_rows
from scripts.repair_controls import check_frozen, make_cases, assign_folds, jobs_for
from scripts.analyze_repair_pilot import metrics, csv_rows, paired_bootstrap, resampled_questions, interval


def matched_interval(delta, mask, templates, cfg):
    rng, draws = np.random.default_rng(cfg["seed"]), []
    for _ in range(cfg["bootstrap_replicates"]):
        ix = resampled_questions(templates, rng)
        draws.append(float(delta[ix][mask[ix]].mean()))
    return interval(draws, .95)


def ridge_predict(train_x, train_y, test_x, alpha, bounds):
    assert alpha > 0
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-12] = 1
    x = (train_x - mean) / scale
    ymean = float(train_y.mean())
    coef = np.linalg.solve(x.T @ x + alpha*np.eye(x.shape[1]), x.T @ (train_y-ymean))
    prediction = np.clip(ymean + (test_x-mean)/scale @ coef, *bounds)
    assert np.isfinite(prediction).all()
    return prediction, {"mean": mean.tolist(), "scale": scale.tolist(), "coefficients": coef.tolist(), "ymean": ymean}


def crossfit(x, y, folds, alpha, bounds):
    prediction = np.full(y.shape, np.nan)
    fits = []
    for fold in sorted(set(folds)):
        test = folds == fold
        train = ~test
        pred, fit = ridge_predict(x[train], y[train], x[test], alpha, bounds)
        prediction[test] = pred
        fits.append({"fold": int(fold), "train_indices": np.flatnonzero(train).tolist(),
                     "test_indices": np.flatnonzero(test).tolist(), **fit})
    assert np.isfinite(prediction).all()
    return prediction, fits


def features(questions, cases, old_prob, free_prob):
    strata = sorted({(c["template"], c["position"]) for c in cases})
    columns = [f"stratum_{t}_p{p+1}" for t, p in strata]
    columns += ["old_C", "old_F", "old_C_squared", "old_F_squared", "old_C_times_F",
                "neutral_clean_input_tokens", "length_delta", "error_delta", "log_abs_correct_result"]
    rows, gains, old_r = [], [], []
    qindex = {q["question_id"]: i for i, q in enumerate(questions)}
    from scripts.repair_pilot import equation
    for case in cases:
        qi, pos = qindex[case["question_id"]], case["position"]
        c, r, s = old_prob[qi, pos, [0, 2, 3]].mean(axis=-1)
        f = float(free_prob[qi].mean())
        value = int(equation(case["correct_step"]).group(2))
        rows.append([float((case["template"], pos) == st) for st in strata] +
                    [c, f, c*c, f*f, c*f, case["input_tokens"]["CN"],
                     case["input_tokens"]["WN"]-case["input_tokens"]["CN"], case["delta"], np.log1p(abs(value))])
        gains.append(r-s)
        old_r.append(r)
    base = np.asarray(rows, dtype=float)
    return {"difficulty": base, "difficulty_plus_gain": np.column_stack([base, gains]),
            "difficulty_plus_R": np.column_stack([base, old_r])}, columns


def prediction_analysis(xsets, targets, folds, templates, cfg):
    results, prediction_arrays, fitted = {}, {}, {}
    for name, banks in targets.items():
        y = banks.mean(axis=-1).reshape(-1)
        bounds = (-1, 1) if name == "neutral_history_gap" else (0, 1)
        result, preds = {}, {}
        for model_name, x in xsets.items():
            pred, fits = crossfit(x, y, folds, cfg["ridge_alpha"], bounds)
            preds[model_name] = pred.reshape(96, 2)
            fitted[f"{name}/{model_name}"] = fits
            result[model_name] = {"mse": float(np.mean((pred-y)**2)),
                                  "bank_mse": [float(np.mean((preds[model_name]-banks[:, :, b])**2)) for b in (0, 1)]}
        for extension in ["difficulty_plus_gain", "difficulty_plus_R"]:
            base_error = (preds["difficulty"]-y.reshape(96, 2))**2
            new_error = (preds[extension]-y.reshape(96, 2))**2
            improvement = base_error-new_error
            level = cfg["primary_ci_level"] if name == "neutral_history_gap" and extension == "difficulty_plus_gain" else .95
            base_mse = result["difficulty"]["mse"]
            result[extension].update(mse_improvement=float(improvement.mean()),
                                    relative_improvement=float(improvement.mean()/base_mse) if base_mse else None,
                                    ci_level=level,
                                    ci=paired_bootstrap(improvement, templates, level, cfg["seed"], cfg["bootstrap_replicates"]),
                                    bank_improvement=[result["difficulty"]["bank_mse"][b]-result[extension]["bank_mse"][b] for b in (0, 1)])
        results[name], prediction_arrays[name] = result, preds
    return results, prediction_arrays, fitted


def decision(cfg, summaries, contrasts, prediction):
    rates = {r["condition"]: r for r in summaries}
    if max(r["truncation"] for r in summaries) > cfg["max_truncation_rate"]:
        return {"status": "protocol_uninterpretable", "history_supported": False, "incremental_supported": False}
    if rates["CN"]["success"] < cfg["min_clean_success"]:
        return {"status": "clean_control_capability_insufficient", "history_supported": False, "incremental_supported": False}
    history = (contrasts["CN-WN"]["ci"][0] > 0 and contrasts["CN-WN"]["difference"] >= cfg["min_history_gap"]
               and contrasts["CN-WN"]["exact_length_ci95"][0] > 0)
    added = prediction["neutral_history_gap"]["difficulty_plus_gain"]
    incremental = (added["ci"][0] > 0 and added["relative_improvement"] is not None
                   and added["relative_improvement"] >= cfg["min_relative_mse_improvement"]
                   and min(added["bank_improvement"]) > 0)
    status = ("both_signals_worth_followup" if history and incremental else
              "history_only_no_adaptive_signal" if history else
              "predictive_signal_without_history_gate" if incremental else "neither_gate_supported")
    return {"status": status, "history_supported": bool(history), "incremental_supported": bool(incremental),
            "note": "Exploratory follow-up, not training or novelty evidence. No automatic expansion."}


def analyze(cfg, config_path):
    from transformers import AutoTokenizer
    provenance = check_frozen(cfg, config_path)
    out, data = Path(cfg["output_dir"]), Path(cfg["data_dir"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
    questions = read_jsonl(data / "questions.jsonl")
    cases = read_jsonl(data / "cases.jsonl")
    assert questions == read_jsonl(Path(cfg["previous_data"]) / "questions.jsonl")
    assert cases == make_cases(read_jsonl(Path(cfg["previous_data"]) / "cases.jsonl"), cfg, tokenizer)
    foldmap = json.loads((data / "folds.json").read_text())
    assert foldmap == assign_folds(questions, cfg)
    jobs = jobs_for(cases, cfg)
    manifest = json.loads((out / "rollout_manifest.json").read_text())
    assert manifest["config_sha256"] == provenance["config_sha256"]
    assert set(manifest["jobs"]) == {k for k, _ in jobs} and len(jobs) == 384
    rows, artifacts = [], {}
    for job_id, specs in jobs:
        entry = manifest["jobs"][job_id]
        batch = completed_rows(entry, specs, cfg, job_id)
        artifacts[entry["path"]] = entry["sha256"]
        for row in batch:
            ids = continuation_ids(tokenizer, row["prompt"], row["prefix"])
            assert row["input_ids_sha256"] == hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            gen = row["generated_ids"]
            assert len(gen) == row["generated_tokens"]
            assert tokenizer.decode(gen, skip_special_tokens=True) == row["prediction"]
            assert (gen[-1] == tokenizer.eos_token_id) == row["stopped_on_eos"]
            assert tokenizer.eos_token_id not in gen[:-1]
            assert row["hit_token_limit"] == (not row["stopped_on_eos"] and len(gen) == cfg["max_new_tokens"])
        rows.extend(batch)
    assert len(rows) == 12288
    qindex = {q["question_id"]: i for i, q in enumerate(questions)}
    cindex = {c: i for i, c in enumerate(cfg["conditions"])}
    matrix = np.full((96, 2, 4, 2, 8), np.nan)
    for row in rows:
        key = qindex[row["question_id"]], row["position"], cindex[row["condition"]], row["bank"], row["sample"]
        assert np.isnan(matrix[key])
        matrix[key] = row["correct"]
    assert np.isfinite(matrix).all()
    previous = read_jsonl(Path(cfg["previous_output"]) / "predictions.jsonl")
    old_matrix = np.full_like(matrix, np.nan)
    free_matrix = np.full((96, 2, 8), np.nan)
    for row in previous:
        for k, v in grade(row["prediction"], row["gold"]).items():
            assert row[k] == v
        qi = qindex[row["question_id"]]
        if row["condition"] == "F":
            key = qi, row["bank"], row["sample"]
            assert np.isnan(free_matrix[key])
            free_matrix[key] = row["correct"]
        else:
            key = qi, row["position"], ["C", "B", "R", "S"].index(row["condition"]), row["bank"], row["sample"]
            assert np.isnan(old_matrix[key])
            old_matrix[key] = row["correct"]
    assert len(previous) == 13824 and np.isfinite(old_matrix).all() and np.isfinite(free_matrix).all()
    probabilities, old_prob = matrix.mean(-1), old_matrix.mean(-1)
    templates = np.array([q["template"] for q in questions])
    mask = np.array([c["exact_length_matched"] for c in cases]).reshape(96, 2)
    assert mask.sum() == provenance["exact_length_matched"]
    neutral = probabilities[:, :, 2]-probabilities[:, :, 3]
    correction = probabilities[:, :, 0]-probabilities[:, :, 1]
    contrast_data = {"CN-WN": neutral, "CC-WC": correction, "cue_interaction": correction-neutral,
                     "CN-old_C": probabilities[:, :, 2]-old_prob[:, :, 0],
                     "WC-old_R": probabilities[:, :, 1]-old_prob[:, :, 2]}
    contrasts = {}
    for name, banks in contrast_data.items():
        delta = banks.mean(-1)
        level = cfg["primary_ci_level"] if name == "CN-WN" else .95
        contrasts[name] = {"difference": float(delta.mean()), "ci_level": level,
                           "ci": paired_bootstrap(delta, templates, level, cfg["seed"], cfg["bootstrap_replicates"]),
                           "bank_differences": banks.mean(axis=(0, 1)).tolist(),
                           "exact_length_difference": float(delta[mask].mean()),
                           "exact_length_ci95": matched_interval(delta, mask, templates, cfg)}
    xsets, feature_names = features(questions, cases, old_prob, free_matrix)
    folds = np.array([foldmap[c["question_id"]] for c in cases])
    assert all(c["question_id"] == questions[i//2]["question_id"] and c["position"] == i % 2 for i, c in enumerate(cases))
    targets = {"neutral_history_gap": neutral, "corrected_success": probabilities[:, :, 1]}
    prediction, pred_arrays, fits = prediction_analysis(xsets, targets, folds, templates, cfg)
    for fitlist in fits.values():
        for fit in fitlist:
            train_q = {cases[i]["question_id"] for i in fit["train_indices"]}
            test_q = {cases[i]["question_id"] for i in fit["test_indices"]}
            assert not train_q & test_q
    summaries = [{"condition": c, **metrics([r for r in rows if r["condition"] == c])} for c in cfg["conditions"]]
    grouped = [{"template": t, "position": pos+1, "condition": c,
                **metrics([r for r in rows if r["template"] == t and r["position"] == pos and r["condition"] == c])}
               for t in sorted(set(templates)) for pos in (0, 1) for c in cfg["conditions"]]
    cv_rows = []
    for name, target in targets.items():
        for i, case in enumerate(cases):
            qi, pos = qindex[case["question_id"]], case["position"]
            cv_rows.append({"target": name, "case_id": case["case_id"], "question_id": case["question_id"],
                            "fold": int(folds[i]), "observed": float(target[qi, pos].mean()),
                            "bank0": float(target[qi, pos, 0]), "bank1": float(target[qi, pos, 1]),
                            **{k: float(v[qi, pos]) for k, v in pred_arrays[name].items()}})
    feature_rows = [{"case_id": case["case_id"], "question_id": case["question_id"], "fold": int(folds[i]),
                     **dict(zip(feature_names, map(float, xsets["difficulty"][i]))),
                     "old_R_minus_S": float(xsets["difficulty_plus_gain"][i, -1]),
                     "old_R": float(xsets["difficulty_plus_R"][i, -1])} for i, case in enumerate(cases)]
    verdict = decision(cfg, summaries, contrasts, prediction)
    stats = {"decision": verdict, "contrasts": contrasts, "prediction": prediction,
             "caution": "Same 96 previously inspected questions; independent new rollouts. Two fixed primary intervals at 97.5%, others exploratory. OOF-error bootstrap holds fitted folds fixed and does not include all model-fitting variability. Inputs include oracle-assisted C/F estimates with equal reuse across predictors; no deployable or training gain claim."}
    save(out / "statistics.json", stats)
    save(out / "cv_fits.json", {"base_features": feature_names, "fits": fits})
    csv_rows(out / "conditions.csv", summaries)
    csv_rows(out / "template_position.csv", grouped)
    csv_rows(out / "cv_predictions.csv", cv_rows)
    csv_rows(out / "features.csv", feature_rows)
    write_rows(out / "predictions.jsonl", rows)
    artifacts[str(out / "predictions.jsonl")] = sha(out / "predictions.jsonl")
    save(out / "validation.json", {"status": "passed", "questions": 96, "cases": 192, "batches": 384,
                                   "new_predictions": len(rows), "reused_predictions": len(previous),
                                   "exact_length_matched": int(mask.sum()), "question_disjoint_cv": True,
                                   "artifact_sha256": artifacts, "config_sha256": sha(config_path),
                                   "analysis_sha256": sha(__file__), "decision": verdict})
    plot(out, summaries, prediction)
    lines = ["# Matched-history repair controls", "", f"Decision: `{verdict['status']}`.", "",
             "96 reused questions, 192 scenarios, 12,288 new continuations; 13,824 historical continuations validated and reused.", "",
             "| Condition | Success | Mean tokens | Truncation |", "|---|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['condition']} | {r['success']:.2%} | {r['mean_tokens']:.1f} | {r['truncation']:.2%} |")
    lines += ["", "## Contrasts", ""]
    for name, r in contrasts.items():
        lines.append(f"- {name}: {100*r['difference']:+.2f} pp; {100*r['ci_level']:g}% CI [{100*r['ci'][0]:+.2f}, {100*r['ci'][1]:+.2f}]. Exact-length subset: {100*r['exact_length_difference']:+.2f} pp.")
    lines += ["", "## Incremental prediction", ""]
    for name, r in prediction.items():
        gain = r["difficulty_plus_gain"]
        lines.append(f"- {name}: baseline MSE {r['difficulty']['mse']:.6f}; +old(R-S) MSE {gain['mse']:.6f}; improvement {gain['mse_improvement']:+.6f}, CI {gain['ci']}; +oldR MSE {r['difficulty_plus_R']['mse']:.6f}.")
    lines += ["", stats["caution"], "", "CC falsely calls a correct previous step incorrect; neutral CN/WN is primary. Positive aggregate prediction is not proof of a useful selection or training policy."]
    (out / "report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"validation": "passed", "decision": verdict, "conditions": summaries,
                      "contrasts": contrasts, "prediction": prediction}, indent=2))


def plot(out, summaries, prediction):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    rates = {r["condition"]: 100*r["success"] for r in summaries}
    x = np.arange(2)
    axes[0].bar(x-.18, [rates["CC"], rates["CN"]], .36, label="Correct history", color="#579e91")
    axes[0].bar(x+.18, [rates["WC"], rates["WN"]], .36, label="Wrong history", color="#bd8576")
    axes[0].set_xticks(x, ["Correction cue*", "Neutral review"])
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Continuation success (%)")
    axes[0].legend(frameon=False)
    r = prediction["neutral_history_gap"]
    axes[1].bar(["C/F difficulty", "+ old (R-S)", "+ old R"], [r[k]["mse"] for k in ["difficulty", "difficulty_plus_gain", "difficulty_plus_R"]], color=["#999999", "#6e9fc2", "#8b76ab"])
    axes[1].set_ylabel("Question-held-out MSE (lower is better)")
    axes[1].set_title("Predicting neutral history deficit CN-WN")
    fig.suptitle("Frozen Qwen2.5-0.5B · 96 reused questions · no LLM training\n*Correct-history correction cue falsely alleges a prior mistake", fontsize=10)
    fig.savefig(out / "diagnostic.png", dpi=180)
    fig.savefig(out / "diagnostic.pdf")
    plt.close(fig)
