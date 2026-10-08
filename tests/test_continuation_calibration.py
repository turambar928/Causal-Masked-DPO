import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.continuation_calibration import (audit_sample, check_frozen, choose_protocol,
    datasets, history_decision, immutable_json, jobs_for, prefixes, require_selection,
    selection_record)
from scripts.repair_pilot import batch_seed, completed_rows, grade, save, sha, write_rows
from scripts.analyze_repair_pilot import paired_bootstrap


def config():
    return json.loads(Path("configs/continuation_calibration_v1.json").read_text())


def summaries():
    return [{"condition": c, "success": .5, "truncation": 0., "mean_input_tokens": 100+i}
            for i, c in enumerate(["base", "review", "state", "continue"])]


def test_fresh_disjoint_balanced_and_reproducible_questions():
    cfg = config()
    initial = datasets(cfg, set())
    excluded = {initial["development"][0][0]["prompt"]}
    result = datasets(cfg, excluded)
    assert result == datasets(cfg, excluded)
    seen = set(excluded)
    for stage, n in [("development", 48), ("validation", 96)]:
        questions, cases = result[stage]
        prompts = {q["prompt"] for q in questions}
        assert len(prompts) == n and not seen.intersection(prompts)
        seen.update(prompts)
        for template in {q["template"] for q in questions}:
            assert sum(q["template"] == template for q in questions) == n//6
        assert len(cases) == n*2
        for q in questions:
            assert {c["position"] for c in cases if c["question_id"] == q["question_id"]} == {0, 1}


def test_prefixes_share_repair_and_never_append_solution():
    cfg = config()
    for _, cases in datasets(cfg, set()).values():
        for case in cases:
            dev = prefixes(case, cfg, "development")
            assert all(p.startswith(case["prefixes"]["C"]) for p in dev.values())
            for selected in cfg["candidate_order"]:
                test = prefixes(case, cfg, "validation", selected)
                suffix = cfg["templates"][selected].format(step=case["correct_step"])
                assert test["clean"] == case["prefixes"]["C"]+suffix
                assert test["wrong"] == case["prefixes"]["B"]+suffix
                assert test["base"] == case["prefixes"]["C"]
                assert case["gold"] not in suffix
    with pytest.raises(AssertionError):
        prefixes(case, cfg, "validation")


def test_fixed_budgets_unique_samples_and_stage_bank_seeds():
    cfg = config()
    seeds = set()
    for stage, (_, cases) in datasets(cfg, set()).items():
        jobs = jobs_for(cases, cfg, stage, "continue" if stage == "validation" else None)
        assert len(jobs) == (96 if stage == "development" else 288)
        assert all(len(specs) == 32 for _, specs in jobs)
        rows = [s for _, specs in jobs for s in specs]
        keys = {(r["case_id"], r["condition"], r["bank"], r["sample"]) for r in rows}
        assert len(keys) == len(rows) == (3072 if stage == "development" else 9216)
        for job_id, specs in jobs:
            seed = batch_seed(cfg, job_id)
            assert seed not in seeds
            seeds.add(seed)
            assert len({s["bank"] for s in specs}) == 1


def test_selection_thresholds_ties_and_no_validation_inputs():
    cfg, rows = config(), summaries()
    assert choose_protocol(rows, cfg)["selected"] == "state"
    rows[2]["mean_input_tokens"] = 200
    assert choose_protocol(rows, cfg)["selected"] == "continue"
    rows[3]["mean_input_tokens"] = 200
    assert choose_protocol(rows, cfg)["selected"] == "state"
    rows[0]["success"] = .45
    rows[2]["success"], rows[3]["success"] = .4, .399
    rows[2]["truncation"] = .05
    assert choose_protocol(rows, cfg)["selected"] == "state"
    rows[2]["truncation"] = .05001
    assert choose_protocol(rows, cfg)["status"] == "calibration_failed"
    rows = summaries()
    rows[0]["success"] = .5501
    assert choose_protocol(rows, cfg)["selected"] is None


def test_validation_blocked_missing_or_failed_selection_and_tampering(tmp_path):
    cfg = {**config(), "output_dir": str(tmp_path)}
    with pytest.raises(RuntimeError, match="Analyze development"):
        require_selection(cfg)
    root = tmp_path/"development"
    rows = summaries()
    rows[2]["success"] = rows[3]["success"] = .1
    save(root/"statistics.json", {"conditions": rows, "decision": choose_protocol(rows, cfg)})
    save(root/"validation.json", {"status": "passed", "predictions": 3072, "artifact_sha256": {}})
    save(root/"rollout_manifest.json", {})
    write_rows(root/"predictions.jsonl", [])
    frozen = selection_record(cfg)
    save(tmp_path/"selection.json", frozen)
    with pytest.raises(RuntimeError, match="Calibration failed"):
        require_selection(cfg)
    frozen["selected"] = "continue"
    save(tmp_path/"selection.json", frozen)
    with pytest.raises(AssertionError, match="selection changed"):
        require_selection(cfg)


def test_immutable_artifacts_and_hash_fail_closed(tmp_path):
    artifact = tmp_path/"statistics.json"
    immutable_json(artifact, {"x": 1})
    immutable_json(artifact, {"x": 1})
    with pytest.raises(AssertionError):
        immutable_json(artifact, {"x": 2})
    cfg = {**config(), "output_dir": str(tmp_path)}
    config_path = tmp_path/"config.json"
    save(config_path, cfg)
    save(tmp_path/"provenance.json", {"config_sha256": "incorrect"})
    with pytest.raises(AssertionError, match="Configuration changed"):
        check_frozen(cfg, config_path)


def test_completed_batch_resume_and_new_text_only_grading(tmp_path):
    cfg = config()
    job = "calibration_development_b0_0000"
    spec = {"gold": "#### 42", "prefix": "#### 42", "condition": "base"}
    row = {**spec, "prediction": "", "job_id": job, "batch_seed": batch_seed(cfg, job),
           "generated_tokens": 1, "stopped_on_eos": True, "hit_token_limit": False, **grade("", spec["gold"])}
    assert not row["correct"]
    path = tmp_path/"batch.jsonl"
    write_rows(path, [row])
    entry = {"status": "complete", "path": str(path), "sha256": sha(path)}
    assert completed_rows(entry, [spec], cfg, job) == [row]
    row["prediction"] = "42"
    write_rows(path, [row])
    with pytest.raises(AssertionError):
        completed_rows(entry, [spec], cfg, job)


def test_paired_statistics_and_validation_gate_boundaries():
    cfg = config()
    values = np.full((12, 2), .125)
    assert paired_bootstrap(values, ["a"]*6+["b"]*6, .975, 42, 50) == [.125, .125]
    rows = [{"condition": c, "success": .5, "truncation": 0.} for c in ("base", "clean", "wrong")]
    contrasts = {"clean-base": {"ci": [-.049, .01]},
                 "clean-wrong": {"difference": .03, "ci": [.001, .1], "bank_differences": [.02, .04]}}
    exact = {"fraction": .9, "ci95": [.001, .1]}
    assert history_decision(cfg, rows, contrasts, exact)["history_supported"]
    contrasts["clean-base"]["ci"][0] = -.05
    assert history_decision(cfg, rows, contrasts, exact)["status"] == "protocol_failed"
    contrasts["clean-base"]["ci"][0] = -.049
    exact["fraction"] = .899
    assert history_decision(cfg, rows, contrasts, exact)["status"] == "history_not_supported"
    exact["fraction"] = .9
    contrasts["clean-wrong"]["bank_differences"][0] = 0
    assert not history_decision(cfg, rows, contrasts, exact)["history_supported"]
    rows[0]["truncation"] = .0501
    assert history_decision(cfg, rows, contrasts, exact)["status"] == "protocol_failed"


def test_fixed_audit_sample_blind_to_correctness():
    cfg = config()
    cases = datasets(cfg, set())["validation"][1]
    rows = [s for _, specs in jobs_for(cases, cfg, "validation", "state") for s in specs]
    selected = audit_sample(rows)
    assert len(selected) == 48
    assert selected == audit_sample(list(reversed(rows)))
    for row in rows:
        row["correct"] = False
    assert [(r["case_id"], r["condition"], r["bank"], r["sample"]) for r in selected] == [
        (r["case_id"], r["condition"], r["bank"], r["sample"]) for r in audit_sample(rows)]
