#!/usr/bin/env python
"""Frozen, development-gated local continuation calibration and fresh-question validation."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmdpo.data import read_jsonl
from cmdpo.verifier import extract_answer, verify_answer
from scripts.repair_pilot import (SOURCE_FILES, batch_seed, build_data, completed_rows,
                                 continuation_ids, equation, grade, now, save, sha,
                                 validate_rows, write_rows)

STAGES = ("development", "validation")
NEW_FILES = ["scripts/continuation_calibration.py", "tests/test_continuation_calibration.py",
             "docs/continuation_calibration_protocol.md", "requirements-repair-pilot.txt"]


def digest_ids(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        assert json.loads(path.read_text()) == value, f"Refusing to replace frozen artifact: {path}"
    else:
        save(path, value)


def datasets(cfg, excluded):
    result, seen = {}, set(excluded)
    for stage in STAGES:
        stage_cfg = {**cfg, "questions_per_template": cfg[f"{stage}_questions_per_template"],
                     "seed": batch_seed(cfg, f"data:{stage}")}
        questions, cases = build_data(stage_cfg, seen)
        result[stage] = (questions, cases)
        seen.update(q["prompt"] for q in questions)
    return result


def conditions(stage):
    return ["base", "review", "state", "continue"] if stage == "development" else ["base", "clean", "wrong"]


def prefixes(case, cfg, stage, selected=None):
    clean, wrong = case["prefixes"]["C"], case["prefixes"]["B"]
    if stage == "development":
        return {"base": clean, **{k: clean + v.format(step=case["correct_step"])
                                 for k, v in cfg["templates"].items()}}
    assert selected in cfg["candidate_order"], "Validation requires an eligible frozen selection"
    suffix = cfg["templates"][selected].format(step=case["correct_step"])
    return {"base": clean, "clean": clean + suffix, "wrong": wrong + suffix}


def jobs_for(cases, cfg, stage, selected=None):
    jobs = []
    for bank in cfg[f"{stage}_banks"]:
        specs = []
        for case in cases:
            for condition, prefix in prefixes(case, cfg, stage, selected).items():
                for sample in range(cfg["samples_per_bank"]):
                    specs.append({**{k: case[k] for k in ("question_id", "case_id", "template", "position", "delta", "prompt", "gold")},
                                  "stage": stage, "condition": condition, "prefix": prefix,
                                  "bank": bank, "sample": sample})
        size = cfg["generation_batch_size"]
        assert len(specs) % size == 0
        for start in range(0, len(specs), size):
            jobs.append((f"calibration_{stage}_b{bank}_{start//size:04d}", specs[start:start+size]))
    return jobs


def prepare(cfg, config_path):
    from transformers import AutoTokenizer
    out, data = Path(cfg["output_dir"]), Path(cfg["data_dir"])
    if out.exists() or data.exists():
        raise FileExistsError("Refusing to overwrite existing calibration data or outputs")
    excluded = {r["prompt"] for p in cfg["excluded_data"] for r in read_jsonl(p)}
    splits = datasets(cfg, excluded)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
    max_context = 0
    for stage, (questions, cases) in splits.items():
        assert len(questions) == (48 if stage == "development" else 96)
        for selected in ([None] if stage == "development" else cfg["candidate_order"]):
            jobs = jobs_for(cases, cfg, stage, selected)
            assert sum(len(s) for _, s in jobs) == (3072 if stage == "development" else 9216)
            for case in cases:
                for prefix in prefixes(case, cfg, stage, selected).values():
                    max_context = max(max_context, len(continuation_ids(tokenizer, case["prompt"], prefix)))
        write_rows(data / stage / "questions.jsonl", questions)
        write_rows(data / stage / "cases.jsonl", cases)
    model = Path(cfg["model"])
    assert max_context + cfg["max_new_tokens"] <= json.loads((model/"config.json").read_text())["max_position_embeddings"]
    prior = Path("outputs/repair_controls_v1/provenance.json")
    previous = json.loads(prior.read_text())
    historical = {str(prior): sha(prior)}
    for group in ["source_sha256", "data_sha256", "previous_sha256", "model_sha256"]:
        for name, checksum in previous[group].items():
            assert sha(name) == checksum, name
            historical[name] = checksum
    for name in ["validation.json", "statistics.json", "rollout_manifest.json", "predictions.jsonl", "frozen_source.tar.gz"]:
        path = prior.parent / name
        historical[str(path)] = sha(path)
    sources = list(dict.fromkeys([*SOURCE_FILES, *NEW_FILES]))
    provenance = {"created": now(), "config_sha256": sha(config_path),
                  "source_sha256": {p: sha(p) for p in sources},
                  "data_sha256": {str(p): sha(p) for p in sorted(data.glob("*/*.jsonl"))},
                  "excluded_sha256": {p: sha(p) for p in cfg["excluded_data"]},
                  "historical_sha256": historical,
                  "model_sha256": {str(p): sha(p) for p in sorted(model.iterdir()) if p.suffix in {".json", ".safetensors", ".txt"}},
                  "development_predictions": 3072, "validation_predictions_if_eligible": 9216,
                  "max_context_tokens": max_context, "note": cfg["note"]}
    out.mkdir(parents=True)
    with tarfile.open(out / "frozen_source.tar.gz", "w:gz") as archive:
        for filename in [config_path, *sources]:
            archive.add(filename, arcname=filename)
    provenance["archive_sha256"] = sha(out/"frozen_source.tar.gz")
    save(out / "provenance.json", provenance)
    print(json.dumps({"prepared": True, "questions": [48, 96], "max_predictions": 12288}))


def check_frozen(cfg, config_path):
    out = Path(cfg["output_dir"])
    p = json.loads((out / "provenance.json").read_text())
    assert sha(config_path) == p["config_sha256"], "Configuration changed"
    assert sha(out/"frozen_source.tar.gz") == p["archive_sha256"], "Source archive changed"
    for group in ["source_sha256", "data_sha256", "excluded_sha256", "historical_sha256", "model_sha256"]:
        for name, checksum in p[group].items():
            assert sha(name) == checksum, name
    return p


def choose_protocol(summaries, cfg):
    by = {r["condition"]: r for r in summaries}
    eligible = [c for c in cfg["candidate_order"]
                if by[c]["success"] >= cfg["min_clean_success"]
                and by[c]["success"] + cfg["max_clean_drop"] >= by["base"]["success"]
                and by[c]["truncation"] <= cfg["max_truncation_rate"]]
    if by["base"]["truncation"] > cfg["max_truncation_rate"]:
        eligible = []
    selected = min(eligible, key=lambda c: (-by[c]["success"], by[c]["mean_input_tokens"], cfg["candidate_order"].index(c))) if eligible else None
    return {"status": "eligible" if selected else "calibration_failed", "selected": selected,
            "eligible_candidates": eligible, "selection_uses": "development_only"}


def selection_record(cfg):
    root = Path(cfg["output_dir"])/"development"
    stats = json.loads((root/"statistics.json").read_text())
    validation = json.loads((root/"validation.json").read_text())
    assert validation["status"] == "passed" and validation["predictions"] == 3072
    for name, checksum in validation["artifact_sha256"].items():
        assert sha(name) == checksum, name
    decision = choose_protocol(stats["conditions"], cfg)
    assert decision == stats["decision"]
    return {**decision, "template": cfg["templates"].get(decision["selected"]),
            "development_sha256": {str(root/name): sha(root/name) for name in
                                   ["statistics.json", "validation.json", "rollout_manifest.json", "predictions.jsonl"]}}


def require_selection(cfg):
    path = Path(cfg["output_dir"])/"selection.json"
    if not path.exists():
        raise RuntimeError("Analyze development and freeze selection before validation")
    stored = json.loads(path.read_text())
    assert stored == selection_record(cfg), "Frozen development selection changed"
    if stored["status"] != "eligible":
        raise RuntimeError("Calibration failed; validation sampling is prohibited")
    return stored["selected"]


def rollout(cfg, config_path, stage, gpu, max_jobs):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4")
    provenance = check_frozen(cfg, config_path)
    selected = require_selection(cfg) if stage == "validation" else None
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(cfg["output_dir"])/stage
    out.mkdir(exist_ok=True)
    path = out/"rollout_manifest.json"
    binding = {"config_sha256": provenance["config_sha256"], "stage": stage,
               "selection_sha256": sha(Path(cfg["output_dir"])/"selection.json") if selected else None}
    state = json.loads(path.read_text()) if path.exists() else {**binding, "jobs": {}}
    assert all(state[k] == v for k, v in binding.items())
    cases = read_jsonl(Path(cfg["data_dir"])/stage/"cases.jsonl")
    jobs = jobs_for(cases, cfg, stage, selected)
    assert set(state["jobs"]) <= {j for j, _ in jobs}
    pending = []
    for job_id, specs in jobs:
        entry = state["jobs"].get(job_id)
        if entry and entry["status"] == "complete":
            completed_rows(entry, specs, cfg, job_id)
        else:
            pending.append((job_id, specs))
    if not pending:
        print(f"All {len(jobs)} {stage} batches verified; no generation performed", flush=True)
        return
    assert torch.cuda.is_available(), "No CUDA; refusing fallback"
    runtime = {"torch": torch.__version__, "transformers": transformers.__version__,
               "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
               "dtype": cfg["dtype"], "attention": "sdpa", "batch_size": cfg["generation_batch_size"]}
    if "runtime" in state:
        assert state["runtime"] == runtime, "Resume runtime changed"
    state.update(runtime=runtime, physical_gpu=str(gpu))
    state.pop("finished", None)
    save(path, state)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True, padding_side="left")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], local_files_only=True,
                                               torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    model.eval()
    for count, (job_id, specs) in enumerate(pending):
        if max_jobs is not None and count >= max_jobs:
            break
        entry = {"status": "running", "started": now(), "batch_seed": batch_seed(cfg, job_id)}
        state["jobs"][job_id] = entry
        save(path, state)
        try:
            ids = [continuation_ids(tokenizer, s["prompt"], s["prefix"]) for s in specs]
            assert len(ids) == cfg["generation_batch_size"]
            encoded = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt").to("cuda")
            assert encoded["input_ids"].shape[1]+cfg["max_new_tokens"] <= model.config.max_position_embeddings
            seed = entry["batch_seed"]
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                generated = model.generate(**encoded, do_sample=True, temperature=cfg["temperature"],
                                           top_p=cfg["top_p"], top_k=cfg["top_k"], repetition_penalty=cfg["repetition_penalty"],
                                           max_new_tokens=cfg["max_new_tokens"], use_cache=True,
                                           pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            rows = []
            for spec, token_ids, sequence in zip(specs, ids, generated[:, encoded["input_ids"].shape[1]:].tolist()):
                eos = tokenizer.eos_token_id in sequence
                length = sequence.index(tokenizer.eos_token_id)+1 if eos else len(sequence)
                tokens = sequence[:length]
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                rows.append({**spec, "job_id": job_id, "batch_seed": seed,
                             "input_ids_sha256": digest_ids(token_ids), "prediction": text,
                             "generated_ids": tokens, "generated_tokens": length,
                             "stopped_on_eos": eos, "hit_token_limit": not eos and length == cfg["max_new_tokens"],
                             **grade(text, spec["gold"])})
            validate_rows(rows, specs, cfg, job_id)
            result_path = out/"batches"/f"{job_id}.jsonl"
            write_rows(result_path, rows)
            entry.update(status="complete", finished=now(), path=str(result_path), sha256=sha(result_path))
            save(path, state)
            n = sum(v["status"] == "complete" for v in state["jobs"].values())
            print(f"COMPLETE {stage} {n}/{len(jobs)} {job_id}", flush=True)
        except BaseException as error:
            entry.update(status="failed", failed=now(), error=type(error).__name__+": "+str(error))
            save(path, state)
            raise
    if all(state["jobs"].get(j, {}).get("status") == "complete" for j, _ in jobs):
        state["finished"] = now()
        save(path, state)


def history_decision(cfg, summaries, contrasts, exact):
    by = {s["condition"]: s for s in summaries}
    protocol = (max(s["truncation"] for s in summaries) <= cfg["max_truncation_rate"]
                and by["clean"]["success"] >= cfg["min_clean_success"]
                and contrasts["clean-base"]["ci"][0] > -cfg["max_clean_drop"])
    history = (contrasts["clean-wrong"]["difference"] >= cfg["min_history_gap"]
               and contrasts["clean-wrong"]["ci"][0] > 0
               and min(contrasts["clean-wrong"]["bank_differences"]) > 0
               and exact["fraction"] >= cfg["min_exact_length_fraction"]
               and exact["ci95"] is not None and exact["ci95"][0] > 0)
    return {"protocol_supported": bool(protocol), "history_supported": bool(protocol and history),
            "status": "both_passed_not_training_evidence" if protocol and history else
                      "history_not_supported" if protocol else "protocol_failed"}


def audit_sample(rows):
    chosen = []
    for template in sorted({r["template"] for r in rows}):
        for position in (0, 1):
            for condition in ("clean", "wrong"):
                subset = [r for r in rows if (r["template"], r["position"], r["condition"]) == (template, position, condition)]
                subset.sort(key=lambda r: hashlib.sha256(f"20261003:audit:{r['case_id']}:{r['condition']}:{r['bank']}:{r['sample']}".encode()).hexdigest())
                chosen.extend(subset[:2])
    assert len(chosen) == 48
    return chosen


def analyze(cfg, config_path, stage):
    import numpy as np
    from transformers import AutoTokenizer
    from scripts.analyze_repair_pilot import metrics, paired_bootstrap, resampled_questions, interval, csv_rows
    provenance = check_frozen(cfg, config_path)
    selected = require_selection(cfg) if stage == "validation" else None
    out, data = Path(cfg["output_dir"])/stage, Path(cfg["data_dir"])/stage
    questions, cases = read_jsonl(data/"questions.jsonl"), read_jsonl(data/"cases.jsonl")
    excluded = {r["prompt"] for p in cfg["excluded_data"] for r in read_jsonl(p)}
    # Rebuild predetermined splits; no validation outcomes are accessed by development analysis.
    assert (questions, cases) == datasets(cfg, excluded)[stage]
    manifest = json.loads((out/"rollout_manifest.json").read_text())
    assert manifest["config_sha256"] == provenance["config_sha256"] and manifest["stage"] == stage
    assert manifest["selection_sha256"] == (sha(Path(cfg["output_dir"])/"selection.json") if selected else None)
    jobs = jobs_for(cases, cfg, stage, selected)
    assert set(manifest["jobs"]) == {j for j, _ in jobs}
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
    rows, artifacts, lengths = [], {}, {}
    for job_id, specs in jobs:
        entry = manifest["jobs"][job_id]
        batch = completed_rows(entry, specs, cfg, job_id)
        artifacts[entry["path"]] = entry["sha256"]
        for row in batch:
            key = (row["case_id"], row["condition"])
            if key not in lengths:
                token_ids = continuation_ids(tokenizer, row["prompt"], row["prefix"])
                lengths[key] = (len(token_ids), digest_ids(token_ids))
            assert row["input_ids_sha256"] == lengths[key][1]
            ids = row["generated_ids"]
            assert len(ids) == row["generated_tokens"]
            assert tokenizer.decode(ids, skip_special_tokens=True) == row["prediction"]
            assert row["stopped_on_eos"] == (ids[-1] == tokenizer.eos_token_id)
            assert tokenizer.eos_token_id not in ids[:-1]
            assert row["hit_token_limit"] == (not row["stopped_on_eos"] and len(ids) == cfg["max_new_tokens"])
        rows.extend(batch)
    expected = 3072 if stage == "development" else 9216
    assert len(rows) == expected
    qindex = {q["question_id"]: i for i, q in enumerate(questions)}
    names = conditions(stage)
    matrix = np.full((len(questions), 2, len(names), len(cfg[f"{stage}_banks"]), cfg["samples_per_bank"]), np.nan)
    case_by = {c["case_id"]: c for c in cases}
    for row in rows:
        ix = (qindex[row["question_id"]], row["position"], names.index(row["condition"]), row["bank"], row["sample"])
        assert np.isnan(matrix[ix]), "Duplicate sample"
        matrix[ix] = row["correct"]
    assert np.isfinite(matrix).all()
    summaries = []
    for condition in names:
        subset = [r for r in rows if r["condition"] == condition]
        # A proxy only: final extracted answer equals supplied target intermediate value,
        # restricted to examples where that value differs from the true final answer.
        eligible = [r for r in subset if not verify_answer(equation(case_by[r["case_id"]]["correct_step"]).group(2), r["gold"])]
        premature = [verify_answer(extract_answer(r["prediction"]) or "", equation(case_by[r["case_id"]]["correct_step"]).group(2)) for r in eligible]
        summaries.append({"condition": condition, **metrics(subset),
                          "mean_input_tokens": float(np.mean([lengths[(r["case_id"], condition)][0] for r in subset])),
                          "intermediate_answer_proxy_n": len(eligible),
                          "intermediate_answer_proxy_rate": float(np.mean(premature)) if premature else None})
    result = {"stage": stage, "conditions": summaries, "selected": selected,
              "note": "New questions from the same six templates; not OOD, autonomous correction, training or novelty evidence. Final-answer grading is not process verification."}
    if stage == "development":
        result["decision"] = choose_protocol(summaries, cfg)
    else:
        templates = np.asarray([q["template"] for q in questions])
        probs = matrix.mean(axis=-1)
        comparisons = {}
        for name, left, right in [("clean-base", 1, 0), ("clean-wrong", 1, 2)]:
            banks = probs[:, :, left] - probs[:, :, right]
            delta = banks.mean(axis=-1)
            comparisons[name] = {"difference": float(delta.mean()), "ci_level": cfg["primary_ci_level"],
                                 "ci": paired_bootstrap(delta, templates, cfg["primary_ci_level"], cfg["seed"], cfg["bootstrap_replicates"]),
                                 "bank_differences": banks.mean(axis=(0, 1)).tolist()}
        mask = np.zeros((len(questions), 2), dtype=bool)
        length_rows = []
        for case in cases:
            clean_len, wrong_len = [lengths[(case["case_id"], c)][0] for c in ("clean", "wrong")]
            mask[qindex[case["question_id"]], case["position"]] = clean_len == wrong_len
            length_rows.append({"case_id": case["case_id"], "clean_tokens": clean_len, "wrong_tokens": wrong_len, "delta": wrong_len-clean_len})
        gap = (probs[:, :, 1]-probs[:, :, 2]).mean(axis=-1)
        rng, draws = np.random.default_rng(cfg["seed"]), []
        for _ in range(cfg["bootstrap_replicates"]):
            ix = resampled_questions(templates, rng)
            if mask[ix].any():
                draws.append(float(gap[ix][mask[ix]].mean()))
        exact = {"count": int(mask.sum()), "fraction": float(mask.mean()),
                 "difference": float(gap[mask].mean()) if mask.any() else None,
                 "ci95": interval(draws, .95) if len(draws) == cfg["bootstrap_replicates"] else None}
        result.update(contrasts=comparisons, exact_length=exact,
                      decision=history_decision(cfg, summaries, comparisons, exact))
        csv_rows(out/"input_lengths.csv", length_rows)
        write_rows(out/"audit_sample.jsonl", audit_sample(rows))
        scenario = []
        for qi, q in enumerate(questions):
            for position in (0, 1):
                scenario.append({"question_id": q["question_id"], "template": q["template"], "position": position,
                                 **{f"{c}_bank{b}": float(probs[qi, position, ci, b]) for ci, c in enumerate(names) for b in (0, 1)}})
        csv_rows(out/"scenario_probabilities.csv", scenario)
    immutable_json(out/"statistics.json", result)
    csv_rows(out/"conditions.csv", summaries)
    grouped = [{"template": template, "position": pos, "condition": c,
                **metrics([r for r in rows if (r["template"], r["position"], r["condition"]) == (template, pos, c)])}
               for template in sorted({q["template"] for q in questions}) for pos in (0, 1) for c in names]
    csv_rows(out/"template_position.csv", grouped)
    write_rows(out/"predictions.jsonl", rows)
    for name in ["predictions.jsonl", "rollout_manifest.json", "statistics.json", "conditions.csv", "template_position.csv"]:
        artifacts[str(out/name)] = sha(out/name)
    if stage == "validation":
        for name in ["audit_sample.jsonl", "input_lengths.csv", "scenario_probabilities.csv"]:
            artifacts[str(out/name)] = sha(out/name)
    immutable_json(out/"validation.json", {"status": "passed", "predictions": len(rows), "questions": len(questions),
                                         "jobs": len(jobs), "artifact_sha256": artifacts,
                                         "config_sha256": provenance["config_sha256"], "question_disjoint": True})
    if stage == "development":
        immutable_json(Path(cfg["output_dir"])/"selection.json", selection_record(cfg))
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "rollout", "analyze"])
    p.add_argument("--config", default="configs/continuation_calibration_v1.json")
    p.add_argument("--stage", choices=STAGES, default="development")
    p.add_argument("--gpu", default="3")
    p.add_argument("--max-jobs", type=int)
    args = p.parse_args()
    if args.max_jobs is not None and args.max_jobs <= 0:
        p.error("--max-jobs must be positive")
    cfg = json.loads(Path(args.config).read_text())
    if args.command == "prepare":
        prepare(cfg, args.config)
        return
    with (Path(cfg["output_dir"])/"experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "rollout":
            rollout(cfg, args.config, args.stage, args.gpu, args.max_jobs)
        else:
            analyze(cfg, args.config, args.stage)


if __name__ == "__main__":
    main()
