#!/usr/bin/env python
"""Matched-history follow-up; historical pilot sources and outputs stay immutable."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmdpo.data import read_jsonl
from scripts.repair_pilot import (sha, now, save, write_rows, continuation_ids, batch_seed,
                                 completed_rows, validate_rows, grade, check_frozen as check_pilot)

NEW_SOURCES = ["scripts/repair_controls.py", "scripts/analyze_repair_controls.py"]


def make_cases(old_cases, cfg, tokenizer):
    result = []
    for old in old_cases:
        correction = cfg["correction_template"].format(step=old["correct_step"])
        neutral = cfg["neutral_template"].format(step=old["correct_step"])
        prefixes = {"CC": old["prefixes"]["C"] + correction,
                    "WC": old["prefixes"]["B"] + correction,
                    "CN": old["prefixes"]["C"] + neutral,
                    "WN": old["prefixes"]["B"] + neutral}
        assert prefixes["WC"] == old["prefixes"]["R"]
        lengths = {c: len(continuation_ids(tokenizer, old["prompt"], p)) for c, p in prefixes.items()}
        result.append({**{k: v for k, v in old.items() if k != "prefixes"}, "prefixes": prefixes,
                       "input_tokens": lengths, "exact_length_matched": lengths["CC"] == lengths["WC"] and lengths["CN"] == lengths["WN"]})
    return result


def assign_folds(questions, cfg):
    folds = {}
    for template in sorted({q["template"] for q in questions}):
        ids = [q["question_id"] for q in questions if q["template"] == template]
        ids.sort(key=lambda q: hashlib.sha256(f"fold:{cfg['seed']}:{q}".encode()).hexdigest())
        for i, qid in enumerate(ids):
            folds[qid] = i % cfg["folds"]
    return folds


def jobs_for(cases, cfg):
    jobs = []
    for case in cases:
        for bank in cfg["banks"]:
            specs = []
            for condition in cfg["conditions"]:
                for sample in range(cfg["samples_per_bank"]):
                    specs.append({"question_id": case["question_id"], "case_id": case["case_id"],
                                  "template": case["template"], "position": case["position"],
                                  "delta": case["delta"], "condition": condition,
                                  "bank": bank, "sample": sample, "prompt": case["prompt"],
                                  "gold": case["gold"], "prefix": case["prefixes"][condition]})
            jobs.append(("controls_" + case["case_id"] + f"_b{bank}", specs))
    return jobs


def prepare(cfg, config_path):
    import numpy
    import scipy
    from transformers import AutoTokenizer
    out, data = Path(cfg["output_dir"]), Path(cfg["data_dir"])
    if out.exists() or data.exists():
        raise FileExistsError("Refusing to overwrite an existing controls experiment")
    old_cfg = json.loads(Path(cfg["previous_config"]).read_text())
    old_provenance = check_pilot(old_cfg, cfg["previous_config"])
    old_root = Path(cfg["previous_output"])
    validation = json.loads((old_root / "validation.json").read_text())
    assert validation["status"] == "passed" and validation["predictions"] == 13824
    for path, checksum in validation["artifact_sha256"].items():
        assert sha(path) == checksum, path
    assert cfg["model"] == old_cfg["model"]
    for key in ["temperature", "top_p", "top_k", "max_new_tokens", "dtype", "generation_batch_size"]:
        assert cfg[key] == old_cfg[key]
    assert cfg["repetition_penalty"] == json.loads((Path(cfg["model"]) / "generation_config.json").read_text())["repetition_penalty"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
    questions = read_jsonl(Path(cfg["previous_data"]) / "questions.jsonl")
    cases = make_cases(read_jsonl(Path(cfg["previous_data"]) / "cases.jsonl"), cfg, tokenizer)
    folds = assign_folds(questions, cfg)
    assert len(questions) == 96 and len(cases) == 192 and len(jobs_for(cases, cfg)) == 384
    write_rows(data / "questions.jsonl", questions)
    write_rows(data / "cases.jsonl", cases)
    save(data / "folds.json", folds)
    sources = {**old_provenance["source_sha256"], **{p: sha(p) for p in NEW_SOURCES}}
    old_files = [old_root / f for f in ["provenance.json", "validation.json", "rollout_manifest.json", "predictions.jsonl", "frozen_source.tar.gz"]]
    old_files += [Path(cfg["previous_config"]), Path(cfg["previous_data"]) / "questions.jsonl", Path(cfg["previous_data"]) / "cases.jsonl"]
    provenance = {"created": now(), "config_sha256": sha(config_path), "source_sha256": sources,
                  "data_sha256": {str(p): sha(p) for p in data.iterdir()},
                  "previous_sha256": {str(p): sha(p) for p in old_files},
                  "model_sha256": old_provenance["model_sha256"],
                  "questions": 96, "cases": 192, "new_batches": 384, "new_predictions": 12288,
                  "exact_length_matched": sum(c["exact_length_matched"] for c in cases),
                  "numpy": numpy.__version__, "scipy": scipy.__version__,
                  "note": "Same questions, newly seeded continuations; not fresh task generalization."}
    save(out / "provenance.json", provenance)
    with tarfile.open(out / "frozen_source.tar.gz", "w:gz") as a:
        for p in [*sources, str(config_path), "tests/test_repair_controls.py", "docs/repair_controls_protocol.md", "requirements-repair-pilot.txt"]:
            a.add(p, arcname=p)
    print({k:v for k,v in provenance.items() if not k.endswith("sha256")})


def check_frozen(cfg, config_path):
    p = json.loads((Path(cfg["output_dir"]) / "provenance.json").read_text())
    assert sha(config_path) == p["config_sha256"]
    for group in ["source_sha256", "data_sha256", "previous_sha256", "model_sha256"]:
        for path, checksum in p[group].items():
            assert sha(path) == checksum, path
    return p


def rollout(cfg, config_path, gpu, max_jobs):
    import fcntl
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4")
    p = check_frozen(cfg, config_path)
    out = Path(cfg["output_dir"])
    with (out / "rollout.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(cfg, p, gpu, max_jobs)


def run_locked(cfg, provenance, gpu, max_jobs):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(cfg["output_dir"])
    path = out / "rollout_manifest.json"
    state = json.loads(path.read_text()) if path.exists() else {"config_sha256": provenance["config_sha256"], "jobs": {}}
    assert state["config_sha256"] == provenance["config_sha256"]
    jobs = jobs_for(read_jsonl(Path(cfg["data_dir"]) / "cases.jsonl"), cfg)
    pending = []
    for job_id, specs in jobs:
        entry = state["jobs"].get(job_id)
        if entry and entry["status"] == "complete":
            completed_rows(entry, specs, cfg, job_id)
        else:
            pending.append((job_id, specs))
    if not pending:
        print("All 384 batches complete and verified; no generation performed", flush=True)
        return
    assert torch.cuda.is_available(), "No GPU; refusing implicit fallback"
    runtime = {"torch": torch.__version__, "transformers": transformers.__version__,
               "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
               "dtype": cfg["dtype"], "attention": "sdpa", "batch_size": cfg["generation_batch_size"]}
    if "runtime" in state:
        assert state["runtime"] == runtime
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
            encoded = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt").to("cuda")
            assert len(ids) == cfg["generation_batch_size"]
            assert encoded["input_ids"].shape[1] + cfg["max_new_tokens"] <= model.config.max_position_embeddings
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
                             "input_ids_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
                             "prediction": text, "generated_ids": tokens, "generated_tokens": length,
                             "stopped_on_eos": eos, "hit_token_limit": not eos and length == cfg["max_new_tokens"],
                             **grade(text, spec["gold"])})
            validate_rows(rows, specs, cfg, job_id)
            result_path = out / "batches" / f"{job_id}.jsonl"
            write_rows(result_path, rows)
            entry.update(status="complete", finished=now(), path=str(result_path), sha256=sha(result_path))
            save(path, state)
            n = sum(v["status"] == "complete" for v in state["jobs"].values())
            print(f"COMPLETE {n}/384 {job_id}", flush=True)
        except BaseException as error:
            entry.update(status="failed", failed=now(), error=type(error).__name__+": "+str(error))
            save(path, state)
            raise
    if all(state["jobs"].get(k, {}).get("status") == "complete" for k, _ in jobs):
        state["finished"] = now()
        save(path, state)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "rollout", "analyze"])
    p.add_argument("--config", default="configs/repair_controls_v1.json")
    p.add_argument("--gpu", default="3")
    p.add_argument("--max-jobs", type=int)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.command == "prepare":
        prepare(cfg, args.config)
    elif args.command == "rollout":
        rollout(cfg, args.config, args.gpu, args.max_jobs)
    else:
        from scripts.analyze_repair_controls import analyze
        analyze(cfg, args.config)


if __name__ == "__main__":
    main()
