#!/usr/bin/env python
"""Frozen local-Qwen repair diagnostic: prepare / rollout / analyze."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cmdpo.data import read_jsonl
from cmdpo.prompting import format_prompt
from cmdpo.verifier import verify_answer

EQUATION = re.compile(r"(-?\d+(?:\s*[+*\-]\s*-?\d+)+)\s*=\s*(-?\d+)")
SOURCE_FILES = ["scripts/repair_pilot.py", "scripts/analyze_repair_pilot.py",
                "scripts/prepare_stage1.py", "scripts/build_harder_math_pairs.py",
                "cmdpo/data.py", "cmdpo/prompting.py", "cmdpo/verifier.py"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def arithmetic(expression):
    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -evaluate(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
        raise ValueError("Unsupported arithmetic expression")
    return evaluate(ast.parse(expression, mode="eval").body)


def equation(step):
    matches = list(EQUATION.finditer(step))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one equation: {step}")
    return matches[0]


def inject_error(step, delta):
    if delta not in (-1, 1):
        raise ValueError("Only +/-1 perturbations are frozen")
    match = equation(step)
    value = int(match.group(2))
    if arithmetic(match.group(1)) != value:
        raise ValueError("Source step is not correct")
    start, end = match.span(2)
    return step[:start] + str(value + delta) + step[end:]


def make_cases(question, index, cfg):
    cases = []
    for pos in cfg["positions"]:
        step = question["correct_steps"][pos]
        delta = 1 if (index + pos) % 2 == 0 else -1
        bad_step = inject_error(step, delta)
        before = question["correct_steps"][:pos]
        clean = "\n".join([*before, step]) + "\n"
        bad = "\n".join([*before, bad_step]) + "\n"
        prefixes = {"C": clean, "B": bad,
                    "R": bad + cfg["correction_template"].format(step=step),
                    "S": bad + cfg["correction_template"].format(step=bad_step)}
        cases.append({"case_id": question["question_id"] + f"_p{pos+1}",
                      "question_id": question["question_id"], "template": question["template"],
                      "position": pos, "delta": delta, "prompt": question["prompt"],
                      "gold": question["gold"], "correct_step": step,
                      "wrong_step": bad_step, "prefixes": prefixes})
    return cases


def build_data(cfg, excluded):
    from scripts.prepare_stage1 import IndependentParameters
    from scripts.build_harder_math_pairs import TEMPLATES
    rng, seen = random.Random(cfg["seed"]), set(excluded)
    questions, cases = [], []
    for template in TEMPLATES:
        accepted, attempts = 0, 0
        while accepted < cfg["questions_per_template"]:
            attempts += 1
            if attempts > 100000:
                raise RuntimeError("Unable to generate fresh questions")
            pair = template(IndependentParameters(rng.randrange(10**12)))
            prompt = f"Question: {pair.question}\nAnswer step by step and end with '#### <answer>'."
            if prompt in seen:
                continue
            seen.add(prompt)
            steps = pair.chosen_steps[:-1]
            for step in steps:
                match = equation(step)
                assert arithmetic(match.group(1)) == int(match.group(2))
            assert arithmetic(equation(steps[-1]).group(1)) == pair.answer
            assert pair.chosen_steps[-1] == f"#### {pair.answer}"
            question = {"question_id": hashlib.sha256(prompt.encode()).hexdigest(),
                        "template": pair.template, "prompt": prompt,
                        "gold": f"#### {pair.answer}", "correct_steps": steps}
            questions.append(question)
            cases.extend(make_cases(question, accepted, cfg))
            accepted += 1
    return questions, cases


def jobs_for(questions, cases, cfg):
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
            jobs.append((case["case_id"] + f"_b{bank}", specs))
    per_job = cfg["generation_batch_size"] // cfg["samples_per_bank"]
    for start in range(0, len(questions), per_job):
        for bank in cfg["banks"]:
            specs = []
            for q in questions[start:start+per_job]:
                for sample in range(cfg["samples_per_bank"]):
                    specs.append({"question_id": q["question_id"], "case_id": None,
                                  "template": q["template"], "position": None, "delta": None,
                                  "condition": "F", "bank": bank, "sample": sample,
                                  "prompt": q["prompt"], "gold": q["gold"], "prefix": ""})
            jobs.append((f"free_{start:03d}_b{bank}", specs))
    return jobs


def batch_seed(cfg, job_id):
    return int(hashlib.sha256(f"{cfg['seed']}:{job_id}".encode()).hexdigest()[:15], 16)


def continuation_ids(tokenizer, prompt, prefix):
    # Encode the exact existing chat generation prompt, then append assistant text.
    # No extra user turn, instruction, or EOS after the partial assistant message.
    prompt_ids = tokenizer(format_prompt(tokenizer, prompt, True), add_special_tokens=False)["input_ids"]
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id in prefix_ids:
        raise ValueError("Partial assistant prefix contains EOS")
    return prompt_ids + prefix_ids


def grade(text, gold):
    # Never concatenate the supplied prefix here.
    return {"correct": verify_answer(text, gold),
            "has_explicit_answer": bool(re.search(r"####|\\boxed\{", text))}


def prepare(cfg, config_path):
    import platform
    import torch
    import transformers
    import tokenizers
    from transformers import AutoTokenizer
    out, data = Path(cfg["output_dir"]), Path(cfg["data_dir"])
    if out.exists() or data.exists():
        raise FileExistsError("Pilot destination already exists; refusing to overwrite")
    excluded = {r["prompt"] for p in cfg["excluded_data"] for r in read_jsonl(p)}
    questions, cases = build_data(cfg, excluded)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
    jobs = jobs_for(questions, cases, cfg)
    assert len(questions) == 96 and len(cases) == 192 and len(jobs) == 432
    assert sum(len(specs) for _, specs in jobs) == 13824
    max_context = max(len(continuation_ids(tokenizer, s["prompt"], s["prefix"]))
                      for _, specs in jobs for s in specs[::cfg["samples_per_bank"]])
    write_rows(data / "questions.jsonl", questions)
    write_rows(data / "cases.jsonl", cases)
    files = {str(p): sha(p) for p in [data / "questions.jsonl", data / "cases.jsonl"]}
    model_files = [p for p in Path(cfg["model"]).iterdir() if p.suffix in {".json", ".safetensors", ".txt"}]
    provenance = {"created": now(), "config_path": str(config_path), "config_sha256": sha(config_path),
                  "source_sha256": {p: sha(p) for p in SOURCE_FILES}, "data_sha256": files,
                  "model_sha256": {str(p): sha(p) for p in sorted(model_files)},
                  "excluded_sha256": {p: sha(p) for p in cfg["excluded_data"]},
                  "excluded_prompts": len(excluded), "questions": len(questions), "cases": len(cases),
                  "jobs": len(jobs), "predictions": 13824, "max_context_tokens": max_context,
                  "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                  "transformers": transformers.__version__, "tokenizers": tokenizers.__version__}}
    save(out / "provenance.json", provenance)
    with tarfile.open(out / "frozen_source.tar.gz", "w:gz") as archive:
        for filename in [str(config_path), *SOURCE_FILES, "tests/test_repair_pilot.py", "docs/repair_pilot_protocol.md"]:
            archive.add(filename, arcname=filename)
    print(json.dumps({k:v for k,v in provenance.items() if not k.endswith("sha256")}, indent=2))


def check_frozen(cfg, config_path):
    p = json.loads((Path(cfg["output_dir"]) / "provenance.json").read_text())
    assert sha(config_path) == p["config_sha256"], "Configuration changed"
    for group in ["source_sha256", "data_sha256", "model_sha256", "excluded_sha256"]:
        for path, checksum in p[group].items():
            assert sha(path) == checksum, path
    return p


def validate_rows(rows, specs, cfg, job_id):
    assert len(rows) == len(specs)
    for row, expected in zip(rows, specs):
        for k, v in expected.items():
            assert row[k] == v, (job_id, k)
        assert row["job_id"] == job_id and row["batch_seed"] == batch_seed(cfg, job_id)
        assert 0 < row["generated_tokens"] <= cfg["max_new_tokens"]
        assert row["stopped_on_eos"] != row["hit_token_limit"]
        for k, v in grade(row["prediction"], row["gold"]).items():
            assert row[k] == v, (job_id, k)


def completed_rows(entry, specs, cfg, job_id):
    assert entry["status"] == "complete"
    assert sha(entry["path"]) == entry["sha256"], job_id
    rows = read_jsonl(entry["path"])
    validate_rows(rows, specs, cfg, job_id)
    return rows


def rollout(cfg, config_path, gpu, max_jobs):
    import fcntl
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "4"
    out = Path(cfg["output_dir"])
    provenance = check_frozen(cfg, config_path)
    with (out / "rollout.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _rollout_locked(cfg, provenance, gpu, max_jobs)


def _rollout_locked(cfg, provenance, gpu, max_jobs):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out, data = Path(cfg["output_dir"]), Path(cfg["data_dir"])
    path = out / "rollout_manifest.json"
    state = json.loads(path.read_text()) if path.exists() else {"config_sha256": provenance["config_sha256"], "jobs": {}}
    assert state["config_sha256"] == provenance["config_sha256"]
    questions, cases = read_jsonl(data / "questions.jsonl"), read_jsonl(data / "cases.jsonl")
    jobs = jobs_for(questions, cases, cfg)
    pending = []
    for job_id, specs in jobs:
        entry = state["jobs"].get(job_id)
        if entry and entry["status"] == "complete":
            completed_rows(entry, specs, cfg, job_id)
        else:
            pending.append((job_id, specs))
    if not pending:
        print("All 432 batches already complete and verified", flush=True)
        return
    assert torch.cuda.is_available(), "A CUDA GPU is required; no implicit CPU fallback"
    runtime = {"torch": torch.__version__, "transformers": transformers.__version__,
               "gpu": torch.cuda.get_device_name(), "compute_capability": list(torch.cuda.get_device_capability()),
               "dtype": cfg["dtype"], "attention": "sdpa", "batch_size": cfg["generation_batch_size"]}
    if "runtime" in state:
        assert state["runtime"] == runtime, "Resume runtime changed"
    state["runtime"] = runtime
    state["physical_gpu"] = str(gpu)
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
                                           top_p=cfg["top_p"], top_k=cfg["top_k"],
                                           max_new_tokens=cfg["max_new_tokens"], use_cache=True,
                                           pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            rows = []
            width = encoded["input_ids"].shape[1]
            for spec, token_ids, sequence in zip(specs, ids, generated[:, width:].tolist()):
                eos = tokenizer.eos_token_id in sequence
                length = sequence.index(tokenizer.eos_token_id) + 1 if eos else len(sequence)
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
            print(f"COMPLETE {n}/432 {job_id}", flush=True)
        except BaseException as error:
            entry.update(status="failed", error=type(error).__name__ + ": " + str(error), failed=now())
            save(path, state)
            raise
    if all(state["jobs"].get(k, {}).get("status") == "complete" for k, _ in jobs):
        state["finished"] = now()
        save(path, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "rollout", "analyze"])
    parser.add_argument("--config", default="configs/repair_pilot_v1.json")
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--max-jobs", type=int, help="Smoke/resume cap; never changes batch composition")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.command == "prepare":
        prepare(cfg, args.config)
    elif args.command == "rollout":
        rollout(cfg, args.config, args.gpu, args.max_jobs)
    else:
        from scripts.analyze_repair_pilot import analyze
        analyze(cfg, args.config)


if __name__ == "__main__":
    main()
