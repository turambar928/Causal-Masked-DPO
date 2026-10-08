#!/usr/bin/env python
"""Frozen stage-two data, training and sharded evaluation; no API dependencies."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cmdpo.data import read_jsonl, write_jsonl


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_rows(path, rows):
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    write_jsonl(tmp, rows)
    tmp.replace(path)


def fresh_tests(cfg, excluded):
    from scripts.prepare_stage1 import IndependentParameters
    from scripts.build_harder_math_pairs import TEMPLATES
    from cmdpo.localization import build_cm_weights
    rng, seen, splits = random.Random(cfg["data_seed"]), set(excluded), {}
    for split, templates in [("id", TEMPLATES[:3] + TEMPLATES[5:]), ("ood", TEMPLATES[3:5])]:
        rows = []
        attempts = 0
        while len(rows) < cfg["eval_rows_per_split"]:
            attempts += 1
            if attempts > 1000000:
                raise RuntimeError("Unable to construct fresh held-out questions")
            pair = templates[len(rows) % len(templates)](IndependentParameters(rng.randrange(10**12)))
            prompt = f"Question: {pair.question}\nAnswer step by step and end with '#### <answer>'."
            if prompt in seen:
                continue
            seen.add(prompt)
            rows.append({"prompt": prompt, "chosen": "\n".join(pair.chosen_steps),
                         "rejected": "\n".join(pair.rejected_steps), "answer": f"#### {pair.answer}",
                         "rejected_steps": pair.rejected_steps, "first_error_step": pair.first_error_step,
                         "step_weights": build_cm_weights(len(pair.rejected_steps), pair.first_error_step, cfg["gamma"]),
                         "metadata": {"source": "stage2_fresh_oracle", "split": split, "template": pair.template,
                                      "row_id": len(rows), "sample_id": hashlib.sha256(prompt.encode()).hexdigest()}})
        splits[split] = rows
    return splits


def prepare(cfg, config_path):
    import numpy as np
    from transformers import AutoTokenizer
    from cmdpo.collator import CMDPOCollator, rejected_token_weights
    from cmdpo.controls import control_weights
    from cmdpo.verifier import verify_answer
    out, data = Path(cfg["output_dir"]), Path(cfg["data_dir"])
    if (out / "provenance.json").exists() or (data / "manifest.json").exists():
        raise FileExistsError("Stage two already frozen; do not overwrite")
    old = Path(cfg["stage1_dir"])
    validation_path = old / "eval512/validation.json"
    previous = json.loads(validation_path.read_text())
    assert previous["status"] == "passed"
    # Historical validation is immutable; validate reused weights against its hashes.
    reused = {}
    for variant in cfg["variants"]:
        if variant in cfg["new_variants"]:
            continue
        for seed in cfg["seeds"]:
            name = f"{variant}_seed{seed}"
            folder = old / "adapters" / name
            files = {}
            for filename in ["adapter_model.safetensors", "adapter_config.json", "run_config.json", "train_metrics.json", "trainer_state.json"]:
                path = folder / filename
                assert sha(path) == previous["artifact_sha256"][str(path)], path
                files[str(path)] = sha(path)
            reused[name] = {"path": str(folder), "files": files}
    old_manifest = json.loads((old / "manifest.json").read_text())
    assert sha(Path(cfg["model"]) / "model.safetensors") == old_manifest["model_sha256"]
    old_data = Path(cfg["stage1_data_dir"])
    old_dm = json.loads((old_data / "manifest.json").read_text())
    for split in ["train", "id", "ood"]:
        assert sha(old_data / f"{split}.jsonl") == old_dm["splits"][split]["sha256"]
    train = read_jsonl(old_data / "train.jsonl")
    assert len(train) == cfg["train_rows"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
    collator = CMDPOCollator(tokenizer, cfg["max_length"], use_chat_template=True, strict_length=True)
    weights_before = [rejected_token_weights(tokenizer, row) for row in train]
    stats, data_files = {}, {}
    for variant in cfg["new_variants"]:
        rows, mass_diffs, unchanged = [], [], 0
        for row, original in zip(train, weights_before):
            weights = control_weights(original, variant, row["metadata"]["sample_id"], cfg["mask_seed"])
            item = {**row, "rejected_token_weights": weights,
                    "metadata": {**row["metadata"], "weight_variant": variant, "mask_seed": cfg["mask_seed"]}}
            expected = np.sum(original, dtype=np.float64) + original[-1]
            actual = collator([item])["rejected_response_mask"][:, 1:].sum().item()
            assert np.isclose(actual, expected, rtol=1e-6, atol=1e-6)
            mass_diffs.append(abs(actual - expected))
            if variant == "prefix_preserving_shuffle":
                assert weights[-1] == original[-1] and sorted(weights) == sorted(original)
                assert all(w == 0 for w, old_w in zip(weights, original) if old_w == 0)
                unchanged += weights == original
            rows.append(item)
        path = data / f"train_{variant}.jsonl"
        write_rows(path, rows)
        data_files[str(path)] = sha(path)
        stats[variant] = {"rows": len(rows), "max_float32_mass_error": max(mass_diffs), "unchanged_masks": unchanged}
    excluded = {r["prompt"] for split in ["train", "id", "ood"] for r in read_jsonl(old_data / f"{split}.jsonl")}
    for split, rows in fresh_tests(cfg, excluded).items():
        for row in rows:
            assert verify_answer(row["chosen"], row["answer"])
            assert not verify_answer(row["rejected"], row["answer"])
        path = data / f"{split}.jsonl"
        write_rows(path, rows)
        data_files[str(path)] = sha(path)
    save(data / "manifest.json", {"config_sha256": sha(config_path), "files": data_files, "controls": stats,
                                   "excluded_stage1_questions": len(excluded), "data_seed": cfg["data_seed"]})
    sources = {str(p): sha(p) for folder in ["cmdpo", "scripts"] for p in Path(folder).glob("*.py")}
    provenance = {"created": now(), "config": cfg, "config_sha256": sha(config_path),
                  "source_sha256": sources, "data_manifest_sha256": sha(data / "manifest.json"),
                  "stage1_validation_sha256": sha(validation_path), "reused": reused,
                  "model_sha256": old_manifest["model_sha256"],
                  "reference_cache_sha256": sha(old / "reference.pt"),
                  "stage1_snapshot_sha256": sha(old / "eval512/reproducibility_source.tar.gz")}
    save(out / "provenance.json", provenance)
    with tarfile.open(out / "frozen_source.tar.gz", "w:gz") as archive:
        for filename in [*sources, str(config_path), "requirements-experiments.txt", *map(str, Path("tests").glob("*.py"))]:
            archive.add(filename, arcname=filename)
    print("Frozen stage2:", stats, "fresh ID/OOD 500 each", flush=True)


def check_frozen(cfg, config_path):
    out = Path(cfg["output_dir"])
    p = json.loads((out / "provenance.json").read_text())
    assert p["config_sha256"] == sha(config_path), "Configuration changed"
    for path in ["cmdpo/collator.py", "cmdpo/controls.py", "cmdpo/loss.py", "cmdpo/reference_cache.py", "cmdpo/trainer.py", "cmdpo/prompting.py",
                 "scripts/train_cmdpo.py", "scripts/evaluate_generation_accuracy.py"]:
        assert sha(path) == p["source_sha256"][path], path
    data_manifest = Path(cfg["data_dir"]) / "manifest.json"
    assert sha(data_manifest) == p["data_manifest_sha256"]
    for path, checksum in json.loads(data_manifest.read_text())["files"].items():
        assert sha(path) == checksum, path
    return p


def train(cfg, config_path, gpu):
    provenance = check_frozen(cfg, config_path)
    out = Path(cfg["output_dir"])
    path = out / "training_manifest.json"
    state = json.loads(path.read_text()) if path.exists() else {"config_sha256": sha(config_path), "jobs": {}}
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    cache = Path(cfg["stage1_dir"]) / "reference.pt"
    assert sha(cache) == provenance["reference_cache_sha256"]
    (out / "logs").mkdir(exist_ok=True)
    for seed in cfg["seeds"]:
        for variant in cfg["new_variants"]:
            name = f"{variant}_seed{seed}"
            previous = state["jobs"].get(name, {})
            if previous.get("status") == "complete":
                assert all(sha(f) == h for f, h in previous["files"].items())
                continue
            adapter = out / "adapters" / name
            command = [sys.executable, "scripts/train_cmdpo.py", "--model", cfg["model"], "--data", f"{cfg['data_dir']}/train_{variant}.jsonl",
                       "--output-dir", str(adapter), "--seed", str(seed), "--use-lora", "--use-chat-template", "--strict-length",
                       "--max-length", str(cfg["max_length"]), "--learning-rate", str(cfg["learning_rate"]), "--epochs", str(cfg["epochs"]),
                       "--beta", str(cfg["beta"]), "--per-device-train-batch-size", str(cfg["train_batch_size"]),
                       "--gradient-accumulation-steps", str(cfg["gradient_accumulation_steps"]), "--reference-cache", str(cache)]
            job = {"status": "running", "started": now(), "command": command, "gpu": gpu, "previous_attempt": previous}
            state["jobs"][name] = job
            save(path, state)
            print("START", name, now(), flush=True)
            with (out / "logs" / f"train_{name}.log").open("a") as log:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                job["pid"] = process.pid
                save(path, state)
                code = process.wait()
            job.update(status="complete" if code == 0 else "failed", finished=now(), returncode=code)
            if code == 0:
                job["files"] = {str(f): sha(f) for f in adapter.iterdir() if f.is_file()}
            save(path, state)
            if code:
                raise RuntimeError(f"Training failed: {name}")
            print("COMPLETE", name, now(), flush=True)
    state["finished"] = now()
    save(path, state)


def evaluate(cfg, config_path, gpu, names, manifest_name):
    provenance = check_frozen(cfg, config_path)
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel
    from scripts.evaluate_generation_accuracy import evaluate_model, load_model
    torch.set_num_threads(4)
    expected = {"base"} | {f"{v}_seed{s}" for v in cfg["variants"] for s in cfg["seeds"]}
    if not names or not set(names).issubset(expected) or len(names) != len(set(names)):
        raise ValueError("Specify unique checkpoint names from the frozen matrix")
    if Path(manifest_name).name != manifest_name:
        raise ValueError("Manifest must be a filename")
    out = Path(cfg["output_dir"]) / "eval512"
    out.mkdir(exist_ok=True)
    path = out / manifest_name
    state = json.loads(path.read_text()) if path.exists() else {"config_sha256": sha(config_path), "jobs": {}}
    assert state["config_sha256"] == sha(config_path)
    state.setdefault("executions", []).append({"pid": os.getpid(), "gpu": gpu, "runs": names, "started": now(),
                                               "orchestrator_sha256": sha(__file__)})
    save(path, state)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(cfg["model"], None, torch.bfloat16, "auto")
    splits = {s: read_jsonl(Path(cfg["data_dir"]) / f"{s}.jsonl") for s in ["id", "ood"]}
    previous_adapter = None
    for name in names:
        if name == "base" and previous_adapter is not None:
            raise ValueError("Base must precede adapters")
        if name != "base":
            if name in provenance["reused"]:
                entry = provenance["reused"][name]
            else:
                training = json.loads((Path(cfg["output_dir"]) / "training_manifest.json").read_text())
                entry = training["jobs"][name]
                assert entry["status"] == "complete", name
                entry = {**entry, "path": str(Path(cfg["output_dir"]) / "adapters" / name)}
            assert all(sha(f) == h for f, h in entry["files"].items()), name
            if previous_adapter is None:
                model = PeftModel.from_pretrained(model, entry["path"], adapter_name=name).eval()
            else:
                model.load_adapter(entry["path"], adapter_name=name)
                model.set_adapter(name)
                model.delete_adapter(previous_adapter)
                model.eval()
            previous_adapter = name
        for split, rows in splits.items():
            key = f"{name}_{split}"
            previous = state["jobs"].get(key, {})
            if previous.get("status") == "complete":
                assert all(sha(f) == h for f, h in previous["files"].items())
                continue
            state["jobs"][key] = {"status": "running", "started": now(), "previous_attempt": previous}
            save(path, state)
            try:
                summary, details = evaluate_model(name, model, tokenizer, rows, cfg["max_new_tokens"], True, cfg["generation_batch_size"])
                sp, dp = out / f"{key}_summary.jsonl", out / f"{key}_details.jsonl"
                write_rows(sp, [summary])
                write_rows(dp, details)
                state["jobs"][key].update(status="complete", finished=now(), files={str(f): sha(f) for f in [sp, dp]})
            except Exception as exc:
                state["jobs"][key].update(status="failed", error=type(exc).__name__, finished=now())
                save(path, state)
                raise
            save(path, state)
            print("COMPLETE", key, summary["accuracy"], flush=True)
    state["finished"] = now()
    save(path, state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "train", "evaluate"])
    parser.add_argument("--config", default="configs/stage2_v1.json")
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--manifest-name", default="evaluation_manifest.json")
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    cfg = json.loads(Path(args.config).read_text())
    if args.action == "prepare":
        prepare(cfg, args.config)
    elif args.action == "train":
        train(cfg, args.config, args.gpu)
    else:
        evaluate(cfg, args.config, args.gpu, args.runs, args.manifest_name)


if __name__ == "__main__":
    main()
