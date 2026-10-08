#!/usr/bin/env python
"""Execute the frozen stage-one matrix, checkpointing status and exact commands."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stage1_v1.json")
    p.add_argument("--train-only", action="store_true")
    a = p.parse_args()
    os.chdir(ROOT)
    cfg = json.loads(Path(a.config).read_text())
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    cfg_digest = hashlib.sha256(Path(a.config).read_bytes()).hexdigest()
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "created": now(), "config": cfg, "config_sha256": cfg_digest, "jobs": {},
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for folder in ["cmdpo", "scripts"] for p in Path(folder).glob("*.py")},
        "base_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model_sha256": hashlib.sha256((Path(cfg["model"]) / "model.safetensors").read_bytes()).hexdigest()}
    if manifest["config_sha256"] != cfg_digest:
        raise ValueError("Configuration changed; use a new experiment output directory")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": cfg["gpu"], "OMP_NUM_THREADS": "4",
           "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}

    def save():
        tmp = manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2) + "\n")
        tmp.replace(manifest_path)

    def run(name, script, args, artifact):
        existing = manifest["jobs"].get(name, {})
        if existing.get("status") == "complete" and Path(artifact).exists():
            print(f"SKIP {name}: completed", flush=True)
            return
        command = [sys.executable, f"scripts/{script}", *map(str, args)]
        log = out / "logs" / f"{name}.log"
        job = {"status": "running", "started": now(), "command": command, "log": str(log), "artifact": str(artifact)}
        if existing:
            job["previous_attempts"] = existing.get("previous_attempts", []) + [
                {k: v for k, v in existing.items() if k != "previous_attempts"}]
        manifest["jobs"][name] = job
        save()
        print(f"START {name} {now()}", flush=True)
        with log.open("a") as f:
            process = subprocess.Popen(command, env=env, stdout=f, stderr=subprocess.STDOUT)
            job["pid"] = process.pid
            save()
            code = process.wait()
        job.update(status="complete" if code == 0 and Path(artifact).exists() else "failed", returncode=code, finished=now())
        if job["status"] == "complete":
            job["artifact_sha256"] = hashlib.sha256(Path(artifact).read_bytes()).hexdigest()
        save()
        print(f"{job['status'].upper()} {name} {now()}", flush=True)
        if job["status"] != "complete":
            raise RuntimeError(f"Job failed: {name}; see {log}")

    cache = out / "reference.pt"
    if not cache.exists():
        run("reference", "cache_reference.py", ["--model", cfg["model"], "--data", f"{cfg['data_dir']}/train.jsonl",
             "--output", cache, "--batch-size", 8], cache)
    for seed in cfg["seeds"]:
        for variant in cfg["variants"]:
            name = f"{variant}_seed{seed}"
            adapter = out / "adapters" / name
            args = ["--model", cfg["model"], "--data", f"{cfg['data_dir']}/train_{variant}.jsonl",
                    "--output-dir", adapter, "--seed", seed, "--use-lora", "--use-chat-template", "--strict-length",
                    "--max-length", cfg["max_length"], "--learning-rate", cfg["learning_rate"], "--epochs", cfg["epochs"],
                    "--beta", cfg["beta"], "--per-device-train-batch-size", cfg.get("train_batch_size", 1),
                    "--gradient-accumulation-steps", cfg["gradient_accumulation_steps"], "--reference-cache", cache]
            if variant == "sft":
                args += ["--objective", "sft"]
            if variant == "normalized":
                args += ["--normalize-rejected"]
            if variant == "process_positive":
                args += ["--process-positive-weight", cfg["process_positive_weight"]]
            if variant == "dpo_nll":
                args += ["--chosen-nll-weight", cfg["chosen_nll_weight"]]
            run(f"train_{name}", "train_cmdpo.py", args, adapter / "train_metrics.json")
    manifest["training_finished"] = now()
    save()
    if a.train_only:
        return
    # The formal evaluator owns the separate, frozen 512-token protocol.
    subprocess.run([sys.executable, "scripts/evaluate_stage1.py", "--config", a.config], check=True)
    manifest["finished"] = now()
    save()
    print("All stage-one jobs complete", flush=True)


if __name__ == "__main__":
    main()
