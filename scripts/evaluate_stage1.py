#!/usr/bin/env python
"""Evaluate completed adapters while the independent training queue advances."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stage1_v1.json")
    p.add_argument("--eval-config", default="configs/stage1_eval_v2.json")
    p.add_argument("--seeds", type=int, nargs="+", help="Evaluate only these frozen training seeds")
    p.add_argument("--runs", nargs="+", help="Restrict work to explicit variant_seedN checkpoint names")
    p.add_argument("--gpu", help="Execution device override; does not change the inference protocol")
    p.add_argument("--manifest-name", default="evaluation_manifest.json")
    p.add_argument("--skip-summary", action="store_true")
    a = p.parse_args()
    os.chdir(ROOT)
    cfg = json.loads(Path(a.config).read_text())
    ecfg = json.loads(Path(a.eval_config).read_text())
    seeds = a.seeds or cfg["seeds"]
    if not set(seeds).issubset(cfg["seeds"]):
        raise ValueError("Shard seeds must belong to the frozen experiment")
    allowed_runs = {f"{variant}_seed{seed}" for seed in seeds for variant in cfg["variants"]}
    selected_runs = set(a.runs) if a.runs else allowed_runs
    if not selected_runs.issubset(allowed_runs):
        raise ValueError("Requested checkpoints must belong to the frozen experiment and selected seeds")
    if Path(a.manifest_name).name != a.manifest_name:
        raise ValueError("Manifest name must be a filename")
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu or ecfg["gpu"]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    from cmdpo.collator import CMDPOCollator
    from cmdpo.data import read_jsonl, write_jsonl
    from scripts.evaluate_generation_accuracy import evaluate_model, load_model
    from scripts.evaluate_likelihood_deltas import mean_probe_logp, probe_rows

    torch.set_num_threads(4)
    train_out = Path(cfg["output_dir"])
    out = Path(ecfg.get("output_dir", cfg["output_dir"]))
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / a.manifest_name
    digest = hashlib.sha256(Path(a.eval_config).read_bytes()).hexdigest()
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "config": ecfg, "config_sha256": digest, "jobs": {},
        "training_config_sha256": hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for folder in ["cmdpo", "scripts"] for p in Path(folder).glob("*.py")}}
    if state["config_sha256"] != digest:
        raise ValueError("Evaluation configuration changed")
    # Reuse immutable completed artifacts from other disjoint evaluation workers.
    for other_path in out.glob("evaluation*manifest.json"):
        if other_path == state_path:
            continue
        other = json.loads(other_path.read_text())
        if other["config_sha256"] != digest or other["training_config_sha256"] != state["training_config_sha256"]:
            raise ValueError("Evaluation shard protocol mismatch")
        for name, job in other["jobs"].items():
            if job["status"] == "complete" and state["jobs"].get(name, {}).get("status") != "complete":
                state["jobs"][name] = job
    state.setdefault("executions", []).append({"pid": os.getpid(), "gpu": a.gpu or ecfg["gpu"], "seeds": seeds,
                                               "runs": sorted(selected_runs),
                                               "orchestrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                               "started": datetime.now(timezone.utc).isoformat()})

    def save():
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(state_path)

    def done(name):
        job = state["jobs"].get(name, {})
        return job.get("status") == "complete" and all(Path(f).exists() for f in job["files"])

    def record(name, fn):
        if done(name):
            print(f"SKIP {name}", flush=True)
            return
        previous_attempt = state["jobs"].get(name, {})
        state["jobs"][name] = {"status": "running", "started": datetime.now(timezone.utc).isoformat()}
        if previous_attempt:
            state["jobs"][name]["previous_attempts"] = previous_attempt.get("previous_attempts", []) + [
                {k: v for k, v in previous_attempt.items() if k != "previous_attempts"}]
        save()
        try:
            files = fn()
        except Exception as e:
            state["jobs"][name].update(status="failed", error=type(e).__name__)
            save()
            raise
        state["jobs"][name].update(status="complete", finished=datetime.now(timezone.utc).isoformat(),
            files={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
        save()
        print(f"COMPLETE {name}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(cfg["model"], None, torch.bfloat16, "auto")
    splits = {split: read_jsonl(Path(cfg["data_dir"]) / f"{split}.jsonl") for split in ["id", "ood"]}
    if ecfg["include_gsm8k_transfer"]:
        splits["gsm8k_test"] = read_jsonl(Path(cfg["data_dir"]) / "gsm8k_test.jsonl")
    collator = CMDPOCollator(tokenizer, cfg["max_length"], use_chat_template=True, strict_length=True)
    probe_data = {split: {probe: probe_rows(rows[:cfg["probe_rows_per_split"]], probe, cfg["gamma"])
                         for probe in ["prefix", "error", "suffix", "cmdpo"]}
                  for split, rows in splits.items() if split != "gsm8k_test"}

    def generation(name, split):
        summary, details = evaluate_model(name, model, tokenizer, splits[split], ecfg["max_new_tokens"], True,
                                           ecfg["generation_batch_size"])
        sp, dp = out / f"{name}_{split}_summary.jsonl", out / f"{name}_{split}_details.jsonl"
        write_rows(sp, [summary])
        write_rows(dp, details)
        print(f"{name}/{split}: {summary['correct']}/{summary['total']}", flush=True)
        return [sp, dp]

    def write_rows(path, rows):
        temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        write_jsonl(temp, rows)
        temp.replace(path)

    before_path = out / "base_heldout_probes.json"
    for split in splits:
        record(f"base_{split}", lambda split=split: generation("base", split))
    if not before_path.exists():
        before = {split: {probe: mean_probe_logp(model, collator, rows, model.device, ecfg["probe_batch_size"],
                                                f"base/{split}/{probe}", True)
                          for probe, rows in probes.items()} for split, probes in probe_data.items()}
        before_path.write_text(json.dumps(before, indent=2) + "\n")
    before = json.loads(before_path.read_text())

    def probe_adapter(name, split):
        after = {probe: mean_probe_logp(model, collator, rows, model.device, ecfg["probe_batch_size"],
                                       f"{name}/{split}/{probe}", True)
                 for probe, rows in probe_data[split].items()}
        row = {"variant": name, "split": split, "aggregation": "mean_nonempty_example_per_token",
               "examples": cfg["probe_rows_per_split"], "use_chat_template": True}
        for probe in after:
            row.update({f"before_{probe}": before[split][probe], f"after_{probe}": after[probe],
                        f"{probe}_delta": after[probe] - before[split][probe]})
        path = out / f"{name}_{split}_probe.jsonl"
        write_rows(path, [row])
        return [path]

    previous = None
    for seed in seeds:
        for variant in cfg["variants"]:
            name = f"{variant}_seed{seed}"
            if name not in selected_runs:
                continue
            adapter = train_out / "adapters" / name
            while not (adapter / "train_metrics.json").exists():
                train = json.loads((train_out / "manifest.json").read_text())
                if any(j["status"] == "failed" for j in train["jobs"].values()):
                    raise RuntimeError("Training queue failed; stopping evaluation wait")
                print(f"Waiting for {name}", flush=True)
                time.sleep(30)
            if previous is None:
                model = PeftModel.from_pretrained(model, adapter, adapter_name=name).eval()
            else:
                model.load_adapter(adapter, adapter_name=name)
                model.set_adapter(name)
                model.delete_adapter(previous)
                model.eval()
            previous = name
            for split in splits:
                record(f"{name}_{split}", lambda split=split: generation(name, split))
                if split in probe_data:
                    record(f"{name}_{split}_probe", lambda split=split: probe_adapter(name, split))
            if not a.skip_summary:
                subprocess.run([sys.executable, "scripts/summarize_stage1.py", "--config", a.config, "--output-dir", str(out)], check=True)
    state["finished"] = datetime.now(timezone.utc).isoformat()
    save()
    print(f"All requested evaluations complete: seeds {seeds}", flush=True)


if __name__ == "__main__":
    main()
