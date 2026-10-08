#!/usr/bin/env python
"""Fail closed unless the full frozen experiment matrix is complete and consistent."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmdpo.data import read_jsonl
from cmdpo.verifier import verify_answer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stage1_v1.json")
    p.add_argument("--eval-config", default="configs/stage1_eval_v2.json")
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    ecfg = json.loads(Path(a.eval_config).read_text())
    root, out = Path(cfg["output_dir"]), Path(ecfg["output_dir"])
    train = json.loads((root / "manifest.json").read_text())
    evaluation = json.loads((out / "evaluation_manifest.json").read_text())
    assert train["config_sha256"] == sha(a.config)
    assert evaluation["config_sha256"] == sha(a.eval_config)
    assert evaluation["training_config_sha256"] == sha(a.config)
    # Orchestration/reporting may evolve, but the numerical experiment must not.
    training_sources = ["cmdpo/collator.py", "cmdpo/loss.py", "cmdpo/trainer.py",
                        "cmdpo/prompting.py", "cmdpo/reference_cache.py", "scripts/train_cmdpo.py"]
    evaluation_sources = ["scripts/evaluate_generation_accuracy.py", "scripts/evaluate_likelihood_deltas.py"]
    source_checksums = {}
    for manifest, sources in [(train, training_sources), (evaluation, evaluation_sources)]:
        for path in sources:
            assert sha(path) == manifest["source_sha256"][path], f"Numerical source changed: {path}"
            source_checksums[path] = sha(path)
    for path in out.glob("evaluation_*_manifest.json"):
        shard = json.loads(path.read_text())
        assert shard["config_sha256"] == evaluation["config_sha256"]
        assert shard["training_config_sha256"] == evaluation["training_config_sha256"]
        for source in training_sources + evaluation_sources:
            assert shard["source_sha256"][source] == source_checksums[source], source
        for name, job in shard["jobs"].items():
            if job["status"] != "complete":
                continue
            previous = evaluation["jobs"].get(name, {})
            if previous.get("status") == "complete":
                assert previous["files"] == job["files"], name
            evaluation["jobs"][name] = job
    assert train["model_sha256"] == sha(Path(cfg["model"]) / "model.safetensors")
    data_dir = Path(cfg["data_dir"])
    data_manifest = json.loads((data_dir / "manifest.json").read_text())
    for split, entry in data_manifest["splits"].items():
        assert sha(data_dir / f"{split}.jsonl") == entry["sha256"]
    for variant, checksum in data_manifest["variants"].items():
        assert sha(data_dir / f"train_{variant}.jsonl") == checksum
    artifacts = {}
    expected_steps = math.ceil(cfg["train_rows"] / (cfg["train_batch_size"] * cfg["gradient_accumulation_steps"]))
    for seed in cfg["seeds"]:
        for variant in cfg["variants"]:
            name = f"{variant}_seed{seed}"
            job = train["jobs"][f"train_{name}"]
            assert job["status"] == "complete", name
            assert sha(job["artifact"]) == job["artifact_sha256"], name
            adapter = root / "adapters" / name
            metrics = json.loads((adapter / "train_metrics.json").read_text())
            state = json.loads((adapter / "trainer_state.json").read_text())
            run = json.loads((adapter / "run_config.json").read_text())
            lora = json.loads((adapter / "adapter_config.json").read_text())
            assert lora["r"] == 16 and lora["lora_alpha"] == 32 and lora["lora_dropout"] == .05, name
            assert set(lora["target_modules"]) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}, name
            assert metrics["epoch"] == cfg["epochs"], name
            assert state["global_step"] == expected_steps, name
            assert run["seed"] == seed and run["use_chat_template"] and run["strict_length"], name
            for actual, expected in [("beta", "beta"), ("max_length", "max_length"),
                                     ("learning_rate", "learning_rate"), ("epochs", "epochs"),
                                     ("per_device_train_batch_size", "train_batch_size"),
                                     ("gradient_accumulation_steps", "gradient_accumulation_steps")]:
                assert run[actual] == cfg[expected], (name, actual)
            assert run["data_sha256"] == data_manifest["variants"][variant], name
            assert run["objective"] == ("sft" if variant == "sft" else "dpo"), name
            assert run["normalize_rejected"] == (variant == "normalized"), name
            assert run["process_positive_weight"] == (cfg["process_positive_weight"] if variant == "process_positive" else 0), name
            assert run["chosen_nll_weight"] == (cfg["chosen_nll_weight"] if variant == "dpo_nll" else 0), name
            artifacts[str(adapter / "adapter_model.safetensors")] = sha(adapter / "adapter_model.safetensors")
            for filename in ["run_config.json", "train_metrics.json", "trainer_state.json", "adapter_config.json"]:
                artifacts[str(adapter / filename)] = sha(adapter / filename)
    total_predictions = 0
    total_probes = 0
    names = ["base"] + [f"{v}_seed{s}" for s in cfg["seeds"] for v in cfg["variants"]]
    for split in ["id", "ood"]:
        gold_rows = read_jsonl(data_dir / f"{split}.jsonl")
        for name in names:
            job = evaluation["jobs"][f"{name}_{split}"]
            assert job["status"] == "complete", (name, split)
            for path, checksum in job["files"].items():
                assert sha(path) == checksum, path
                artifacts[path] = checksum
            rows = read_jsonl(out / f"{name}_{split}_details.jsonl")
            assert len(rows) == len(gold_rows) == cfg["eval_rows_per_split"]
            for i, (row, gold) in enumerate(zip(rows, gold_rows)):
                assert row["index"] == i and row["prompt"] == gold["prompt"] and row["gold"] == gold["answer"]
                assert row["correct"] == verify_answer(row["prediction"], row["gold"])
                assert 0 < row["prediction_tokens"] <= ecfg["max_new_tokens"]
                assert row["stopped_on_eos"] != row["hit_token_limit"]
            summary = read_jsonl(out / f"{name}_{split}_summary.jsonl")[0]
            assert summary["correct"] == sum(r["correct"] for r in rows)
            assert summary["accuracy"] == summary["correct"] / len(rows)
            total_predictions += len(rows)
            if name == "base":
                continue
            probe_job = evaluation["jobs"][f"{name}_{split}_probe"]
            assert probe_job["status"] == "complete"
            for path, checksum in probe_job["files"].items():
                assert sha(path) == checksum
                artifacts[path] = checksum
            probe = read_jsonl(out / f"{name}_{split}_probe.jsonl")[0]
            assert probe["examples"] == cfg["probe_rows_per_split"] and probe["use_chat_template"]
            assert probe["aggregation"] == "mean_nonempty_example_per_token"
            for key in ["prefix", "error", "suffix", "cmdpo"]:
                assert math.isfinite(probe[f"{key}_delta"])
                assert math.isclose(probe[f"after_{key}"] - probe[f"before_{key}"], probe[f"{key}_delta"])
            total_probes += 1
    result = {"status": "passed", "training_runs": len(names) - 1,
              "generation_runs": len(names) * 2, "predictions": total_predictions,
              "heldout_probe_runs": total_probes, "artifact_sha256": artifacts,
              "numerical_source_sha256": source_checksums,
              "scope": "Local Qwen arithmetic only; no API calls; not a GSM8K training benchmark"}
    (out / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print({k: v for k, v in result.items() if k != "artifact_sha256"})


if __name__ == "__main__":
    main()
