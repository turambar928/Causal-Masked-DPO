#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmdpo.collator import CMDPOCollator
from cmdpo.data import read_jsonl, write_jsonl
from cmdpo.loss import masked_sequence_logps


def probe_rows(rows: list[dict[str, Any]], probe: str, gamma: float) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        item = dict(row)
        n = len(item["rejected_steps"])
        m = int(item["first_error_step"])
        if probe == "prefix":
            weights = [1.0 if k < m else 0.0 for k in range(n)]
        elif probe == "error":
            weights = [1.0 if k == m else 0.0 for k in range(n)]
        elif probe == "suffix":
            weights = [1.0 if k > m else 0.0 for k in range(n)]
        elif probe == "cmdpo":
            weights = [0.0 if k < m else (1.0 if k == m else gamma ** (k - m)) for k in range(n)]
        else:
            raise ValueError(probe)
        item["step_weights"] = weights
        output.append(item)
    return output


def mean_probe_logp(
    model: torch.nn.Module,
    collator: CMDPOCollator,
    rows: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    label: str,
) -> float:
    values: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            print(f"[{label}] rows {start + 1}-{end}/{len(rows)}", flush=True)
            batch = collator(rows[start:end])
            batch = {k: v.to(device) for k, v in batch.items()}
            value = masked_sequence_logps(
                model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_response_mask"],
            )
            values.extend(float(x) for x in value.detach().cpu().tolist())
    return sum(values) / max(1, len(values))


def load_model(
    model_path: str,
    adapter_path: str | None,
    dtype: torch.dtype,
    device_map: str | None,
) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--adapter", action="append", default=[], help="name=path")
    args = parser.parse_args()

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
    device_map = "auto" if torch.cuda.is_available() else None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    collator = CMDPOCollator(tokenizer=tokenizer, max_length=args.max_length)
    rows = read_jsonl(args.data)

    probes = {probe: probe_rows(rows, probe, args.gamma) for probe in ["prefix", "error", "suffix", "cmdpo"]}

    base_model = load_model(args.model, None, dtype=dtype, device_map=device_map)
    before = {
        probe: mean_probe_logp(base_model, collator, probe_items, device, args.batch_size, f"base/{probe}")
        for probe, probe_items in probes.items()
    }
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = []
    for adapter_arg in args.adapter:
        if "=" not in adapter_arg:
            raise ValueError("--adapter must use name=path")
        name, path = adapter_arg.split("=", 1)
        model = load_model(args.model, path, dtype=dtype, device_map=device_map)
        after = {
            probe: mean_probe_logp(model, collator, probe_items, device, args.batch_size, f"{name}/{probe}")
            for probe, probe_items in probes.items()
        }
        row = {"variant": name}
        for probe in ["prefix", "error", "suffix", "cmdpo"]:
            row[f"before_{probe}"] = before[probe]
            row[f"after_{probe}"] = after[probe]
            row[f"{probe}_delta"] = after[probe] - before[probe]
        results.append(row)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_jsonl(args.output, results)
    print("variant,prefix_delta,error_delta,suffix_delta,cmdpo_delta")
    for row in results:
        print(
            f"{row['variant']},"
            f"{row['prefix_delta']:.4f},"
            f"{row['error_delta']:.4f},"
            f"{row['suffix_delta']:.4f},"
            f"{row['cmdpo_delta']:.4f}"
        )


if __name__ == "__main__":
    main()
