#!/usr/bin/env python
import argparse
import hashlib
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmdpo.collator import CMDPOCollator, _pad_1d
from cmdpo.data import read_jsonl
from cmdpo.loss import token_logprobs
from cmdpo.reference_cache import identity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    collator = CMDPOCollator(tokenizer, max_length=768, use_chat_template=True, strict_length=True)
    sequences = {}
    for row in read_jsonl(a.data):
        for text, complete in [(row["chosen"], True), (row["rejected"], True),
                               ("\n".join(row["rejected_steps"][:row["first_error_step"]]), False)]:
            seq = collator._encode_pair(row["prompt"], text, complete=complete)["input_ids"]
            sequences[identity(seq)] = seq
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                               device_map="auto", local_files_only=True).eval()
    cached = {}
    entries = list(sequences.items())
    with torch.inference_mode():
        for start in range(0, len(entries), a.batch_size):
            batch = entries[start:start + a.batch_size]
            ids = _pad_1d([seq for _, seq in batch], tokenizer.pad_token_id).to(model.device)
            attention = _pad_1d([[1] * len(seq) for _, seq in batch], 0).to(model.device)
            logps = token_logprobs(model, ids, attention).cpu()
            for j, (key, seq) in enumerate(batch):
                cached[key] = logps[j, :len(seq) - 1].clone()
            if start % 256 == 0:
                print(f"Cached {min(start + a.batch_size, len(entries))}/{len(entries)}", flush=True)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": str(Path(a.model).resolve()),
                "model_config_sha256": hashlib.sha256((Path(a.model) / "config.json").read_bytes()).hexdigest(),
                "data_sha256": hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
                "sequences": cached}, a.output)
    print(f"Saved {len(cached)} exact sequences", flush=True)


if __name__ == "__main__":
    main()
