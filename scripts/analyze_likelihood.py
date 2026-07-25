#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmdpo.collator import CMDPOCollator
from cmdpo.data import read_jsonl, write_jsonl
from cmdpo.loss import masked_sequence_logps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=1536)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    rows = read_jsonl(args.data)
    collator = CMDPOCollator(tokenizer=tokenizer, max_length=args.max_length)
    outputs = []
    with torch.no_grad():
        for row in rows:
            batch = collator([row])
            batch = {k: v.to(model.device) for k, v in batch.items()}
            rejected_logp = masked_sequence_logps(
                model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_response_mask"],
            )
            out = dict(row)
            out["weighted_rejected_logp"] = float(rejected_logp.item())
            outputs.append(out)
    write_jsonl(args.output, outputs)
    print(f"Wrote likelihood analysis to {args.output}")


if __name__ == "__main__":
    main()
