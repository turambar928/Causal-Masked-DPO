"""Validate cached base-model log probabilities against the exact input token IDs."""
import hashlib
import json
from pathlib import Path

import torch


def identity(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def attach_reference_cache(rows, collator, path, model, normalize_rejected):
    cache = torch.load(path, map_location="cpu", weights_only=True)
    config_hash = hashlib.sha256((Path(model) / "config.json").read_bytes()).hexdigest()
    if cache["model_config_sha256"] != config_hash or cache["model"] != str(Path(model).resolve()):
        raise ValueError("Reference cache model mismatch")
    output = []
    for row in rows:
        row = dict(row)
        for side in ["chosen", "rejected", "positive"]:
            text = row.get("positive_prefix", "") if side == "positive" else row[side]
            if side == "positive" and "positive_prefix" not in row:
                continue
            weights = None
            if side == "rejected":
                from cmdpo.collator import rejected_token_weights
                weights = rejected_token_weights(collator.tokenizer, row)
            encoded = collator._encode_pair(row["prompt"], text, weights, complete=side != "positive")
            key = identity(encoded["input_ids"])
            if key not in cache["sequences"]:
                raise ValueError(f"Reference cache missing exact {side} sequence")
            logps = cache["sequences"][key]
            mask = torch.tensor(encoded["response_mask"][1:], dtype=torch.float32)
            if logps.shape != mask.shape:
                raise ValueError("Reference cache length mismatch")
            value = (logps * mask).sum()
            if side == "rejected" and normalize_rejected:
                value = value / mask.sum().clamp_min(1)
            row[f"{side}_ref_logps"] = float(value)
        output.append(row)
    return output
