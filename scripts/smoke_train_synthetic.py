#!/usr/bin/env python
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmdpo.loss import cmdpo_loss, masked_sequence_logps


def make_batch(rejected_weights: list[float], batch_size: int = 16) -> dict[str, torch.Tensor]:
    # prompt = [1, 2]
    # chosen response = [30, 31, 32, 33]
    # rejected response = [10, 11, 20, 21]
    # rejected prefix [10, 11] is treated as correct-but-different reasoning.
    chosen = [1, 2, 30, 31, 32, 33]
    rejected = [1, 2, 10, 11, 20, 21]
    chosen_mask = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    rejected_mask = [0.0, 0.0] + rejected_weights

    return {
        "chosen_input_ids": torch.tensor([chosen] * batch_size, dtype=torch.long),
        "chosen_attention_mask": torch.ones(batch_size, len(chosen), dtype=torch.long),
        "chosen_response_mask": torch.tensor([chosen_mask] * batch_size, dtype=torch.float),
        "rejected_input_ids": torch.tensor([rejected] * batch_size, dtype=torch.long),
        "rejected_attention_mask": torch.ones(batch_size, len(rejected), dtype=torch.long),
        "rejected_response_mask": torch.tensor([rejected_mask] * batch_size, dtype=torch.float),
    }


def span_logp(model: GPT2LMHeadModel, token_weights: list[float]) -> float:
    batch = make_batch(token_weights, batch_size=1)
    with torch.no_grad():
        value = masked_sequence_logps(
            model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_response_mask"],
        )
    return float(value.item())


def train_variant(name: str, rejected_weights: list[float]) -> dict[str, float]:
    torch.manual_seed(7)
    config = GPT2Config(
        vocab_size=64,
        n_positions=16,
        n_ctx=16,
        n_embd=64,
        n_layer=2,
        n_head=4,
        bos_token_id=0,
        eos_token_id=0,
    )
    policy = GPT2LMHeadModel(config)
    ref = copy.deepcopy(policy)
    ref.eval()
    for param in ref.parameters():
        param.requires_grad_(False)

    prefix_mask = [1.0, 1.0, 0.0, 0.0]
    error_mask = [0.0, 0.0, 1.0, 1.0]
    before_prefix = span_logp(policy, prefix_mask)
    before_error = span_logp(policy, error_mask)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4)
    batch = make_batch(rejected_weights, batch_size=32)
    for _ in range(150):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = cmdpo_loss(policy, ref, batch, beta=0.1)
        loss.backward()
        optimizer.step()

    after_prefix = span_logp(policy, prefix_mask)
    after_error = span_logp(policy, error_mask)
    return {
        "name": name,
        "prefix_delta": after_prefix - before_prefix,
        "error_delta": after_error - before_error,
        "before_prefix": before_prefix,
        "after_prefix": after_prefix,
        "before_error": before_error,
        "after_error": after_error,
    }


def main() -> None:
    variants = [
        ("vanilla_dpo", [1.0, 1.0, 1.0, 1.0]),
        ("cm_dpo", [0.0, 0.0, 1.0, 0.5]),
        ("first_error_only", [0.0, 0.0, 1.0, 0.0]),
    ]
    print("variant,prefix_delta,error_delta,before_prefix,after_prefix,before_error,after_error")
    for name, weights in variants:
        result = train_variant(name, weights)
        print(
            f"{result['name']},"
            f"{result['prefix_delta']:.4f},"
            f"{result['error_delta']:.4f},"
            f"{result['before_prefix']:.4f},"
            f"{result['after_prefix']:.4f},"
            f"{result['before_error']:.4f},"
            f"{result['after_error']:.4f}"
        )


if __name__ == "__main__":
    main()
