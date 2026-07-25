#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmdpo.collator import CMDPOCollator
from cmdpo.data import read_jsonl
from cmdpo.loss import cmdpo_loss, masked_sequence_logps


class WhitespaceTokenizer:
    def __init__(self, texts: list[str]) -> None:
        vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
        for text in texts:
            for token, _, _ in self._scan(text):
                if token not in vocab:
                    vocab[token] = len(vocab)
        self.vocab = vocab
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    @staticmethod
    def _scan(text: str) -> list[tuple[str, int, int]]:
        return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    def __len__(self) -> int:
        return len(self.vocab)

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        **_: Any,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        pieces = self._scan(text)
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok, _, _ in pieces]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        out: dict[str, list[int] | list[tuple[int, int]]] = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(start, end) for _, start, end in pieces]
        return out


def variant_rows(rows: list[dict[str, Any]], variant: str, gamma: float) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        n = len(item["rejected_steps"])
        m = int(item["first_error_step"])
        if variant == "vanilla":
            item["step_weights"] = [1.0] * n
        elif variant == "cm":
            item["step_weights"] = [0.0 if k < m else (1.0 if k == m else gamma ** (k - m)) for k in range(n)]
        elif variant == "first_error_only":
            item["step_weights"] = [1.0 if k == m else 0.0 for k in range(n)]
        elif variant == "prefix_probe":
            item["step_weights"] = [1.0 if k < m else 0.0 for k in range(n)]
        elif variant == "error_probe":
            item["step_weights"] = [1.0 if k == m else 0.0 for k in range(n)]
        elif variant == "suffix_probe":
            item["step_weights"] = [1.0 if k > m else 0.0 for k in range(n)]
        else:
            raise ValueError(variant)
        out.append(item)
    return out


def mean_rejected_logp(model: GPT2LMHeadModel, collator: CMDPOCollator, rows: list[dict[str, Any]]) -> float:
    values = []
    with torch.no_grad():
        for row in rows:
            batch = collator([row])
            value = masked_sequence_logps(
                model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_response_mask"],
            )
            values.append(float(value.item()))
    return sum(values) / max(1, len(values))


def train_one(
    base_model: GPT2LMHeadModel,
    ref_model: GPT2LMHeadModel,
    collator: CMDPOCollator,
    rows: list[dict[str, Any]],
    variant: str,
    gamma: float,
    steps: int,
) -> dict[str, float | str]:
    policy = copy.deepcopy(base_model)
    ref = copy.deepcopy(ref_model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    train_rows = variant_rows(rows, variant, gamma)
    prefix_rows = variant_rows(rows, "prefix_probe", gamma)
    error_rows = variant_rows(rows, "error_probe", gamma)
    suffix_rows = variant_rows(rows, "suffix_probe", gamma)

    before_prefix = mean_rejected_logp(policy, collator, prefix_rows)
    before_error = mean_rejected_logp(policy, collator, error_rows)
    before_suffix = mean_rejected_logp(policy, collator, suffix_rows)

    opt = torch.optim.AdamW(policy.parameters(), lr=5e-4)
    for step in range(steps):
        row = train_rows[step % len(train_rows)]
        batch = collator([row])
        opt.zero_grad(set_to_none=True)
        loss, _ = cmdpo_loss(policy, ref, batch, beta=0.1)
        loss.backward()
        opt.step()

    after_prefix = mean_rejected_logp(policy, collator, prefix_rows)
    after_error = mean_rejected_logp(policy, collator, error_rows)
    after_suffix = mean_rejected_logp(policy, collator, suffix_rows)
    return {
        "variant": variant,
        "prefix_delta": after_prefix - before_prefix,
        "error_delta": after_error - before_error,
        "suffix_delta": after_suffix - before_suffix,
        "before_prefix": before_prefix,
        "after_prefix": after_prefix,
        "before_error": before_error,
        "after_error": after_error,
        "before_suffix": before_suffix,
        "after_suffix": after_suffix,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--gamma", type=float, default=0.5)
    args = parser.parse_args()

    rows = read_jsonl(args.data)
    texts = []
    for row in rows:
        texts.extend([row["prompt"], row["chosen"], row["rejected"]])
    tokenizer = WhitespaceTokenizer(texts)
    collator = CMDPOCollator(tokenizer=tokenizer, max_length=256)

    torch.manual_seed(11)
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=256,
        n_ctx=256,
        n_embd=96,
        n_layer=2,
        n_head=4,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    base = GPT2LMHeadModel(config)
    ref = copy.deepcopy(base)

    print("variant,prefix_delta,error_delta,suffix_delta,before_prefix,after_prefix,before_error,after_error,before_suffix,after_suffix")
    for variant in ["vanilla", "cm", "first_error_only"]:
        result = train_one(base, ref, collator, rows, variant, args.gamma, args.steps)
        print(
            f"{result['variant']},"
            f"{result['prefix_delta']:.4f},"
            f"{result['error_delta']:.4f},"
            f"{result['suffix_delta']:.4f},"
            f"{result['before_prefix']:.4f},"
            f"{result['after_prefix']:.4f},"
            f"{result['before_error']:.4f},"
            f"{result['after_error']:.4f},"
            f"{result['before_suffix']:.4f},"
            f"{result['after_suffix']:.4f}"
        )


if __name__ == "__main__":
    main()
