from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from cmdpo.segmentation import find_step_char_spans
from cmdpo.prompting import format_prompt


def response_token_weights(
    tokenizer: Any,
    response: str,
    steps: list[str],
    step_weights: list[float],
) -> list[float]:
    """Expand step-level weights to response token weights."""
    tokenized = tokenizer(
        response,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = tokenized["offset_mapping"]
    token_weights = [0.0] * len(offsets)
    spans = find_step_char_spans(response, steps)

    if len(spans) != len(step_weights):
        raise ValueError("Step spans and weights must have the same length")
    # Assign separators to the preceding step, so vanilla weights cover ALL tokens.
    for step_idx, ((start, end), weight) in enumerate(zip(spans, step_weights)):
        start = 0 if step_idx == 0 else start
        end = spans[step_idx + 1][0] if step_idx + 1 < len(spans) else len(response)
        for idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_end <= start or tok_start >= end:
                continue
            token_weights[idx] = float(weight)
    return token_weights


def _pad_1d(sequences: list[list[int]], pad_value: int) -> torch.Tensor:
    max_len = max(len(seq) for seq in sequences)
    out = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
    for i, seq in enumerate(sequences):
        out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


def rejected_token_weights(tokenizer: Any, item: dict[str, Any]) -> list[float]:
    """One mask resolution rule shared by the policy and reference cache."""
    weights = item.get("rejected_token_weights")
    if weights is not None:
        return list(weights)
    return response_token_weights(tokenizer, item["rejected"], item["rejected_steps"], item["step_weights"])


def _pad_float(sequences: list[list[float]], pad_value: float = 0.0) -> torch.Tensor:
    max_len = max(len(seq) for seq in sequences)
    out = torch.full((len(sequences), max_len), pad_value, dtype=torch.float)
    for i, seq in enumerate(sequences):
        out[i, : len(seq)] = torch.tensor(seq, dtype=torch.float)
    return out


@dataclass
class CMDPOCollator:
    tokenizer: Any
    max_length: int = 1536
    use_chat_template: bool = False
    strict_length: bool = False

    def _encode_pair(
        self,
        prompt: str,
        response: str,
        response_weights: list[float] | None = None,
        complete: bool = True,
    ) -> dict[str, list[int] | list[float]]:
        formatted = format_prompt(self.tokenizer, prompt, self.use_chat_template)
        prompt_ids = self.tokenizer(formatted, add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]
        if response_weights is None:
            response_weights = [1.0] * len(response_ids)
        if len(response_weights) != len(response_ids):
            raise ValueError("response_weights length must match response token length")
        if self.use_chat_template and complete and response_ids:
            response_ids = response_ids + [self.tokenizer.eos_token_id]
            response_weights = response_weights + [response_weights[-1]]

        input_ids = prompt_ids + response_ids
        response_mask = [0.0] * len(prompt_ids) + response_weights
        attention_mask = [1] * len(input_ids)

        if len(input_ids) > self.max_length:
            if self.strict_length:
                raise ValueError(f"Sequence length {len(input_ids)} exceeds {self.max_length}; refusing truncation")
            input_ids = input_ids[: self.max_length]
            response_mask = response_mask[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
        }

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        chosen_encoded = []
        rejected_encoded = []
        positive_encoded = []
        has_positive = False
        for item in features:
            chosen_encoded.append(self._encode_pair(item["prompt"], item["chosen"]))
            rejected_weights = rejected_token_weights(self.tokenizer, item)
            rejected_encoded.append(self._encode_pair(item["prompt"], item["rejected"], rejected_weights))
            positive_prefix = item.get("positive_prefix")
            if positive_prefix:
                positive_encoded.append(self._encode_pair(item["prompt"], positive_prefix, complete=False))
                has_positive = True
            else:
                positive_encoded.append(self._encode_pair(item["prompt"], "", complete=False))

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        batch: dict[str, torch.Tensor] = {}
        for prefix, encoded in (("chosen", chosen_encoded), ("rejected", rejected_encoded)):
            batch[f"{prefix}_input_ids"] = _pad_1d([x["input_ids"] for x in encoded], pad_id)
            batch[f"{prefix}_attention_mask"] = _pad_1d([x["attention_mask"] for x in encoded], 0)
            batch[f"{prefix}_response_mask"] = _pad_float([x["response_mask"] for x in encoded], 0.0)
        if has_positive:
            batch["positive_input_ids"] = _pad_1d([x["input_ids"] for x in positive_encoded], pad_id)
            batch["positive_attention_mask"] = _pad_1d([x["attention_mask"] for x in positive_encoded], 0)
            batch["positive_response_mask"] = _pad_float([x["response_mask"] for x in positive_encoded], 0.0)
        for key in ["chosen_ref_logps", "rejected_ref_logps", "positive_ref_logps"]:
            if key in features[0]:
                batch[key] = torch.tensor([item[key] for item in features], dtype=torch.float32)
        return batch
