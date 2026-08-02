from __future__ import annotations

import torch
import torch.nn.functional as F


def token_logprobs(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    logps = F.log_softmax(logits, dim=-1)
    gathered = torch.gather(logps, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return gathered


def masked_sequence_logps(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    logps = token_logprobs(model, input_ids, attention_mask)
    shifted_mask = response_mask[:, 1:].to(logps.dtype)
    shifted_attention = attention_mask[:, 1:].to(logps.dtype)
    weighted = (logps * shifted_mask * shifted_attention).sum(dim=-1)
    if not normalize:
        return weighted
    normalizer = (shifted_mask * shifted_attention).sum(dim=-1).clamp_min(1.0)
    return weighted / normalizer


def cmdpo_loss(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    beta: float = 0.1,
    normalize_rejected: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    chosen_logps = masked_sequence_logps(
        policy_model,
        batch["chosen_input_ids"],
        batch["chosen_attention_mask"],
        batch["chosen_response_mask"],
    )
    rejected_logps = masked_sequence_logps(
        policy_model,
        batch["rejected_input_ids"],
        batch["rejected_attention_mask"],
        batch["rejected_response_mask"],
        normalize=normalize_rejected,
    )
    with torch.no_grad():
        chosen_ref_logps = masked_sequence_logps(
            ref_model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_response_mask"],
        )
        rejected_ref_logps = masked_sequence_logps(
            ref_model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_response_mask"],
            normalize=normalize_rejected,
        )

    chosen_rewards = chosen_logps - chosen_ref_logps
    rejected_rewards = rejected_logps - rejected_ref_logps
    logits = beta * (chosen_rewards - rejected_rewards)
    losses = -F.logsigmoid(logits)

    metrics = {
        "loss": losses.detach().mean(),
        "chosen_reward": chosen_rewards.detach().mean(),
        "rejected_reward": rejected_rewards.detach().mean(),
        "reward_margin": (chosen_rewards - rejected_rewards).detach().mean(),
    }
    return losses.mean(), metrics
