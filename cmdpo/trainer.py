from __future__ import annotations

from typing import Any

import torch
from transformers import Trainer

from cmdpo.loss import cmdpo_loss


class CMDPOTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        ref_model: torch.nn.Module,
        beta: float = 0.1,
        normalize_rejected: bool = False,
        process_positive_weight: float = 0.0,
        chosen_nll_weight: float = 0.0,
        objective: str = "dpo",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.beta = beta
        self.normalize_rejected = normalize_rejected
        self.process_positive_weight = process_positive_weight
        self.chosen_nll_weight = chosen_nll_weight
        self.objective = objective
        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad_(False)

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss, metrics = cmdpo_loss(
            model,
            self.ref_model,
            inputs,
            beta=self.beta,
            normalize_rejected=self.normalize_rejected,
            process_positive_weight=self.process_positive_weight,
            chosen_nll_weight=self.chosen_nll_weight,
            objective=self.objective,
        )
        if return_outputs:
            return loss, metrics
        return loss
