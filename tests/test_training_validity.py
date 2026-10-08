"""Regression tests for the controls used as scientific evidence."""
import copy
from types import SimpleNamespace

import torch

from cmdpo.loss import cmdpo_loss
from cmdpo.collator import CMDPOCollator, response_token_weights
from cmdpo.prompting import format_prompt


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(12, 8)
        self.head = torch.nn.Linear(8, 12)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


def test_positive_term_changes_policy_gradient_but_never_reference_gradient():
    torch.manual_seed(9)
    policy = TinyLM()
    ref = copy.deepcopy(policy)
    batch = {}
    for name, ids in [("chosen", [1, 2, 3]), ("rejected", [1, 4, 5]), ("positive", [1, 6, 7])]:
        batch[f"{name}_input_ids"] = torch.tensor([ids])
        batch[f"{name}_attention_mask"] = torch.ones(1, 3, dtype=torch.long)
        batch[f"{name}_response_mask"] = torch.tensor([[0., 1., 1.]])
    grads = []
    for weight in [0., 0.2]:
        policy.zero_grad()
        loss, _ = cmdpo_loss(policy, ref, batch, process_positive_weight=weight)
        loss.backward()
        grads.append(torch.cat([p.grad.flatten() for p in policy.parameters()]))
    assert torch.linalg.vector_norm(grads[1] - grads[0]) > 1e-4
    assert all(p.grad is None for p in ref.parameters())


class CharTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    chat_template = "test"

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text],
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def apply_chat_template(self, messages, **kwargs):
        return "<user>" + messages[0]["content"] + "<assistant>"


def test_vanilla_covers_newlines_and_masked_prefix_stays_zero():
    t = CharTokenizer()
    response = "first\nsecond"
    assert response_token_weights(t, response, ["first", "second"], [1, 1]) == [1.] * len(response)
    assert response_token_weights(t, response, ["first", "second"], [0, 1]) == [0.] * 6 + [1.] * 6


def test_chat_prefix_equals_generation_and_partial_prefix_has_no_eos():
    t = CharTokenizer()
    c = CMDPOCollator(t, use_chat_template=True)
    prefix = t(format_prompt(t, "2+2?", True))["input_ids"]
    result = c._encode_pair("2+2?", "four")
    assert result["input_ids"][:len(prefix)] == prefix
    assert result["response_mask"][:len(prefix)] == [0.] * len(prefix)
    assert result["input_ids"][-1] == t.eos_token_id
    assert c._encode_pair("2+2?", "four", complete=False)["input_ids"][-1] == ord("r")


def test_strict_length_refuses_silent_error_truncation():
    import pytest
    with pytest.raises(ValueError, match="refusing truncation"):
        CMDPOCollator(CharTokenizer(), max_length=3, strict_length=True)._encode_pair("question", "answer")


def test_cached_reference_matches_live_loss_and_gradient():
    from cmdpo.loss import reference_logps
    torch.manual_seed(123)
    policy, ref = TinyLM(), TinyLM()
    batch = {}
    for name, ids in [("chosen", [1, 2, 3]), ("rejected", [1, 4, 5]), ("positive", [1, 6, 7])]:
        batch[f"{name}_input_ids"] = torch.tensor([ids])
        batch[f"{name}_attention_mask"] = torch.ones(1, 3, dtype=torch.long)
        batch[f"{name}_response_mask"] = torch.tensor([[0., 1., .25]])
    cached = dict(batch)
    values = reference_logps(ref, batch, normalize_rejected=True, process_positive_weight=.2)
    for side, value in zip(["chosen", "rejected", "positive"], values):
        cached[f"{side}_ref_logps"] = value
    outputs = []
    for reference, data in [(ref, batch), (None, cached)]:
        policy.zero_grad()
        loss, _ = cmdpo_loss(policy, reference, data, normalize_rejected=True, process_positive_weight=.2)
        loss.backward()
        outputs.append((loss.detach(), torch.cat([p.grad.flatten() for p in policy.parameters()])))
    for live, frozen in zip(outputs[0], outputs[1]):
        torch.testing.assert_close(live, frozen, rtol=0, atol=0)


def test_template_split_is_disjoint_and_reproducible():
    from scripts.prepare_stage1 import build_splits
    splits = build_splits(40, 20)
    assert splits == build_splits(40, 20)
    prompts = [r["prompt"] for rows in splits.values() for r in rows]
    assert len(prompts) == len(set(prompts))
    assert {r["metadata"]["template"] for r in splits["train"]}.isdisjoint(
        {r["metadata"]["template"] for r in splits["ood"]})
