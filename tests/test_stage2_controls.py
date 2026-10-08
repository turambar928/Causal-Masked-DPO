import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from cmdpo.collator import CMDPOCollator, rejected_token_weights
from cmdpo.controls import control_weights
from cmdpo.loss import cmdpo_loss, reference_logps, token_logprobs
from cmdpo.reference_cache import attach_reference_cache, identity


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(12, 8)
        self.rnn = torch.nn.GRU(8, 8, batch_first=True)
        self.head = torch.nn.Linear(8, 12)

    def forward(self, input_ids, attention_mask):
        hidden, _ = self.rnn(self.embed(input_ids))
        return SimpleNamespace(logits=self.head(hidden))


def test_exact_mass_and_shuffle_boundaries():
    w = [0., 0., 1., 1., .25, .25, .0625, .0625]
    for variant in ["mass_matched_uniform", "prefix_preserving_shuffle"]:
        out = control_weights(w, variant, "sample")
        assert sum(out) + out[-1] == pytest.approx(sum(w) + w[-1])
        assert out == control_weights(w, variant, "sample")
        if variant == "prefix_preserving_shuffle":
            assert out[:2] == [0., 0.] and out[-1] == w[-1]
            assert sorted(out) == sorted(w) and out != w
        else:
            assert len(set(out)) == 1
    assert control_weights([0., 1., 1.], "prefix_preserving_shuffle", "degenerate") == [0., 1., 1.]
    for invalid in [[], [float("nan")], [-1.]]:
        with pytest.raises(ValueError):
            control_weights(invalid, "mass_matched_uniform", "x")


class NumberTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, **kwargs):
        return {"input_ids": [int(c) for c in text], "offset_mapping": [(i, i+1) for i in range(len(text))]}


def test_custom_masks_cache_loss_and_gradients(tmp_path):
    torch.manual_seed(123)
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}")
    collator = CMDPOCollator(NumberTokenizer())
    row = {"prompt": "2", "chosen": "345", "rejected": "367", "rejected_steps": ["367"],
           "step_weights": [1.], "rejected_token_weights": [0., 1., .25]}
    ref, policy = TinyLM().eval(), TinyLM().eval()
    cache = {"model": str(model_path.resolve()), "model_config_sha256": hashlib.sha256(b"{}").hexdigest(), "sequences": {}}
    for side in ["chosen", "rejected"]:
        encoded = collator._encode_pair(row["prompt"], row[side])
        ids = torch.tensor([encoded["input_ids"]])
        cache["sequences"][identity(encoded["input_ids"])] = token_logprobs(ref, ids, torch.ones_like(ids))[0].detach()
    path = tmp_path / "cache.pt"
    torch.save(cache, path)
    for custom in [False, True]:
        item = copy.deepcopy(row)
        if not custom:
            item.pop("rejected_token_weights")
        assert rejected_token_weights(collator.tokenizer, item) == ([0., 1., .25] if custom else [1., 1., 1.])
        for normalize in [False, True]:
            cached = attach_reference_cache([item], collator, path, model_path, normalize)
            outputs = []
            for reference, data in [(ref, [item]), (None, cached)]:
                policy.zero_grad()
                loss, _ = cmdpo_loss(policy, reference, collator(data), normalize_rejected=normalize)
                loss.backward()
                outputs.append((loss.detach(), torch.cat([p.grad.flatten() for p in policy.parameters()])))
            for live, saved in zip(outputs[0], outputs[1]):
                torch.testing.assert_close(live, saved, rtol=1e-6, atol=1e-6)


def test_common_prefix_cancels_scores_and_parameter_gradients():
    torch.manual_seed(314)
    policy, ref = TinyLM().eval(), TinyLM().eval()
    batch = {}
    for side, ids in [("chosen", [2, 3, 4, 5]), ("rejected", [2, 3, 6, 7])]:
        batch[f"{side}_input_ids"] = torch.tensor([ids])
        batch[f"{side}_attention_mask"] = torch.ones(1, 4, dtype=torch.long)
        batch[f"{side}_response_mask"] = torch.tensor([[0., 1., 1., 1.]])
    trimmed = copy.deepcopy(batch)
    for side in ["chosen", "rejected"]:
        trimmed[f"{side}_response_mask"][0, 1] = 0.
    outputs = []
    for data in [batch, trimmed]:
        policy.zero_grad()
        loss, metrics = cmdpo_loss(policy, ref, data)
        loss.backward()
        outputs.append((loss.detach(), metrics["reward_margin"], torch.cat([p.grad.flatten() for p in policy.parameters()])))
    for full, partial in zip(*outputs):
        torch.testing.assert_close(full, partial, rtol=1e-5, atol=1e-6)
    assert outputs[0][-1].norm() > 0  # cancellation does not remove suffix learning


def test_fresh_tests_are_balanced_disjoint_and_repeatable():
    from scripts.stage2 import fresh_tests
    cfg = {"data_seed": 20260930, "eval_rows_per_split": 20, "gamma": .25}
    initial = fresh_tests(cfg, set())
    excluded = {r["prompt"] for rows in initial.values() for r in rows}
    second = fresh_tests(cfg, excluded)
    assert second == fresh_tests(cfg, excluded)
    prompts = [r["prompt"] for rows in second.values() for r in rows]
    assert len(prompts) == len(set(prompts)) == 40
    assert not excluded.intersection(prompts)
    assert {r["metadata"]["template"] for r in second["id"]}.isdisjoint({r["metadata"]["template"] for r in second["ood"]})


def test_primary_bootstrap_constant_difference_and_reproducibility():
    import numpy as np
    from scripts.analyze_stage2 import bootstrap
    assert bootstrap(np.zeros((3, 20)), .975, replicates=100) == [0., 0.]
    assert bootstrap(np.ones((3, 20)), .975, replicates=100) == [1., 1.]
    delta = np.random.default_rng(99).integers(-1, 2, (3, 20))
    assert bootstrap(delta, .975, replicates=100) == bootstrap(delta, .975, replicates=100)


def test_bootstrap_rejects_unpaired_shape_or_invalid_level():
    import numpy as np
    from scripts.analyze_stage2 import bootstrap
    with pytest.raises(ValueError):
        bootstrap(np.zeros(20), .975)
    with pytest.raises(ValueError):
        bootstrap(np.zeros((3, 20)), 1.)
