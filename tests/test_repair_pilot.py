import json
from pathlib import Path

import numpy as np
import pytest

from scripts.repair_pilot import (arithmetic, batch_seed, build_data, completed_rows,
                                 continuation_ids, equation, grade, inject_error, jobs_for,
                                 sha, validate_rows, write_rows)
from scripts.analyze_repair_pilot import (decide, paired_bootstrap, residual_rank_correlation,
                                         reproducibility)


class CharTokenizer:
    eos_token_id = 1
    chat_template = "test"

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text]}

    def apply_chat_template(self, messages, **kwargs):
        return "<user>" + messages[0]["content"] + "<assistant>"


def config():
    return json.loads(Path("configs/repair_pilot_v1.json").read_text())


def test_arithmetic_and_single_result_injection():
    assert arithmetic("12 * 3 - 4 + 2") == 34
    assert arithmetic("-3 + 2") == -1
    with pytest.raises(ValueError):
        arithmetic("pow(2, 3)")
    step = "There are 12 * 3 = 36 items."
    assert inject_error(step, 1) == "There are 12 * 3 = 37 items."
    assert inject_error(step, -1) == "There are 12 * 3 = 35 items."
    with pytest.raises(ValueError):
        inject_error("12 * 3 = 35", 1)


def test_data_balance_freshness_and_conditions():
    cfg = config()
    questions, cases = build_data(cfg, set())
    assert (questions, cases) == build_data(cfg, set())
    assert len(questions) == len({q["prompt"] for q in questions}) == 96
    excluded = {q["prompt"] for q in questions[:10]}
    new, _ = build_data(cfg, excluded)
    assert not excluded & {q["prompt"] for q in new}
    for template in {q["template"] for q in questions}:
        for position in (0, 1):
            selected = [c for c in cases if c["template"] == template and c["position"] == position]
            assert len(selected) == 16
            assert sum(c["delta"] == 1 for c in selected) == 8
    for c in cases:
        clean, bad, repair, sham = [c["prefixes"][k] for k in "CBRS"]
        assert bad == clean.replace(c["correct_step"], c["wrong_step"])
        assert repair == bad + cfg["correction_template"].format(step=c["correct_step"])
        assert sham == bad + cfg["correction_template"].format(step=c["wrong_step"])
        assert all("####" not in p for p in c["prefixes"].values())
        m = equation(c["wrong_step"])
        assert arithmetic(m.group(1)) != int(m.group(2))


def test_continuation_is_partial_assistant_and_grade_never_uses_prefix():
    tokenizer = CharTokenizer()
    ids = continuation_ids(tokenizer, "question", "12 * 3 = 36\n")
    assert ids[-1] == ord("\n") and tokenizer.eos_token_id not in ids
    assert not grade("", "#### 36")["correct"]
    assert grade("#### 36", "#### 36") == {"correct": True, "has_explicit_answer": True}
    with pytest.raises(ValueError):
        continuation_ids(tokenizer, "question", chr(tokenizer.eos_token_id))


def test_jobs_and_bank_seeds_are_frozen():
    cfg = config()
    questions, cases = build_data(cfg, set())
    jobs = jobs_for(questions, cases, cfg)
    assert len(jobs) == 432 and sum(len(s) for _, s in jobs) == 13824
    assert all(len(s) == 32 for _, s in jobs)
    assert len({batch_seed(cfg, job) for job, _ in jobs}) == 432
    assert jobs == jobs_for(questions, cases, cfg)
    assert batch_seed(cfg, jobs[0][0]) != batch_seed(cfg, jobs[1][0])


def test_resume_validates_checksum_and_regrading(tmp_path):
    cfg = config()
    spec = {"gold": "#### 1"}
    row = {**spec, "job_id": "test", "batch_seed": batch_seed(cfg, "test"),
           "prediction": "#### 1", "generated_tokens": 3, "stopped_on_eos": True,
           "hit_token_limit": False, **grade("#### 1", spec["gold"])}
    path = tmp_path / "batch.jsonl"
    write_rows(path, [row])
    entry = {"status": "complete", "path": str(path), "sha256": sha(path)}
    assert completed_rows(entry, [spec], cfg, "test") == [row]
    bad = {**row, "correct": False}
    with pytest.raises(AssertionError):
        validate_rows([bad], [spec], cfg, "test")
    write_rows(path, [bad])
    with pytest.raises(AssertionError):
        completed_rows(entry, [spec], cfg, "test")


def test_cluster_statistics_constant_and_reproducible():
    labels = np.array(["a"] * 8 + ["b"] * 8)
    delta = np.full((16, 2), .25)
    assert paired_bootstrap(delta, labels, .95, 1, 50) == [.25, .25]
    rng = np.random.default_rng(1)
    bank = rng.normal(size=(16, 2))
    gains = np.stack([bank, bank], axis=-1)
    assert residual_rank_correlation(gains, labels) == pytest.approx(1.)
    first = reproducibility(gains, labels, 2, 50)
    assert first == reproducibility(gains, labels, 2, 50)
    assert first["ci"][0] == pytest.approx(1.)
    assert reproducibility(np.zeros((16, 2, 2)), labels, 2, 50)["ci"] is None


def test_decision_gates():
    cfg = config()
    rates, trunc = {"C": .8}, {"C": 0, "B": 0, "R": 0, "S": 0, "F": 0}
    comparisons = {k: {"ci": [.1, .4], "difference": .2,
                       "reproducibility": {"ci": [.1, .5]}} for k in ("C-B", "R-B", "R-S")}
    assert decide(cfg, rates, trunc, comparisons) == "worth_followup_not_training_evidence"
    assert decide(cfg, rates, {**trunc, "F": .051}, comparisons) == "protocol_uninterpretable"
    assert decide(cfg, {"C": .19}, trunc, comparisons) == "small_model_capability_insufficient"
    comparisons["R-S"]["reproducibility"]["ci"] = [-.1, .2]
    assert decide(cfg, rates, trunc, comparisons) == "repair_gain_without_stable_heterogeneity"
    comparisons["C-B"]["ci"] = [-.1, .4]
    assert decide(cfg, rates, trunc, comparisons) == "no_reliable_full_mechanism_support"
