import json
from pathlib import Path

import numpy as np
import pytest

from scripts.repair_pilot import build_data, batch_seed
from scripts.repair_controls import make_cases, assign_folds, jobs_for
from scripts.analyze_repair_controls import crossfit, features, decision, matched_interval


class CharacterTokenizer:
    eos_token_id = 1
    chat_template = "test"

    def __call__(self, text, **kwargs):
        return {"input_ids": list(map(ord, text))}

    def apply_chat_template(self, messages, **kwargs):
        return "<user>"+messages[0]["content"]+"<assistant>"


def cfg():
    return json.loads(Path("configs/repair_controls_v1.json").read_text())


def inputs():
    old_cfg = json.loads(Path("configs/repair_pilot_v1.json").read_text())
    questions, old_cases = build_data(old_cfg, set())
    return questions, old_cases, make_cases(old_cases, cfg(), CharacterTokenizer())


def test_factorial_prefixes_change_only_history_within_cue():
    questions, old_cases, cases = inputs()
    for old, case in zip(old_cases, cases):
        p = case["prefixes"]
        assert p["WC"] == old["prefixes"]["R"]
        assert p["CC"] == old["prefixes"]["C"] + cfg()["correction_template"].format(step=case["correct_step"])
        assert p["WN"] == old["prefixes"]["B"] + cfg()["neutral_template"].format(step=case["correct_step"])
        assert p["CN"] == old["prefixes"]["C"] + cfg()["neutral_template"].format(step=case["correct_step"])
        assert p["CN"].endswith(case["correct_step"]+"\n") and p["WN"].endswith(case["correct_step"]+"\n")
        assert p["WN"].count(case["wrong_step"]) == 1
        assert case["wrong_step"] not in p["CN"]
        assert "####" not in "".join(p.values())


def test_fold_assignment_is_question_disjoint_balanced_and_deterministic():
    questions, _, cases = inputs()
    folds = assign_folds(questions, cfg())
    assert folds == assign_folds(questions, cfg())
    for template in {q["template"] for q in questions}:
        for fold in range(4):
            assert sum(q["template"] == template and folds[q["question_id"]] == fold for q in questions) == 4
    for i in range(0, len(cases), 2):
        assert folds[cases[i]["question_id"]] == folds[cases[i+1]["question_id"]]


def test_new_jobs_have_distinct_rng_and_fixed_budget():
    _, _, cases = inputs()
    jobs = jobs_for(cases, cfg())
    assert len(jobs) == 384 and sum(len(s) for _, s in jobs) == 12288
    assert all(len(s) == 32 for _, s in jobs)
    assert len({batch_seed(cfg(), k) for k, _ in jobs}) == 384
    assert all({s["condition"] for s in specs} == {"CC", "WC", "CN", "WN"} for _, specs in jobs)


def test_held_out_labels_do_not_change_their_own_predictions_or_scalers():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(32, 5)), rng.normal(size=32)
    folds = np.repeat(np.arange(4), 8)
    pred, fits = crossfit(x, y, folds, 10., (-1, 1))
    changed = y.copy()
    changed[folds == 0] = 10000
    pred2, fits2 = crossfit(x, changed, folds, 10., (-1, 1))
    np.testing.assert_array_equal(pred[folds == 0], pred2[folds == 0])
    assert fits[0] == fits2[0]
    np.testing.assert_allclose(fits[0]["mean"], x[folds != 0].mean(0))
    assert set(fits[0]["test_indices"]).isdisjoint(fits[0]["train_indices"])


def test_features_use_only_previous_estimates_and_fixed_metadata():
    questions, _, cases = inputs()
    old = np.zeros((96, 2, 4, 2))
    old[:, :, 0] = .6
    old[:, :, 2] = .4
    old[:, :, 3] = .1
    free = np.full((96, 2, 8), .5)
    sets, names = features(questions, cases, old, free)
    assert sets["difficulty"].shape == (192, len(names))
    np.testing.assert_allclose(sets["difficulty_plus_gain"][:, -1], .3)
    np.testing.assert_allclose(sets["difficulty_plus_R"][:, -1], .4)
    np.testing.assert_array_equal(sets["difficulty_plus_gain"][:, :-1], sets["difficulty"])


def test_exact_length_bootstrap_constant_and_decision_boundaries():
    conf = cfg()
    conf["bootstrap_replicates"] = 30
    values = np.full((12, 2), .25)
    mask = np.ones((12, 2), dtype=bool)
    mask[0, 0] = False
    assert matched_interval(values, mask, np.repeat(["a", "b"], 6), conf) == [.25, .25]
    summaries = [{"condition": c, "success": .6, "truncation": 0.} for c in conf["conditions"]]
    contrasts = {"CN-WN": {"ci": [.1, .3], "difference": .2, "exact_length_ci95": [.1, .3]}}
    prediction = {"neutral_history_gap": {"difficulty_plus_gain": {"ci": [.01, .02], "relative_improvement": .1, "bank_improvement": [.01, .01]}}}
    assert decision(conf, summaries, contrasts, prediction)["status"] == "both_signals_worth_followup"
    prediction["neutral_history_gap"]["difficulty_plus_gain"]["ci"][0] = -.01
    assert decision(conf, summaries, contrasts, prediction)["status"] == "history_only_no_adaptive_signal"
    contrasts["CN-WN"]["ci"][0] = -.1
    assert decision(conf, summaries, contrasts, prediction)["status"] == "neither_gate_supported"
    summaries[0]["truncation"] = .051
    assert decision(conf, summaries, contrasts, prediction)["status"] == "protocol_uninterpretable"
