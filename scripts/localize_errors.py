#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmdpo.data import read_jsonl, write_jsonl
from cmdpo.arithmetic_check import locate_arithmetic_error
from cmdpo.localization import build_cm_weights, localize_from_success_rates
from cmdpo.segmentation import segment_steps
from cmdpo.verifier import verify_answer


def load_generation_model(model_name: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def rollout_success_rates(
    model: Any,
    tokenizer: Any,
    prompt: str,
    steps: list[str],
    answer: str | None,
    num_rollouts: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[float]:
    if answer is None:
        return [0.0] * len(steps)

    rates = []
    for idx in range(len(steps)):
        prefix = prompt.rstrip() + "\n" + "\n".join(steps[: idx + 1]).rstrip() + "\nContinue the solution."
        encoded = tokenizer(prefix, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **encoded,
            do_sample=True,
            num_return_sequences=num_rollouts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        prompt_len = encoded["input_ids"].shape[-1]
        completions = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
        successes = sum(verify_answer(completion, answer) for completion in completions)
        rates.append(successes / max(1, num_rollouts))
    return rates


def heuristic_success_rates(steps: list[str], answer: str | None) -> list[float]:
    """Cheap fallback when rollout completions are unavailable.

    This is intentionally conservative: if an intermediate prefix already exposes a
    wrong final answer, later prefixes are treated as low-success. Otherwise the
    script delays the first-error guess until the final step.
    """
    if not steps:
        return []
    if answer is None:
        return [0.0] * len(steps)

    rates: list[float] = []
    wrong_seen = False
    for idx in range(len(steps)):
        prefix = "\n".join(steps[: idx + 1])
        if verify_answer(prefix, answer):
            rates.append(1.0)
        elif wrong_seen:
            rates.append(0.0)
        elif idx == len(steps) - 1:
            rates.append(0.0)
        else:
            rates.append(0.8)
        if idx == len(steps) - 1 and not verify_answer(prefix, answer):
            wrong_seen = True
    return rates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--filter-short", action="store_true")
    parser.add_argument("--disable-arithmetic-check", action="store_true")
    parser.add_argument("--model", help="Optional model name/path for rollout-based localization.")
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    model = tokenizer = None
    if args.model:
        model, tokenizer = load_generation_model(args.model)

    output_rows = []
    skipped = 0
    for row in read_jsonl(args.input):
        steps = row.get("rejected_steps") or segment_steps(row["rejected"])
        if args.filter_short and len(steps) < 2:
            skipped += 1
            continue

        arithmetic_error = None
        if not args.disable_arithmetic_check:
            arithmetic_error = locate_arithmetic_error(row.get("prompt", ""), steps)

        if arithmetic_error is not None:
            first_error = int(arithmetic_error)
            weights = build_cm_weights(len(steps), first_error, args.gamma)
            row["rejected_steps"] = steps
            row["first_error_step"] = first_error
            row["prefix_success_rates"] = []
            row["step_weights"] = weights
            row["localization_confidence"] = 1.0
            output_rows.append(row)
            continue

        if "prefix_success_rates" in row:
            result = localize_from_success_rates(
                row["prefix_success_rates"],
                gamma=args.gamma,
                tau=args.tau,
            )
        elif "first_error_step" in row:
            first_error = int(row["first_error_step"])
            weights = build_cm_weights(len(steps), first_error, args.gamma)
            row["rejected_steps"] = steps
            row["step_weights"] = weights
            row.setdefault("localization_confidence", 1.0)
            output_rows.append(row)
            continue
        else:
            if model is not None and tokenizer is not None:
                rates = rollout_success_rates(
                    model,
                    tokenizer,
                    row["prompt"],
                    steps,
                    row.get("answer"),
                    args.num_rollouts,
                    args.max_new_tokens,
                    args.temperature,
                    args.top_p,
                )
            else:
                rates = heuristic_success_rates(steps, row.get("answer"))
            result = localize_from_success_rates(rates, gamma=args.gamma, tau=args.tau)

        if result.confidence < args.min_confidence:
            skipped += 1
            continue

        row["rejected_steps"] = steps
        row.update(result.to_dict())
        row["localization_confidence"] = row.pop("confidence")
        output_rows.append(row)

    write_jsonl(args.output, output_rows)
    print(f"Wrote {len(output_rows)} localized rows to {args.output}; skipped={skipped}")


if __name__ == "__main__":
    main()
