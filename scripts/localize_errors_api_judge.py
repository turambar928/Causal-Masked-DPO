#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmdpo.api_client import chat_completion, load_api_config, make_client
from cmdpo.data import read_jsonl, write_jsonl
from cmdpo.localization import build_cm_weights
from cmdpo.segmentation import segment_steps


_JSON_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


def _parse_json_object(text: str) -> dict[str, Any]:
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError("judge response did not contain a JSON object")
    return json.loads(match.group(0))


def _clamp_step(value: Any, num_steps: int) -> int:
    try:
        idx = int(value)
    except (TypeError, ValueError):
        idx = num_steps - 1
    return max(0, min(idx, num_steps - 1))


def build_judge_prompt(prompt: str, gold_answer: str, steps: list[str]) -> str:
    numbered_steps = "\n".join(f"{idx}. {step}" for idx, step in enumerate(steps))
    return f"""You are checking a rejected math solution against the gold answer.

Question and instruction:
{prompt}

Gold solution / answer:
{gold_answer}

Rejected solution steps, indexed from 0:
{numbered_steps}

Find the first step where the rejected solution becomes mathematically or logically wrong.
If all reasoning steps are correct but the solution is incomplete or missing the final answer, choose the final listed step.
If a step is merely verbose but still correct, do not mark it wrong.

Return only one JSON object with this exact schema:
{{"first_error_step": <integer index>, "confidence": <number from 0 to 1>, "rationale": "<short reason>"}}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-file", default="api.txt")
    parser.add_argument("--model")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--min-step-chars", type=int, default=8)
    parser.add_argument("--max-step-chars", type=int, default=500)
    args = parser.parse_args()

    config = load_api_config(args.api_file, preferred_model=args.model)
    client = make_client(config)
    model = args.model or config.default_model

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    output_rows = []
    failures = 0
    for row in tqdm(rows):
        steps = row.get("rejected_steps") or segment_steps(
            row["rejected"],
            min_chars=args.min_step_chars,
            max_chars=args.max_step_chars,
        )
        if not steps:
            failures += 1
            continue

        judge_prompt = build_judge_prompt(row["prompt"], row["answer"], steps)
        raw = chat_completion(
            client,
            model,
            judge_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        try:
            parsed = _parse_json_object(raw)
            first_error = _clamp_step(parsed.get("first_error_step"), len(steps))
            confidence = float(parsed.get("confidence", 0.0))
            confidence = max(0.0, min(confidence, 1.0))
            rationale = str(parsed.get("rationale", "")).strip()
        except Exception as exc:
            failures += 1
            first_error = len(steps) - 1
            confidence = 0.0
            rationale = f"parse_failed: {exc}"

        item = dict(row)
        item["rejected_steps"] = steps
        item["first_error_step"] = first_error
        item["step_weights"] = build_cm_weights(len(steps), first_error, args.gamma)
        item["localization_confidence"] = confidence
        item["judge_rationale"] = rationale
        item["metadata"] = {
            **item.get("metadata", {}),
            "localizer": "api_judge",
            "localizer_model": model,
            "gamma": args.gamma,
        }
        output_rows.append(item)

    write_jsonl(args.output, output_rows)
    print(f"Wrote {len(output_rows)} API-judge localized rows to {args.output}; failures={failures}")


if __name__ == "__main__":
    main()
