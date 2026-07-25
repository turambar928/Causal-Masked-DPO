#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmdpo.api_client import chat_completion, load_api_config, make_client
from cmdpo.data import read_jsonl, write_jsonl
from cmdpo.localization import localize_from_success_rates
from cmdpo.segmentation import segment_steps
from cmdpo.verifier import verify_answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-file", default="api.txt")
    parser.add_argument("--model")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--num-rollouts", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    config = load_api_config(args.api_file, preferred_model=args.model)
    client = make_client(config)
    model = args.model or config.default_model
    output_rows = []

    for row in tqdm(read_jsonl(args.input)):
        steps = row.get("rejected_steps") or segment_steps(row["rejected"])
        rates = []
        for idx in range(len(steps)):
            prefix = "\n".join(steps[: idx + 1])
            rollout_prompt = (
                f"{row['prompt']}\n\n"
                "A partial solution is given below. Continue it naturally to the final answer. "
                "End with '#### <answer>'.\n\n"
                f"Partial solution:\n{prefix}"
            )
            successes = 0
            for _ in range(args.num_rollouts):
                completion = chat_completion(
                    client,
                    model,
                    rollout_prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                successes += int(verify_answer(completion, row["answer"]))
            rates.append(successes / max(1, args.num_rollouts))
        result = localize_from_success_rates(rates, gamma=args.gamma, tau=args.tau)
        row["rejected_steps"] = steps
        row.update(result.to_dict())
        row["localization_confidence"] = row.pop("confidence")
        row["metadata"] = {**row.get("metadata", {}), "localizer_model": model}
        output_rows.append(row)

    write_jsonl(args.output, output_rows)
    print(f"Wrote {len(output_rows)} API-localized rows to {args.output}")


if __name__ == "__main__":
    main()
