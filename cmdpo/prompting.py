"""Shared prompt protocol for preference training, probes, and generation."""
from __future__ import annotations

from typing import Any

PROTOCOL_VERSION = "math-chat-v1"


def math_instruction(prompt: str) -> str:
    if "####" in prompt:
        return prompt
    return (f"{prompt}\n\nSolve the problem. Show the calculation briefly and end "
            "with exactly one line: #### <answer>")


def format_prompt(tokenizer: Any, prompt: str, use_chat_template: bool) -> str:
    if use_chat_template:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("Chat protocol requested but tokenizer has no chat template")
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": math_instruction(prompt)}],
            add_generation_prompt=True, tokenize=False,
        )
    return prompt
