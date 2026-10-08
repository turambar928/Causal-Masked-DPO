from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import httpx

from openai import OpenAI


DEFAULT_QWEN_MODEL = "Qwen3.8-27B-no-thinking"


def require_qwen(model: str) -> None:
    if not model.rsplit("/", 1)[-1].lower().startswith("qwen"):
        raise ValueError("This experiment permits Qwen models only; no automatic model fallback")


@dataclass
class APIConfig:
    base_url: str
    api_key: str
    default_model: str
    models: list[str]


def load_api_config(path: str | Path = "api.txt", preferred_model: str | None = None) -> APIConfig:
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()]
    values = [line for line in lines if line]
    base_url = next((line for line in values if line.startswith("http://") or line.startswith("https://")), None)
    api_key = next((line for line in values if line.startswith("sk-")), None)
    if base_url is None:
        raise ValueError(f"No base_url found in {path}")
    if api_key is None:
        raise ValueError(f"No api key found in {path}")
    if not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    models = [
        line
        for line in values
        if line.rsplit("/", 1)[-1].lower().startswith("qwen")
        and "tokens" not in line.lower()
        and "request" not in line.lower()
    ]
    # Last verified Qwen route; api.txt also contains stale model names.
    default_model = preferred_model or DEFAULT_QWEN_MODEL
    require_qwen(default_model)
    return APIConfig(base_url=base_url, api_key=api_key, default_model=default_model, models=models)


def make_client(config: APIConfig, *, direct: bool | None = None) -> OpenAI:
    require_qwen(config.default_model)
    if direct is None:
        direct = os.environ.get("CMDPO_API_DIRECT", "0") == "1"
    return OpenAI(api_key=config.api_key, base_url=config.base_url,
                  http_client=httpx.Client(trust_env=not direct), timeout=90, max_retries=0)


@dataclass
class ChatResult:
    text: str
    usage: dict | None
    served_model: str | None
    finish_reason: str | None


def stream_chat_completion(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> ChatResult:
    require_qwen(model)
    parts = []
    usage = None
    served_model = None
    finish_reason = None
    # Consume the entire stream, including the final usage-only chunk. Errors
    # propagate; never silently retry with a non-streaming request or another model.
    with client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=temperature, max_tokens=max_tokens,
        stream=True, stream_options={"include_usage": True},
    ) as stream:
        for chunk in stream:
            reported_model = getattr(chunk, "model", None)
            if reported_model:
                require_qwen(reported_model)
                served_model = reported_model
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage.model_dump()
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            content = choice.delta.content
            if content:
                parts.append(content)
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason
    if finish_reason is None:
        raise ValueError("API stream ended without a completion finish_reason")
    return ChatResult("".join(parts), usage, served_model, finish_reason)


def chat_completion(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    return stream_chat_completion(client, model, prompt, temperature, max_tokens).text
