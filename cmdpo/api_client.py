from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI


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

    model_prefixes = ("gpt-", "claude", "google/", "gemini", "Qwen", "qwen", "codex")
    models = [
        line
        for line in values
        if line.startswith(model_prefixes)
        and "tokens" not in line.lower()
        and "request" not in line.lower()
    ]
    default_model = preferred_model or next((m for m in models if m.startswith("gpt-")), None) or (models[0] if models else "")
    if not default_model:
        raise ValueError(f"No model name found in {path}; pass --model explicitly")
    return APIConfig(base_url=base_url, api_key=api_key, default_model=default_model, models=models)


def make_client(config: APIConfig) -> OpenAI:
    return OpenAI(api_key=config.api_key, base_url=config.base_url)


def chat_completion(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if isinstance(response, str):
        return response
    content = response.choices[0].message.content
    return content or ""
