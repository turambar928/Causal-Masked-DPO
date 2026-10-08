"""Offline API protocol checks. No credentials or network are used."""
from types import SimpleNamespace as NS

import pytest

from cmdpo.api_client import DEFAULT_QWEN_MODEL, chat_completion, require_qwen, stream_chat_completion


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __enter__(self):
        return iter(self.chunks)

    def __exit__(self, *args):
        self.closed = True


class FakeClient:
    def __init__(self, chunks):
        self.chat = NS(completions=self)
        self.stream = FakeStream(chunks)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.stream


def chunk(content=None, finish=None, model=DEFAULT_QWEN_MODEL):
    return NS(model=model, choices=[NS(delta=NS(content=content), finish_reason=finish)], usage=None)


def test_qwen_stream_is_fully_consumed_including_usage():
    client = FakeClient([chunk(), chunk("5"), chunk("6", "stop"),
                         NS(model=DEFAULT_QWEN_MODEL, choices=[], usage=NS(model_dump=lambda: {"total_tokens": 10}))])
    result = stream_chat_completion(client, DEFAULT_QWEN_MODEL, "test")
    assert result.text == "56"
    assert result.usage == {"total_tokens": 10}
    assert result.served_model == DEFAULT_QWEN_MODEL
    assert result.finish_reason == "stop"
    assert client.calls[0]["stream"] is True
    assert client.stream.closed


@pytest.mark.parametrize("model", ["gpt-5.6-sol", "claude-sonnet-4-6", "google/gemma-4-26B-A4B-it"])
def test_non_qwen_request_is_rejected_before_network(model):
    client = FakeClient([])
    with pytest.raises(ValueError, match="Qwen"):
        chat_completion(client, model, "test")
    assert not client.calls


def test_non_qwen_reported_model_is_not_accepted():
    client = FakeClient([chunk("text", "stop", model="gpt-5.6-sol")])
    with pytest.raises(ValueError, match="Qwen"):
        chat_completion(client, DEFAULT_QWEN_MODEL, "test")
    assert client.stream.closed


def test_premature_stream_end_is_not_accepted_as_a_complete_answer():
    client = FakeClient([chunk("partial")])
    with pytest.raises(ValueError, match="finish_reason"):
        chat_completion(client, DEFAULT_QWEN_MODEL, "test")
    assert len(client.calls) == 1


def test_provider_prefixed_qwen_model_is_allowed():
    require_qwen("Qwen/Qwen2.5-7B-Instruct")
