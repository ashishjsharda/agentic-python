import json

import httpx
import pytest

from agentic.llm import (AnthropicClient, BudgetExceeded, CostTracker,
                         extract_text, parse_sse_text)
from agentic.reliability import RateLimitError, post_with_retry


def ok_body(text="hi", usage=(100, 20)):
    return {"content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": usage[0],
                      "output_tokens": usage[1]}}


def test_model_client_sends_expected_request():
    seen = {}

    def handler(request):
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body("hello"))

    client = AnthropicClient(api_key="test-key",
                         transport=httpx.MockTransport(handler))
    data = client.complete([{"role": "user", "content": "hi"}],
                           system="be brief")
    assert extract_text(data) == "hello"
    assert seen["headers"]["x-api-key"] == "test-key"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"]["system"] == "be brief"
    assert "tools" not in seen["body"]


def test_model_client_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json=ok_body())

    client = AnthropicClient(api_key="k",
                         transport=httpx.MockTransport(handler))
    client.complete([{"role": "user", "content": "hi"}])
    assert attempts["n"] == 3


def test_retry_gives_up_after_max_retries():
    calls = []

    def always_limited():
        calls.append(1)
        raise RateLimitError("429")

    with pytest.raises(RateLimitError):
        post_with_retry(always_limited, max_retries=3,
                        sleep=lambda s: None)
    assert len(calls) == 3


def test_cost_tracker_enforces_budget():
    tracker = CostTracker(3.0, 15.0, budget_usd=0.01)
    tracker.record({"input_tokens": 1000, "output_tokens": 100})
    assert tracker.cost_usd == pytest.approx(0.0045)
    with pytest.raises(BudgetExceeded):
        tracker.record({"input_tokens": 1000, "output_tokens": 300})


def test_client_records_usage_into_tracker():
    tracker = CostTracker(1.0, 1.0, budget_usd=10)
    client = AnthropicClient(api_key="k", tracker=tracker,
                         transport=httpx.MockTransport(
                             lambda r: httpx.Response(200, json=ok_body())))
    client.complete([{"role": "user", "content": "hi"}])
    assert (tracker.input_tokens, tracker.output_tokens) == (100, 20)


def test_parse_sse_text_yields_only_text_deltas():
    lines = [
        "event: message_start",
        'data: {"type": "message_start"}',
        'data: {"type": "content_block_delta", "delta": '
        '{"type": "text_delta", "text": "Hel"}}',
        'data: {"type": "content_block_delta", "delta": '
        '{"type": "text_delta", "text": "lo"}}',
        'data: {"type": "message_stop"}',
    ]
    assert "".join(parse_sse_text(lines)) == "Hello"
