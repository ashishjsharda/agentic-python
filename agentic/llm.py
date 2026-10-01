"""Chapter 2: talking to model providers directly over HTTP.

Everything above this module speaks one canonical format: messages whose
content is a string or a list of blocks ("text", "tool_use",
"tool_result"), and responses shaped like {"content": [...],
"usage": {...}, "stop_reason": ...}. Each provider client translates.
"""
import json
import os
from typing import Protocol

import httpx

from .reliability import RateLimitError, ServerError, post_with_retry

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"


def auth_headers(api_key=None):
    return {
        "x-api-key": api_key or os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }


def call_model(messages, model=DEFAULT_ANTHROPIC_MODEL, max_tokens=1024):
    response = httpx.post(
        ANTHROPIC_URL,
        headers=auth_headers(),
        json={"model": model, "max_tokens": max_tokens,
              "messages": messages},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def extract_text(response):
    return "".join(
        block["text"] for block in response["content"]
        if block["type"] == "text"
    )


def parse_sse_text(lines):
    for line in lines:
        if not line.startswith("data: "):
            continue
        event = json.loads(line[len("data: "):])
        if event.get("type") != "content_block_delta":
            continue
        if event["delta"].get("type") == "text_delta":
            yield event["delta"]["text"]


def stream_text(messages, model=DEFAULT_ANTHROPIC_MODEL, max_tokens=1024):
    payload = {"model": model, "max_tokens": max_tokens,
               "messages": messages, "stream": True}
    with httpx.stream("POST", ANTHROPIC_URL, headers=auth_headers(),
                      json=payload, timeout=60) as response:
        response.raise_for_status()
        yield from parse_sse_text(response.iter_lines())


class BudgetExceeded(Exception):
    pass


class CostTracker:
    def __init__(self, input_usd_per_mtok, output_usd_per_mtok,
                 budget_usd):
        self.input_price = input_usd_per_mtok / 1_000_000
        self.output_price = output_usd_per_mtok / 1_000_000
        self.budget_usd = budget_usd
        self.input_tokens = 0
        self.output_tokens = 0

    @property
    def cost_usd(self):
        return (self.input_tokens * self.input_price
                + self.output_tokens * self.output_price)

    def record(self, usage):
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
        if self.cost_usd > self.budget_usd:
            raise BudgetExceeded(
                f"Spent ${self.cost_usd:.4f} of ${self.budget_usd:.2f}"
            )


class ModelClient(Protocol):
    """What the agent loop needs from any provider."""

    def complete(self, messages, tools=None, system=None,
                 max_tokens=1024, tool_choice=None): ...


class HTTPModelClient:
    """Shared plumbing: HTTP, retries, error mapping, cost tracking."""

    url = ""

    def __init__(self, model, api_key=None, tracker=None,
                 max_retries=5, transport=None):
        self.model = model
        self.api_key = api_key
        self.tracker = tracker
        self.max_retries = max_retries
        self._http = httpx.Client(timeout=60, transport=transport)

    def complete(self, messages, tools=None, system=None,
                 max_tokens=1024, tool_choice=None):
        body = self.build_request(messages, tools, system, max_tokens,
                                  tool_choice)
        raw = post_with_retry(lambda: self._post(body),
                              max_retries=self.max_retries)
        response = self.parse_response(raw)
        if self.tracker:
            self.tracker.record(response["usage"])
        return response

    def _post(self, body):
        response = self._http.post(self.url, headers=self.headers(),
                                   json=body)
        if response.status_code == 429:
            raise RateLimitError(response.text)
        if response.status_code >= 500:
            raise ServerError(f"{response.status_code}: {response.text}")
        response.raise_for_status()
        return response.json()


class AnthropicClient(HTTPModelClient):
    url = ANTHROPIC_URL

    def __init__(self, model=DEFAULT_ANTHROPIC_MODEL, **kwargs):
        super().__init__(model, **kwargs)

    def headers(self):
        return auth_headers(self.api_key)

    def build_request(self, messages, tools, system, max_tokens,
                      tool_choice):
        body = {"model": self.model, "max_tokens": max_tokens,
                "messages": messages}
        if tools:
            body["tools"] = tools
        if system:
            body["system"] = system
        if tool_choice:
            body["tool_choice"] = tool_choice
        return body

    def parse_response(self, raw):
        return {"content": raw["content"],
                "usage": raw.get("usage", {}),
                "stop_reason": raw.get("stop_reason")}


class OpenAIClient(HTTPModelClient):
    url = OPENAI_URL

    def headers(self):
        key = self.api_key or os.environ["OPENAI_API_KEY"]
        return {"authorization": f"Bearer {key}",
                "content-type": "application/json"}

    def build_request(self, messages, tools, system, max_tokens,
                      tool_choice):
        body = {"model": self.model, "max_completion_tokens": max_tokens,
                "messages": to_openai_messages(messages, system)}
        if tools:
            body["tools"] = [
                {"type": "function", "function": {
                    "name": t["name"], "description": t["description"],
                    "parameters": t["input_schema"]}}
                for t in tools
            ]
        if tool_choice:
            body["tool_choice"] = to_openai_tool_choice(tool_choice)
        return body

    def parse_response(self, raw):
        choice = raw["choices"][0]
        message = choice["message"]
        content = []
        if message.get("content"):
            content.append({"type": "text", "text": message["content"]})
        for call in message.get("tool_calls") or []:
            content.append({"type": "tool_use", "id": call["id"],
                            "name": call["function"]["name"],
                            "input": _parse_arguments(call)})
        usage = raw.get("usage", {})
        return {"content": content,
                "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                          "output_tokens":
                              usage.get("completion_tokens", 0)},
                "stop_reason": _STOP_REASONS.get(choice["finish_reason"],
                                                 choice["finish_reason"])}


_STOP_REASONS = {"stop": "end_turn", "tool_calls": "tool_use",
                 "length": "max_tokens"}


def _parse_arguments(call):
    try:
        return json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        # Surfaces as a schema validation error the model can fix.
        return {"_unparseable_arguments": call["function"]["arguments"]}


def to_openai_messages(messages, system=None):
    out = [{"role": "system", "content": system}] if system else []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            out.append({"role": message["role"], "content": content})
            continue
        texts = [b["text"] for b in content if b["type"] == "text"]
        if message["role"] == "assistant":
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"],
                                   "arguments": json.dumps(b["input"])}}
                     for b in content if b["type"] == "tool_use"]
            turn = {"role": "assistant",
                    "content": "\n".join(texts) or None}
            if calls:
                turn["tool_calls"] = calls
            out.append(turn)
            continue
        for block in content:
            if block["type"] == "tool_result":
                text = block["content"]
                if block.get("is_error"):
                    text = f"ERROR: {text}"
                out.append({"role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": text})
        if texts:
            out.append({"role": "user", "content": "\n".join(texts)})
    return out


def to_openai_tool_choice(choice):
    if choice["type"] == "tool":
        return {"type": "function", "function": {"name": choice["name"]}}
    return {"any": "required", "auto": "auto",
            "none": "none"}[choice["type"]]


def make_client(provider=None, **kwargs):
    """AGENT_PROVIDER=anthropic|openai and AGENT_MODEL pick the model."""
    provider = provider or os.environ.get("AGENT_PROVIDER", "anthropic")
    model = os.environ.get("AGENT_MODEL")
    if provider == "anthropic":
        return AnthropicClient(model or DEFAULT_ANTHROPIC_MODEL, **kwargs)
    if provider == "openai":
        if not model:
            raise ValueError("Set AGENT_MODEL to an OpenAI model name")
        return OpenAIClient(model, **kwargs)
    raise ValueError(f"Unknown provider: {provider}")
