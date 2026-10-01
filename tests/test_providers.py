"""Chapter 2: the same agent runs unchanged on two providers."""
import json

import httpx

from agentic.agent import Agent
from agentic.llm import (AnthropicClient, OpenAIClient, make_client,
                         to_openai_messages)
from agentic.tools import CALCULATOR_SCHEMA, calculator

TOOLS = {"calculator": calculator}


def scripted(responses, seen):
    replies = iter(responses)

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=next(replies))

    return httpx.MockTransport(handler)


def test_openai_request_translation():
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Let me compute."},
            {"type": "tool_use", "id": "t1", "name": "calculator",
             "input": {"expression": "2+2"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "4"},
            {"type": "text", "text": "Also explain."}]},
    ]
    out = to_openai_messages(messages, system="Be brief.")
    assert [m["role"] for m in out] == ["system", "user", "assistant",
                                        "tool", "user"]
    call = out[2]["tool_calls"][0]
    assert call["function"]["name"] == "calculator"
    assert json.loads(call["function"]["arguments"]) == {
        "expression": "2+2"}
    assert out[3] == {"role": "tool", "tool_call_id": "t1",
                      "content": "4"}


def test_openai_marks_tool_errors():
    out = to_openai_messages([{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "boom",
         "is_error": True}]}])
    assert out[0]["content"] == "ERROR: boom"


def test_openai_client_request_and_response_shapes():
    seen = []
    reply = {"choices": [{"finish_reason": "tool_calls", "message": {
        "content": None, "tool_calls": [{
            "id": "c1", "type": "function", "function": {
                "name": "calculator",
                "arguments": '{"expression": "6*7"}'}}]}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3}}
    client = OpenAIClient("some-model", api_key="k",
                          transport=scripted([reply], seen))
    response = client.complete(
        [{"role": "user", "content": "6*7"}], tools=[CALCULATOR_SCHEMA],
        tool_choice={"type": "tool", "name": "calculator"})
    body = seen[0]
    assert body["tools"][0]["function"]["parameters"] == \
        CALCULATOR_SCHEMA["input_schema"]
    assert body["tool_choice"] == {"type": "function",
                                   "function": {"name": "calculator"}}
    assert response["content"] == [{"type": "tool_use", "id": "c1",
                                    "name": "calculator",
                                    "input": {"expression": "6*7"}}]
    assert response["usage"] == {"input_tokens": 12, "output_tokens": 3}
    assert response["stop_reason"] == "tool_use"


def test_unparseable_openai_arguments_become_a_validation_error():
    seen = []
    bad = {"choices": [{"finish_reason": "tool_calls", "message": {
        "content": None, "tool_calls": [{
            "id": "c1", "type": "function", "function": {
                "name": "calculator", "arguments": "{oops"}}]}}]}
    done = {"choices": [{"finish_reason": "stop",
                         "message": {"content": "gave up"}}]}
    client = OpenAIClient("m", api_key="k",
                          transport=scripted([bad, done], seen))
    Agent(client, TOOLS, [CALCULATOR_SCHEMA]).run("x")
    tool_message = seen[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["content"].startswith("ERROR: ")
    assert "Invalid arguments" in tool_message["content"]


def test_same_agent_runs_on_both_providers():
    anthropic_replies = [
        {"content": [{"type": "tool_use", "id": "a1",
                      "name": "calculator",
                      "input": {"expression": "17*23"}}],
         "stop_reason": "tool_use"},
        {"content": [{"type": "text", "text": "391"}],
         "stop_reason": "end_turn"},
    ]
    openai_replies = [
        {"choices": [{"finish_reason": "tool_calls", "message": {
            "content": None, "tool_calls": [{
                "id": "o1", "type": "function", "function": {
                    "name": "calculator",
                    "arguments": '{"expression": "17*23"}'}}]}}]},
        {"choices": [{"finish_reason": "stop",
                      "message": {"content": "391"}}]},
    ]
    a_seen, o_seen = [], []
    clients = [
        AnthropicClient(api_key="k",
                        transport=scripted(anthropic_replies, a_seen)),
        OpenAIClient("m", api_key="k",
                     transport=scripted(openai_replies, o_seen)),
    ]
    for client in clients:
        agent = Agent(client, TOOLS, [CALCULATOR_SCHEMA])
        assert agent.run("What is 17 * 23?") == "391"
    assert a_seen[1]["messages"][-1]["content"][0]["content"] == "391"
    assert o_seen[1]["messages"][-1] == {
        "role": "tool", "tool_call_id": "o1", "content": "391"}


def test_make_client_reads_environment(monkeypatch):
    monkeypatch.setenv("AGENT_PROVIDER", "openai")
    monkeypatch.setenv("AGENT_MODEL", "some-model")
    client = make_client()
    assert isinstance(client, OpenAIClient)
    assert client.model == "some-model"
    monkeypatch.delenv("AGENT_PROVIDER")
    monkeypatch.delenv("AGENT_MODEL")
    assert isinstance(make_client(), AnthropicClient)
