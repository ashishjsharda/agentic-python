import json

import pytest

from agentic.agent import Agent, MaxStepsExceeded
from agentic.testing import FakeClient, text, tool_use
from agentic.tools import (CALCULATOR_SCHEMA, READ_FILE_SCHEMA,
                           WEB_SEARCH_SCHEMA, WRITE_FILE_SCHEMA, Workspace,
                           calculator, execute_tool_call, make_web_search)

SCHEMAS = {s["name"]: s for s in [CALCULATOR_SCHEMA]}
TOOLS = {"calculator": calculator}


@pytest.mark.parametrize("expr,expected", [
    ("2 + 2", 4), ("0.15 * 348.20", pytest.approx(52.23)),
    ("-(3 ** 2) % 5", 1), ("(1 + 2) / 4", 0.75),
])
def test_calculator(expr, expected):
    assert calculator(expr) == expected


@pytest.mark.parametrize("expr", [
    "__import__('os').system('ls')", "open('x')", "2 ** 1000",
    "[1, 2]", "True + 1",
])
def test_calculator_rejects_unsafe_input(expr):
    with pytest.raises(ValueError):
        calculator(expr)


def test_dispatch_reports_unknown_tool():
    out = execute_tool_call({"name": "nope", "input": {}}, TOOLS, SCHEMAS)
    assert "Unknown tool" in out["error"]


def test_dispatch_rejects_hallucinated_argument():
    call = {"name": "calculator", "input": {"expr": "2+2"}}
    assert "Invalid arguments" in execute_tool_call(call, TOOLS,
                                                    SCHEMAS)["error"]


def test_dispatch_turns_exceptions_into_errors():
    call = {"name": "calculator", "input": {"expression": "1/0"}}
    out = execute_tool_call(call, TOOLS, SCHEMAS)
    assert out["error"].startswith("ZeroDivisionError")


def test_workspace_blocks_path_traversal(tmp_path):
    ws = Workspace(tmp_path / "ws")
    ws.write_file("notes/a.txt", "hello")
    assert ws.read_file("notes/a.txt") == "hello"
    with pytest.raises(PermissionError):
        ws.read_file("../../etc/passwd")


def test_web_search_truncates_snippets():
    backend = lambda q: [{"title": "t", "url": "u", "snippet": "x" * 999}]
    results = make_web_search(backend, max_snippet_chars=10)("q")
    assert results == [{"title": "t", "url": "u", "snippet": "x" * 10}]


def test_all_schemas_are_valid_json_schema():
    import jsonschema
    for schema in [CALCULATOR_SCHEMA, WEB_SEARCH_SCHEMA,
                   READ_FILE_SCHEMA, WRITE_FILE_SCHEMA]:
        jsonschema.Draft202012Validator.check_schema(
            schema["input_schema"])


def test_agent_stops_after_final_answer():
    fake = FakeClient([
        tool_use("calculator", {"expression": "2+2"}),
        text("The answer is 4."),
    ])
    agent = Agent(fake, TOOLS, [CALCULATOR_SCHEMA])
    assert agent.run("what's 2+2") == "The answer is 4."
    result_turn = fake.calls[1]["messages"][-1]
    assert result_turn["content"][0]["content"] == "4"
    assert result_turn["content"][0]["tool_use_id"] == "call_1"


def test_agent_feeds_errors_back_so_model_can_recover():
    fake = FakeClient([
        tool_use("calculator", {"expr": "2+2"}),
        tool_use("calculator", {"expression": "2+2"}, id="call_2"),
        text("4"),
    ])
    agent = Agent(fake, TOOLS, [CALCULATOR_SCHEMA])
    assert agent.run("2+2?") == "4"
    first_result = fake.calls[1]["messages"][-1]["content"][0]
    assert first_result["is_error"] is True


def test_agent_raises_after_max_steps():
    fake = FakeClient([tool_use("calculator", {"expression": "1"},
                                id=f"c{i}") for i in range(5)])
    agent = Agent(fake, TOOLS, [CALCULATOR_SCHEMA], max_steps=3)
    with pytest.raises(MaxStepsExceeded):
        agent.run("loop forever")
    assert len(fake.calls) == 3


def test_messages_alternate_roles_and_pair_tool_results():
    fake = FakeClient([tool_use("calculator", {"expression": "1+1"}),
                       text("2")])
    Agent(fake, TOOLS, [CALCULATOR_SCHEMA]).run("1+1")
    messages = fake.calls[-1]["messages"]
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "user"]
    use_ids = [b["id"] for b in messages[1]["content"]]
    result_ids = [b["tool_use_id"] for b in messages[2]["content"]]
    assert use_ids == result_ids
    json.dumps(messages)  # the whole transcript is serializable
