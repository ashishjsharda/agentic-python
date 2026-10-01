"""Chapter 17: the LangGraph build behaves like the hand-written one."""
import pytest
from langgraph.errors import GraphRecursionError

from agentic.testing import FakeClient, tool_use
from research_assistant.assistant import build_assistant_v1
from research_assistant.langgraph_version import ask, build_graph
from tests.test_research_v1 import fake_wikipedia

URL = "https://en.wikipedia.org/wiki/Eiffel_Tower"
GOOD = {"answer": "1887 to 1889", "confidence": "high",
        "citations": [{"url": URL, "quote": "constructed from 1887"}]}


def script():
    return [
        tool_use("search_wikipedia", {"query": "Eiffel Tower"}, id="s"),
        tool_use("read_wikipedia", {"title": "Eiffel Tower"}, id="r"),
        tool_use("submit_answer", {**GOOD, "citations": []}, id="bad"),
        tool_use("submit_answer", GOOD, id="ok"),
    ]


def test_both_implementations_give_the_same_answer_and_transcript():
    raw_client, graph_client = FakeClient(script()), FakeClient(script())
    raw = build_assistant_v1(raw_client,
                             wikipedia_tools=fake_wikipedia([]))
    graph = build_graph(graph_client, wikipedia_tools=fake_wikipedia([]))
    assert raw.run("When?") == ask(graph, "When?")
    raw_calls = [[m["role"] for m in c["messages"]]
                 for c in raw_client.calls]
    graph_calls = [[m["role"] for m in c["messages"]]
                   for c in graph_client.calls]
    assert raw_calls == graph_calls


def test_graph_step_limit_is_the_recursion_limit():
    looping = FakeClient([tool_use("calculator", {"expression": "1"},
                                   id=f"c{i}") for i in range(50)])
    graph = build_graph(looping, wikipedia_tools=fake_wikipedia([]))
    with pytest.raises(GraphRecursionError):
        ask(graph, "loop", max_steps=3)
