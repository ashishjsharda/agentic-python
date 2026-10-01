"""Chapter 17: version 1 of the assistant, rebuilt on LangGraph.

Same client, same tools, same answer contract. Only the loop changes
hands: LangGraph owns the control flow and the state.
"""
import json
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from agentic.agent import ANSWER_TOOL
from agentic.structured import schema_tool
from agentic.tools import Toolbox, calculator_tool, execute_tool_call

from .assistant import SYSTEM_V1
from .models import ResearchAnswer
from .sources import make_wikipedia_tools


class State(TypedDict):
    messages: Annotated[list, operator.add]
    answer: ResearchAnswer | None


def build_graph(client, wikipedia_tools=None):
    search, read = wikipedia_tools or make_wikipedia_tools()
    box = Toolbox(search, read, calculator_tool)
    schemas = [*box.schemas, schema_tool(
        ResearchAnswer, ANSWER_TOOL, "Submit your final answer.")]
    by_name = {s["name"]: s for s in box.schemas}

    def call_model(state):
        response = client.complete(state["messages"], tools=schemas,
                                   system=SYSTEM_V1)
        return {"messages": [{"role": "assistant",
                              "content": response["content"]}]}

    def run_tools(state):
        calls = [b for b in state["messages"][-1]["content"]
                 if b["type"] == "tool_use"]
        results, answer = [], None
        for call in calls:
            if call["name"] == ANSWER_TOOL:
                try:
                    answer = ResearchAnswer.model_validate(call["input"])
                    continue
                except ValidationError as e:
                    output = {"error": f"Invalid answer:\n{e}"}
            else:
                output = execute_tool_call(call, box.functions, by_name)
            is_error = isinstance(output, dict) and "error" in output
            results.append({"type": "tool_result",
                            "tool_use_id": call["id"],
                            "content": output if isinstance(output, str)
                            else json.dumps(output, default=str),
                            "is_error": is_error})
        if answer:
            return {"answer": answer, "messages": []}
        return {"messages": [{"role": "user", "content": results}]}

    def route_after_model(state):
        content = state["messages"][-1]["content"]
        return ("tools" if any(b["type"] == "tool_use" for b in content)
                else "nudge")

    def nudge(state):
        return {"messages": [{"role": "user", "content":
                              f"Call {ANSWER_TOOL} to finish."}]}

    graph = StateGraph(State)
    graph.add_node("model", call_model)
    graph.add_node("tools", run_tools)
    graph.add_node("nudge", nudge)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route_after_model,
                                ["tools", "nudge"])
    graph.add_conditional_edges(
        "tools", lambda s: END if s["answer"] else "model", [END, "model"])
    graph.add_edge("nudge", "model")
    return graph.compile()


def ask(graph, question, max_steps=12):
    state = graph.invoke(
        {"messages": [{"role": "user", "content": question}],
         "answer": None},
        {"recursion_limit": max_steps * 2 + 1},
    )
    return state["answer"]
