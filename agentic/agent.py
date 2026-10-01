"""Chapter 5: the core agent loop."""
import json
import time

from pydantic import BaseModel, ValidationError

from .llm import extract_text
from .structured import schema_tool
from .tools import execute_tool_call

ANSWER_TOOL = "submit_answer"


class MaxStepsExceeded(Exception):
    pass


class Agent:
    def __init__(self, client, tools, tool_schemas, system=None,
                 max_steps=10, trace=None, answer_model=None):
        self.client = client
        self.tools = tools
        self.schemas = {s["name"]: s for s in tool_schemas}
        self.tool_schemas = list(tool_schemas)
        self.answer_model = answer_model
        if answer_model:
            self.tool_schemas.append(schema_tool(
                answer_model, ANSWER_TOOL,
                "Submit your final answer. Call this exactly once, "
                "when you are done.",
            ))
        self.system = system
        self.max_steps = max_steps
        self.trace = trace or (lambda event, data: None)

    def run(self, user_message):
        self.trace("task_start", {"message": user_message})
        messages = [{"role": "user", "content": user_message}]
        return self.continue_run(messages)

    def continue_run(self, messages):
        for step in range(self.max_steps):
            started = time.perf_counter()
            response = self.client.complete(
                messages, tools=self.tool_schemas, system=self.system
            )
            self.trace("model_response", {
                "step": step, "content": response["content"],
                "usage": response.get("usage", {}),
                "duration_ms": _ms_since(started),
            })
            messages.append(
                {"role": "assistant", "content": response["content"]}
            )

            tool_calls = [b for b in response["content"]
                          if b["type"] == "tool_use"]
            if not tool_calls:
                if not self.answer_model:
                    return self._finish(step, extract_text(response))
                messages.append({"role": "user", "content":
                                 f"Call {ANSWER_TOOL} to finish."})
                continue

            answer_calls = [tc for tc in tool_calls
                            if tc["name"] == ANSWER_TOOL]
            results = self.run_tools([tc for tc in tool_calls
                                      if tc["name"] != ANSWER_TOOL])
            for call in answer_calls:
                try:
                    answer = self.answer_model.model_validate(call["input"])
                    return self._finish(step, answer)
                except ValidationError as e:
                    results.append(_error_result(call, f"Invalid answer:"
                                                       f"\n{e}"))
            self.trace("tool_results", {"step": step,
                                        "results": results})
            messages.append({"role": "user", "content": results})

        self.trace("max_steps", {"max_steps": self.max_steps})
        raise MaxStepsExceeded(
            f"No final answer after {self.max_steps} steps"
        )

    def _finish(self, step, answer):
        text = answer if isinstance(answer, str) else \
            answer.model_dump_json()
        self.trace("final_answer", {"step": step, "text": text})
        return answer

    def run_tools(self, tool_calls):
        return [self._run_tool(tc) for tc in tool_calls]

    def _run_tool(self, tool_call):
        started = time.perf_counter()
        output = execute_tool_call(tool_call, self.tools, self.schemas)
        if isinstance(output, BaseModel):
            output = output.model_dump()
        is_error = isinstance(output, dict) and "error" in output
        self.trace("tool_call", {
            "name": tool_call["name"], "input": tool_call["input"],
            "is_error": is_error, "duration_ms": _ms_since(started),
        })
        content = output if isinstance(output, str) else json.dumps(
            output, default=str
        )
        return {"type": "tool_result", "tool_use_id": tool_call["id"],
                "content": content, "is_error": is_error}


def _error_result(tool_call, message):
    return {"type": "tool_result", "tool_use_id": tool_call["id"],
            "content": message, "is_error": True}


def _ms_since(started):
    return round((time.perf_counter() - started) * 1000, 1)
