"""Chapter 12: a scripted stand-in for the model, plus an eval harness."""
from dataclasses import dataclass
from typing import Callable


def text(t):
    return {"content": [{"type": "text", "text": t}],
            "usage": {"input_tokens": 10, "output_tokens": 5}}


def tool_use(name, tool_input, id="call_1"):
    return {"content": [{"type": "tool_use", "id": id, "name": name,
                         "input": tool_input}],
            "usage": {"input_tokens": 10, "output_tokens": 5}}


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, messages, tools=None, system=None,
                 max_tokens=1024, tool_choice=None):
        self.calls.append({"messages": list(messages), "tools": tools,
                           "system": system, "tool_choice": tool_choice})
        return next(self.responses)


@dataclass
class EvalCase:
    name: str
    prompt: str
    check: Callable[[str], bool]


def run_evals(make_agent, cases, trials=3):
    pass_rates = {}
    for case in cases:
        passed = 0
        for _ in range(trials):
            try:
                passed += bool(case.check(make_agent().run(case.prompt)))
            except Exception:
                pass
        pass_rates[case.name] = passed / trials
    return pass_rates
