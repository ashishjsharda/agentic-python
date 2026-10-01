"""Chapter 9: approval gates that pause, persist, and resume a run."""
import json
from dataclasses import asdict, dataclass

from .agent import Agent


class ApprovalRequired(Exception):
    def __init__(self, pending, completed):
        super().__init__(f"{len(pending)} tool call(s) need approval")
        self.pending = pending
        self.completed = completed


@dataclass
class PausedRun:
    messages: list
    pending: list
    completed: list

    def to_json(self):
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data):
        return cls(**json.loads(data))


class ApprovalAgent(Agent):
    def __init__(self, *args, needs_approval=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.needs_approval = set(needs_approval)

    def run_tools(self, tool_calls):
        pending = [tc for tc in tool_calls
                   if tc["name"] in self.needs_approval]
        allowed = [tc for tc in tool_calls
                   if tc["name"] not in self.needs_approval]
        completed = [self._run_tool(tc) for tc in allowed]
        if pending:
            raise ApprovalRequired(pending, completed)
        return completed

    def continue_run(self, messages):
        try:
            return super().continue_run(messages)
        except ApprovalRequired as e:
            return PausedRun(messages, e.pending, e.completed)

    def resume(self, paused, approved, note=""):
        results = list(paused.completed)
        for tool_call in paused.pending:
            if approved:
                results.append(self._run_tool(tool_call))
            else:
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call["id"],
                    "content": f"Denied by a human reviewer. {note}",
                    "is_error": True,
                })
        paused.messages.append({"role": "user", "content": results})
        return self.continue_run(paused.messages)
