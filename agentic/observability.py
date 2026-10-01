"""Chapter 11: structured traces, replay, and per-tool metrics."""
import json
import sys
import time
from collections import defaultdict


class TraceLogger:
    """Pass an instance as Agent(trace=...) to log every step as JSON."""

    def __init__(self, task_id, sink):
        self.task_id = task_id
        self.sink = sink

    def __call__(self, event, data):
        record = {"task_id": self.task_id, "type": event,
                  "timestamp": time.time(), "data": data}
        self.sink.write(json.dumps(record, default=str) + "\n")
        self.sink.flush()


def read_trace(lines, task_id):
    events = (json.loads(line) for line in lines if line.strip())
    return [e for e in events if e["task_id"] == task_id]


def replay_messages(events, upto_step=None):
    messages = []
    for e in events:
        step = e["data"].get("step")
        if upto_step is not None and step is not None and step > upto_step:
            break
        if e["type"] == "task_start":
            messages.append({"role": "user",
                             "content": e["data"]["message"]})
        elif e["type"] == "model_response":
            messages.append({"role": "assistant",
                             "content": e["data"]["content"]})
        elif e["type"] == "tool_results":
            messages.append({"role": "user",
                             "content": e["data"]["results"]})
    return messages


def summarize_trace(events):
    tools = defaultdict(lambda: {"calls": 0, "errors": 0, "total_ms": 0})
    summary = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0,
               "model_ms": 0.0, "hit_max_steps": False}
    for e in events:
        data = e["data"]
        if e["type"] == "model_response":
            summary["model_calls"] += 1
            summary["model_ms"] += data["duration_ms"]
            summary["input_tokens"] += data["usage"].get("input_tokens", 0)
            summary["output_tokens"] += data["usage"].get(
                "output_tokens", 0)
        elif e["type"] == "tool_call":
            stats = tools[data["name"]]
            stats["calls"] += 1
            stats["errors"] += data["is_error"]
            stats["total_ms"] += data["duration_ms"]
        elif e["type"] == "max_steps":
            summary["hit_max_steps"] = True
    summary["tools"] = dict(tools)
    return summary


def print_trace(events, out=sys.stdout):
    for e in events:
        data = e["data"]
        if e["type"] == "task_start":
            print(f"TASK   {data['message']}", file=out)
        elif e["type"] == "tool_call":
            flag = " ERROR" if data["is_error"] else ""
            print(f"TOOL   {data['name']}({json.dumps(data['input'])})"
                  f" {data['duration_ms']}ms{flag}", file=out)
        elif e["type"] == "final_answer":
            print(f"ANSWER {data['text']}", file=out)
        elif e["type"] == "max_steps":
            print("STOPPED: max steps reached", file=out)


if __name__ == "__main__":
    # python -m agentic.observability trace.jsonl TASK_ID
    with open(sys.argv[1]) as f:
        print_trace(read_trace(f, sys.argv[2]))
