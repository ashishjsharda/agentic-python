"""Chapter 4: run the agent. Uses the real API if ANTHROPIC_API_KEY is
set, otherwise a scripted fake so you can see the loop offline."""
import os

from agentic.agent import Agent
from agentic.llm import make_client
from agentic.testing import FakeClient, text, tool_use
from agentic.tools import CALCULATOR_SCHEMA, calculator

if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"):
    client = make_client()
else:
    client = FakeClient([
        tool_use("calculator", {"expression": "0.15 * 348.20"}),
        text("15% of $348.20 is $52.23, which is more than $50."),
    ])

agent = Agent(client, {"calculator": calculator}, [CALCULATOR_SCHEMA],
              trace=lambda event, data: print(f"[{event}]"))
print(agent.run("What is 15% of $348.20, and is that more than $50?"))
