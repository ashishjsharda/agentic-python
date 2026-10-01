"""Chapter 5: run the research assistant against a live model.

    export ANTHROPIC_API_KEY=...          # or:
    export AGENT_PROVIDER=openai OPENAI_API_KEY=... AGENT_MODEL=...
    python -m examples.assistant_v1 "How tall is Mount Kilimanjaro?"
"""
import sys

from agentic.llm import make_client
from research_assistant.assistant import build_assistant_v1


def show(event, data):
    if event == "tool_call":
        print(f"  -> {data['name']}({data['input']})")


def main():
    question = " ".join(sys.argv[1:]) or "When was the Eiffel Tower built?"
    answer = build_assistant_v1(make_client(), trace=show).run(question)
    print(f"\n{answer.answer}\nconfidence: {answer.confidence}")
    for c in answer.citations:
        print(f"  [{c.url}] \"{c.quote}\"")


if __name__ == "__main__":
    main()
