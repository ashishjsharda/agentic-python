"""Chapter 6: the assistant remembers past research."""
import hashlib
from typing import Annotated

from pydantic import Field

from agentic.tools import tool

SCHEMA_VERSION = 1


def make_memory_tool(store):
    @tool
    def recall_past_research(
        query: Annotated[str, Field(description="What to look for")],
        k: Annotated[int, Field(ge=1, le=5)] = 3,
    ):
        """Search answers from earlier research sessions. Check this
        first. Treat hits as leads: verify and cite the original
        sources, never this tool."""
        return [{"question": h.metadata.get("question"),
                 "answer": h.text,
                 "sources": h.metadata.get("sources", []),
                 "similarity": round(h.score, 3)}
                for h in store.search(query, k=k)]

    return recall_past_research


def remember(store, question, answer):
    """Store confident answers; skip the ones we weren't sure about."""
    if answer.confidence == "low":
        return False
    key = hashlib.sha256(question.strip().lower().encode()).hexdigest()
    store.add(key[:16], f"Q: {question}\nA: {answer.answer}", {
        "question": question,
        "sources": [c.url for c in answer.citations],
        "confidence": answer.confidence,
        "schema_version": SCHEMA_VERSION,
    })
    return True
