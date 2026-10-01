"""Chapter 18: production wiring. Configuration comes from the
environment; see docker-compose.yml.

    uvicorn research_assistant.app:app
"""
import os

from agentic.llm import make_client
from agentic.pgmemory import PgVectorStore

from .assistant import ask, build_assistant
from .db import Database
from .logs import configure_logging
from .service import create_app

configure_logging()
DSN = os.environ["DATABASE_URL"]
PRICES = (float(os.environ["AGENT_PRICE_IN_PER_MTOK"]),
          float(os.environ["AGENT_PRICE_OUT_PER_MTOK"]))

memory = PgVectorStore(DSN, namespace="research-memory")
documents = PgVectorStore(DSN, namespace="documents")


def run_research(question, tracker, trace):
    assistant = build_assistant(make_client(tracker=tracker),
                                memory=memory, documents=documents,
                                trace=trace)
    return ask(assistant, question, memory)


app = create_app(
    Database(DSN), run_research, PRICES,
    daily_budget_usd=float(os.environ.get("DAILY_BUDGET_USD", "2.00")),
)
