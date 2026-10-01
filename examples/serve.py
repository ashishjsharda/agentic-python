"""Chapter 13: serve the agent over HTTP.

    uvicorn examples.serve:app --port 8000
    curl -X POST localhost:8000/run -H 'content-type: application/json' \\
         -d '{"user_id": "u1", "message": "What is 17 * 23?"}'
"""
from agentic.agent import Agent
from agentic.llm import make_client
from agentic.service import create_app
from agentic.tools import CALCULATOR_SCHEMA, calculator

# Set these to your model's current per-million-token prices.
PRICES = (3.00, 15.00)


def make_agent(tracker):
    return Agent(make_client(tracker=tracker),
                 {"calculator": calculator}, [CALCULATOR_SCHEMA])


app = create_app(make_agent, prices=PRICES, budget_per_user_usd=1.00)
