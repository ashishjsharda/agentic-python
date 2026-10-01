"""Chapter 13: the agent as an HTTP service with per-user budgets."""
import asyncio
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agent import MaxStepsExceeded
from .llm import BudgetExceeded, CostTracker


class TaskRequest(BaseModel):
    user_id: str
    message: str


def create_app(agent_factory, prices, budget_per_user_usd=1.00):
    """agent_factory(tracker) -> Agent whose client records into tracker.

    prices = (input_usd_per_million_tokens, output_usd_per_million).
    Spend lives in memory here; use Redis or a database in production.
    """
    app = FastAPI()
    spent = defaultdict(float)

    @app.post("/run")
    async def run_agent(request: TaskRequest):
        remaining = budget_per_user_usd - spent[request.user_id]
        if remaining <= 0:
            raise HTTPException(429, "Budget exhausted for this user")
        tracker = CostTracker(*prices, budget_usd=remaining)
        agent = agent_factory(tracker)
        try:
            result = await asyncio.to_thread(agent.run, request.message)
        except BudgetExceeded as e:
            raise HTTPException(402, str(e)) from e
        except MaxStepsExceeded as e:
            raise HTTPException(503, str(e)) from e
        finally:
            spent[request.user_id] += tracker.cost_usd
        return {"result": result, "cost_usd": round(tracker.cost_usd, 6)}

    return app
