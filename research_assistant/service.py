"""Chapter 18: the research assistant as an authenticated service."""
import asyncio
import logging
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from agentic.agent import MaxStepsExceeded
from agentic.llm import BudgetExceeded, CostTracker

from .logs import log, request_id, trace_to_log

logger = logging.getLogger("research_assistant")


class ResearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


def create_app(db, run_research, prices, daily_budget_usd=2.00):
    """run_research(question, tracker, trace) -> ResearchAnswer."""
    app = FastAPI(title="Research Assistant")

    @app.middleware("http")
    async def with_request_id(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request_id.set(rid)
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        log(logger, "http_request", method=request.method,
            path=request.url.path, status=response.status_code,
            ms=round((time.perf_counter() - started) * 1000, 1))
        return response

    def current_user(authorization: str = Header(default="")):
        scheme, _, key = authorization.partition(" ")
        user = (db.user_for_key(key)
                if scheme.lower() == "bearer" and key else None)
        if user is None:
            raise HTTPException(401, "Missing or invalid API key",
                                headers={"WWW-Authenticate": "Bearer"})
        return user

    @app.get("/healthz")
    def health():
        with db.connect() as conn:
            conn.execute("SELECT 1")
        return {"status": "ok"}

    @app.post("/research", status_code=201)
    async def research(body: ResearchRequest,
                       user: str = Depends(current_user)):
        remaining = daily_budget_usd - db.spent_today(user)
        if remaining <= 0:
            raise HTTPException(429, "Daily budget exhausted")
        run_id = db.start_run(user, body.question)
        tracker = CostTracker(*prices, budget_usd=remaining)
        started = time.perf_counter()
        status, answer, error, code = "failed", None, None, None
        try:
            result = await asyncio.to_thread(
                run_research, body.question, tracker,
                trace_to_log(logger, run_id))
            status, answer = "succeeded", result.model_dump()
        except BudgetExceeded as e:
            error, code = str(e), 402
        except MaxStepsExceeded as e:
            error, code = str(e), 503
        except Exception as e:
            logger.exception("run_crashed", extra={
                "request_id": request_id.get(),
                "fields": {"run_id": str(run_id)}})
            error, code = f"{type(e).__name__}", 500
        ms = round((time.perf_counter() - started) * 1000)
        db.finish_run(run_id, status, tracker.cost_usd, ms,
                      answer=answer, error=error)
        log(logger, "run_finished", run_id=str(run_id), user=user,
            status=status, cost_usd=round(tracker.cost_usd, 6), ms=ms)
        if code:
            raise HTTPException(code, f"Run {run_id} {status}: {error}")
        return {"id": str(run_id), "status": status, "answer": answer}

    @app.get("/research/{run_id}")
    def get_research(run_id: uuid.UUID,
                     user: str = Depends(current_user)):
        run = db.get_run(run_id, user)
        if run is None:
            raise HTTPException(404, "Not found")
        return run

    return app
