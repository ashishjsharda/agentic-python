"""Chapter 18: the authenticated, persistent research service."""
import json
import logging

import pytest
from fastapi.testclient import TestClient

from agentic.agent import MaxStepsExceeded
from research_assistant.db import Database
from research_assistant.logs import JsonFormatter
from research_assistant.models import ResearchAnswer
from research_assistant.service import create_app

ANSWER = ResearchAnswer.model_validate({
    "answer": "1889", "confidence": "high",
    "citations": [{"url": "https://en.wikipedia.org/wiki/Eiffel_Tower",
                   "quote": "completed in 1889"}]})


@pytest.fixture
def db(dsn):
    database = Database(dsn)
    with database.connect() as conn:
        conn.execute("TRUNCATE api_keys, research_runs")
    return database


def fake_research(question, tracker, trace):
    tracker.record({"input_tokens": 100_000, "output_tokens": 10_000})
    trace("tool_call", {"name": "read_wikipedia", "duration_ms": 5,
                        "is_error": False})
    if "loop" in question:
        raise MaxStepsExceeded("No final answer after 15 steps")
    return ANSWER


@pytest.fixture
def http(db):
    app = create_app(db, fake_research, prices=(1.0, 5.0),
                     daily_budget_usd=0.40)
    return TestClient(app)


def auth(key):
    return {"authorization": f"Bearer {key}"}


def test_requests_without_a_valid_key_are_rejected(http, db):
    assert http.post("/research", json={"question": "hi?"}) \
        .status_code == 401
    assert http.post("/research", json={"question": "hi?"},
                     headers=auth("ra_wrong")).status_code == 401


def test_keys_are_stored_hashed(db):
    key = db.create_api_key("alice")
    with db.connect() as conn:
        stored = conn.execute("SELECT key_hash FROM api_keys") \
            .fetchone()[0]
    assert key.startswith("ra_") and key not in stored
    assert db.user_for_key(key) == "alice"


def test_research_is_persisted_and_private(http, db):
    alice, bob = db.create_api_key("alice"), db.create_api_key("bob")
    created = http.post("/research", headers=auth(alice),
                        json={"question": "When was the tower built?"})
    assert created.status_code == 201
    run_id = created.json()["id"]
    assert created.json()["answer"]["answer"] == "1889"
    assert created.headers["x-request-id"]

    stored = http.get(f"/research/{run_id}", headers=auth(alice)).json()
    assert stored["status"] == "succeeded"
    assert float(stored["cost_usd"]) == pytest.approx(0.15)
    assert http.get(f"/research/{run_id}",
                    headers=auth(bob)).status_code == 404


def test_budget_step_limit_and_refusal(http, db):
    key = db.create_api_key("carol")
    ask = lambda q: http.post("/research", headers=auth(key),  # noqa
                              json={"question": q}).status_code
    assert ask("q one") == 201        # spends 0.15 of 0.40
    assert ask("loop forever") == 503  # spends 0.15, then gives up
    assert ask("q two") == 402        # 0.10 left: stopped mid-run
    assert ask("q three") == 429      # nothing left: refused up front
    with db.connect() as conn:
        statuses = [r[0] for r in conn.execute(
            "SELECT status FROM research_runs ORDER BY created_at")]
    assert statuses == ["succeeded", "failed", "failed"]
    assert db.spent_today("carol") == pytest.approx(0.45)


def test_logs_are_single_line_json(http, db, caplog):
    key = db.create_api_key("dana")
    with caplog.at_level(logging.INFO, logger="research_assistant"):
        http.post("/research", headers=auth(key),
                  json={"question": "When?"})
    lines = [json.loads(JsonFormatter().format(r))
             for r in caplog.records
             if r.name.startswith("research_assistant")]
    events = [line["event"] for line in lines]
    assert {"tool_call", "run_finished", "http_request"} <= set(events)
    finished = next(x for x in lines if x["event"] == "run_finished")
    assert finished["user"] == "dana" and finished["status"] == \
        "succeeded"
    assert all(line["request_id"] != "-" for line in lines)
