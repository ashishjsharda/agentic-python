"""Chapter 18: persistence for users, API keys, and research runs."""
import hashlib
import json
import secrets
import uuid

import psycopg

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    key_hash    text PRIMARY KEY,
    user_id     text NOT NULL,
    revoked     boolean NOT NULL DEFAULT false,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS research_runs (
    id          uuid PRIMARY KEY,
    user_id     text NOT NULL,
    question    text NOT NULL,
    status      text NOT NULL,
    answer      jsonb,
    error       text,
    cost_usd    numeric(12, 6) NOT NULL DEFAULT 0,
    duration_ms integer,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS research_runs_by_user
    ON research_runs (user_id, created_at DESC);
"""


def hash_key(key):
    return hashlib.sha256(key.encode()).hexdigest()


class Database:
    def __init__(self, dsn):
        self.dsn = dsn
        with self.connect() as conn:
            conn.execute(SCHEMA)

    def connect(self):
        return psycopg.connect(self.dsn, autocommit=True)

    def create_api_key(self, user_id):
        """Returns the key once. Only its hash is ever stored."""
        key = "ra_" + secrets.token_urlsafe(32)
        with self.connect() as conn:
            conn.execute("INSERT INTO api_keys (key_hash, user_id) "
                         "VALUES (%s, %s)", (hash_key(key), user_id))
        return key

    def user_for_key(self, key):
        with self.connect() as conn:
            row = conn.execute(
                "SELECT user_id FROM api_keys "
                "WHERE key_hash = %s AND NOT revoked",
                (hash_key(key),)).fetchone()
        return row[0] if row else None

    def spent_today(self, user_id):
        with self.connect() as conn:
            (total,) = conn.execute(
                "SELECT coalesce(sum(cost_usd), 0) FROM research_runs "
                "WHERE user_id = %s AND created_at > now() - "
                "interval '1 day'", (user_id,)).fetchone()
        return float(total)

    def start_run(self, user_id, question):
        run_id = uuid.uuid4()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO research_runs (id, user_id, question, status)"
                " VALUES (%s, %s, %s, 'running')",
                (run_id, user_id, question))
        return run_id

    def finish_run(self, run_id, status, cost_usd, duration_ms,
                   answer=None, error=None):
        with self.connect() as conn:
            conn.execute(
                "UPDATE research_runs SET status = %s, answer = %s, "
                "error = %s, cost_usd = %s, duration_ms = %s "
                "WHERE id = %s",
                (status, json.dumps(answer) if answer else None, error,
                 cost_usd, duration_ms, run_id))

    def get_run(self, run_id, user_id):
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id, question, status, answer, error, cost_usd, "
                "created_at FROM research_runs "
                "WHERE id = %s AND user_id = %s",
                (run_id, user_id)).fetchone()
        if not row:
            return None
        keys = ["id", "question", "status", "answer", "error",
                "cost_usd", "created_at"]
        return dict(zip(keys, row, strict=True))
