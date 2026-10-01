"""Chapter 6: long-term memory in Postgres with pgvector."""
import json

import psycopg

from .memory import Hit, hashing_embed

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS memories (
    namespace   text NOT NULL,
    id          text NOT NULL,
    text        text NOT NULL,
    metadata    jsonb NOT NULL DEFAULT '{{}}',
    embedding   vector({dim}) NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (namespace, id)
);
CREATE INDEX IF NOT EXISTS memories_embedding_hnsw
    ON memories USING hnsw (embedding vector_cosine_ops);
"""


def to_pgvector(vector):
    return "[" + ",".join(f"{float(x):.6f}" for x in vector) + "]"


class PgVectorStore:
    """The same add/search interface as VectorStore, in Postgres."""

    def __init__(self, dsn, namespace="default", embed=hashing_embed,
                 dim=512):
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.namespace = namespace
        self.embed = embed
        self.dim = dim
        self.conn.execute(SCHEMA.format(dim=dim))

    def _vector(self, text):
        vector = self.embed(text)
        if len(vector) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, "
                             f"got {len(vector)}")
        return to_pgvector(vector)

    def add(self, id, text, metadata=None):
        self.conn.execute(
            """INSERT INTO memories (id, namespace, text, metadata,
                                     embedding)
               VALUES (%s, %s, %s, %s, %s::vector)
               ON CONFLICT (namespace, id) DO UPDATE
               SET text = EXCLUDED.text, metadata = EXCLUDED.metadata,
                   embedding = EXCLUDED.embedding""",
            (id, self.namespace, text, json.dumps(metadata or {}),
             self._vector(text)),
        )

    def search(self, query, k=5):
        vector = self._vector(query)
        rows = self.conn.execute(
            """SELECT id, text, metadata,
                      1 - (embedding <=> %s::vector) AS score
               FROM memories
               WHERE namespace = %s
               ORDER BY embedding <=> %s::vector
               LIMIT %s""",
            (vector, self.namespace, vector, k),
        ).fetchall()
        return [Hit(id, text, metadata, float(score))
                for id, text, metadata, score in rows]

    def delete_namespace(self):
        self.conn.execute("DELETE FROM memories WHERE namespace = %s",
                          (self.namespace,))
