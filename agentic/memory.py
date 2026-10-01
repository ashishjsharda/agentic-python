"""Chapter 5: long-term memory and context budgeting."""
import hashlib
import json
import re
from dataclasses import dataclass

import numpy as np


def hashing_embed(text, dim=512):
    """Offline stand-in for an embeddings API (bag of hashed words).

    Swap in a real embeddings provider for semantic similarity; the
    VectorStore below does not care which function produces vectors.
    """
    vector = np.zeros(dim)
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.md5(token.encode()).hexdigest()
        vector[int(digest, 16) % dim] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


@dataclass
class Hit:
    id: str
    text: str
    metadata: dict
    score: float


class VectorStore:
    def __init__(self, embed=hashing_embed):
        self.embed = embed
        self.ids, self.texts, self.metadata, self.vectors = [], [], [], []

    def add(self, id, text, metadata=None):
        self.ids.append(id)
        self.texts.append(text)
        self.metadata.append(metadata or {})
        self.vectors.append(np.asarray(self.embed(text), dtype=float))

    def search(self, query, k=5):
        if not self.vectors:
            return []
        matrix = np.vstack(self.vectors)
        q = np.asarray(self.embed(query), dtype=float)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
        scores = matrix @ q / np.maximum(norms, 1e-12)
        top = np.argsort(-scores)[:k]
        return [Hit(self.ids[i], self.texts[i], self.metadata[i],
                    float(scores[i])) for i in top]

    def save(self, path):
        rows = [{"id": i, "text": t, "metadata": m, "vector": v.tolist()}
                for i, t, m, v in zip(self.ids, self.texts, self.metadata,
                                      self.vectors, strict=True)]
        with open(path, "w") as f:
            json.dump(rows, f)

    @classmethod
    def load(cls, path, embed=hashing_embed):
        store = cls(embed)
        with open(path) as f:
            for row in json.load(f):
                store.ids.append(row["id"])
                store.texts.append(row["text"])
                store.metadata.append(row["metadata"])
                store.vectors.append(np.array(row["vector"]))
        return store


def estimate_tokens(messages):
    return len(json.dumps(messages, default=str)) // 4


def compact_history(messages, summarize, keep_last=6, max_tokens=50_000):
    if estimate_tokens(messages) <= max_tokens:
        return messages
    split = len(messages) - keep_last
    # The kept tail must open with an assistant turn so every tool
    # result still sits right after the tool call it answers.
    while split < len(messages) and messages[split]["role"] != "assistant":
        split += 1
    if split <= 0 or split >= len(messages):
        return messages
    summary = summarize(messages[:split])
    return [{"role": "user",
             "content": f"Summary of earlier conversation:\n{summary}"},
            *messages[split:]]


def make_summarizer(client):
    def summarize(old_messages):
        transcript = json.dumps(old_messages, default=str)[:100_000]
        response = client.complete([{"role": "user", "content": (
            "Summarize this agent transcript in under 200 words. Keep "
            "facts, numbers, decisions, and open questions.\n\n"
            + transcript)}])
        return "".join(b["text"] for b in response["content"]
                       if b["type"] == "text")

    return summarize
