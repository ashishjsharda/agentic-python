"""Chapter 6: chunking, indexing, and retrieval as a tool."""
import re

RETRIEVE_DOCS_SCHEMA = {
    "name": "retrieve_docs",
    "description": (
        "Search the internal knowledge base. Returns passages with a "
        "source label. Cite the source for every claim you use."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "k": {"type": "integer", "minimum": 1, "maximum": 10,
                  "default": 5},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


def chunk_text(text, max_words=300, overlap=50):
    if not 0 <= overlap < max_words:
        raise ValueError("overlap must be >= 0 and < max_words")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], []
    for paragraph in paragraphs:
        words = paragraph.split()
        if current and len(current) + len(words) > max_words:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
        current.extend(words)
        while len(current) > max_words:
            chunks.append(" ".join(current[:max_words]))
            current = current[max_words - overlap:]
    if current:
        chunks.append(" ".join(current))
    return chunks


def index_document(store, source, text, **chunk_options):
    for i, chunk in enumerate(chunk_text(text, **chunk_options)):
        store.add(f"{source}#{i}", chunk, {"source": source, "chunk": i})


def keyword_score(query, text):
    terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(terms & words) / len(terms) if terms else 0.0


def hybrid_search(store, query, k=5, alpha=0.5, candidates=50):
    hits = store.search(query, k=candidates)
    for hit in hits:
        hit.score = (alpha * hit.score
                     + (1 - alpha) * keyword_score(query, hit.text))
    return sorted(hits, key=lambda h: h.score, reverse=True)[:k]


def make_retrieve_docs(store):
    def retrieve_docs(query, k=5):
        return [
            {"source": f"{h.metadata['source']} (chunk "
                       f"{h.metadata['chunk']})",
             "text": h.text, "score": round(h.score, 3)}
            for h in hybrid_search(store, query, k=k)
        ]

    return retrieve_docs
