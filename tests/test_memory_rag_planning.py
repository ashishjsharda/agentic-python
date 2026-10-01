import pytest

from agentic.agent import Agent
from agentic.memory import VectorStore, compact_history, estimate_tokens
from agentic.planning import (critique, make_plan, parse_json_object,
                              plan_and_execute)
from agentic.rag import (RETRIEVE_DOCS_SCHEMA, chunk_text, hybrid_search,
                         index_document, make_retrieve_docs)
from agentic.testing import FakeClient, text, tool_use


def test_vector_store_ranks_relevant_text_first(tmp_path):
    store = VectorStore()
    store.add("a", "refund policy: refunds within 30 days of purchase")
    store.add("b", "office hours are nine to five on weekdays")
    assert store.search("how many days for a refund", k=1)[0].id == "a"
    store.save(tmp_path / "s.json")
    loaded = VectorStore.load(tmp_path / "s.json")
    assert loaded.search("office hours", k=1)[0].id == "b"


def test_empty_store_returns_nothing():
    assert VectorStore().search("anything") == []


def test_compact_history_keeps_tool_pairs_intact():
    msgs = [{"role": "user", "content": "start " + "x" * 4000}]
    for i in range(6):
        msgs.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": f"t{i}", "name": "calc",
             "input": {}}]})
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}",
             "content": "y" * 500}]})
    out = compact_history(msgs, summarize=lambda old: "SUMMARY",
                          keep_last=3, max_tokens=100)
    assert out[0]["role"] == "user" and "SUMMARY" in out[0]["content"]
    assert out[1]["role"] == "assistant"
    assert estimate_tokens(out) < estimate_tokens(msgs)


def test_compact_history_is_noop_under_budget():
    msgs = [{"role": "user", "content": "hi"}]
    assert compact_history(msgs, lambda m: "S") is msgs


def test_chunking_respects_size_and_overlap():
    text = "\n\n".join(" ".join(f"p{p}w{w}" for w in range(120))
                       for p in range(5))
    chunks = chunk_text(text, max_words=200, overlap=20)
    assert all(len(c.split()) <= 200 for c in chunks)
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]
    joined = " ".join(chunks)
    assert all(f"p{p}w119" in joined for p in range(5))


def test_chunking_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("a b c", max_words=10, overlap=10)


def test_retrieval_tool_returns_cited_passages():
    store = VectorStore()
    index_document(store, "handbook.md",
                   "Vacation: employees get 20 days of paid leave.\n\n"
                   "Expenses: submit receipts within 60 days.")
    index_document(store, "security.md",
                   "Rotate API keys every 90 days.", max_words=50,
                   overlap=5)
    hits = make_retrieve_docs(store)("how many vacation days", k=1)
    assert hits[0]["source"].startswith("handbook.md")
    assert hybrid_search(store, "API keys", k=1)[0].id == "security.md#0"


def test_retrieval_as_a_tool_in_the_loop():
    store = VectorStore()
    index_document(store, "faq.md", "The warranty lasts two years.")
    fake = FakeClient([tool_use("retrieve_docs",
                                {"query": "warranty length"}),
                       text("Two years [faq.md (chunk 0)].")])
    agent = Agent(fake, {"retrieve_docs": make_retrieve_docs(store)},
                  [RETRIEVE_DOCS_SCHEMA])
    assert "Two years" in agent.run("How long is the warranty?")
    assert "faq.md" in fake.calls[1]["messages"][-1]["content"][0][
        "content"]


def test_parse_json_object_tolerates_code_fences():
    raw = 'Here you go:\n```json\n{"steps": ["a", "b"]}\n```'
    assert parse_json_object(raw) == {"steps": ["a", "b"]}


def test_make_plan_and_critique():
    fake = FakeClient([text('{"steps": ["search", "compute"]}'),
                       text('{"complete": false, "missing": "units"}')])
    assert make_plan("price task", fake) == ["search", "compute"]
    assert critique("t", "a", fake)["missing"] == "units"


def test_plan_and_execute_replans_when_critique_fails():
    planner = FakeClient([
        text('{"steps": ["step one"]}'),
        text('{"complete": false, "missing": "a citation"}'),
        text('{"steps": ["step one with citation"]}'),
        text('{"complete": true, "missing": ""}'),
    ])
    worker = FakeClient([text("draft"), text("draft [source]")])
    agent = Agent(worker, {}, [])
    answer = plan_and_execute("task", planner, agent, max_replans=1)
    assert answer.endswith("draft [source]")
    assert "a citation" in planner.calls[2]["messages"][0]["content"]
