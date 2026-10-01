"""Chapters 6-10 and 16: the research assistant grows."""
from agentic.hitl import PausedRun
from agentic.memory import VectorStore
from agentic.pgmemory import PgVectorStore
from agentic.security import mark_untrusted
from agentic.testing import FakeClient, text, tool_use
from agentic.tools import Workspace
from research_assistant.assistant import ask, build_assistant, build_team
from research_assistant.documents import index_directory
from research_assistant.memory import make_memory_tool, remember
from research_assistant.models import ResearchAnswer
from research_assistant.planner import ask_with_plan
from tests.test_research_v1 import fake_wikipedia

URL = "https://en.wikipedia.org/wiki/Eiffel_Tower"


def answer(confidence="high", text_="Built 1887-1889."):
    return {"answer": text_, "confidence": confidence,
            "citations": [{"url": URL, "quote": "constructed"}]}


def test_pgvector_store_roundtrip(dsn, namespace):
    store = PgVectorStore(dsn, namespace=namespace)
    store.add("a", "refunds are allowed within 30 days", {"n": 1})
    store.add("b", "the office opens at nine")
    hit = store.search("refund window days", k=1)[0]
    assert (hit.id, hit.metadata) == ("a", {"n": 1})
    store.add("a", "refunds are allowed within 45 days", {"n": 2})
    assert "45" in store.search("refunds allowed", k=1)[0].text
    other = PgVectorStore(dsn, namespace=namespace + "-other")
    assert other.search("refund") == []
    other.add("a", "same id, different namespace")
    assert "45" in store.search("refunds allowed", k=1)[0].text
    store.delete_namespace()
    other.delete_namespace()


def test_memory_remembers_confident_answers_only(dsn, namespace):
    store = PgVectorStore(dsn, namespace=namespace)
    sure = ResearchAnswer.model_validate(answer("high"))
    unsure = ResearchAnswer.model_validate(answer("low"))
    assert remember(store, "When was the Eiffel Tower built?", sure)
    assert not remember(store, "Who will win in 2031?", unsure)
    recalled = make_memory_tool(store)(query="Eiffel Tower built")
    assert recalled[0]["sources"] == [URL]
    assert len(store.search("2031", k=5)) == 1
    store.delete_namespace()


def test_assistant_uses_memory_then_stores_the_new_answer():
    memory = VectorStore()
    fake = FakeClient([
        tool_use("recall_past_research", {"query": "Eiffel"}, id="m"),
        tool_use("submit_answer", answer(), id="a"),
    ])
    assistant = build_assistant(fake, wikipedia_tools=fake_wikipedia([]),
                                memory=memory)
    result = ask(assistant, "When was the Eiffel Tower built?", memory)
    assert result.confidence == "high"
    assert memory.search("Eiffel Tower", k=1)[0].metadata[
        "sources"] == [URL]
    assert "recall_past_research" in fake.calls[0]["system"]


def test_documents_are_indexed_and_searchable(tmp_path):
    (tmp_path / "policies").mkdir()
    (tmp_path / "policies" / "travel.md").write_text(
        "Travel policy: economy class for flights under six hours.")
    (tmp_path / "onboarding.md").write_text("Laptops ship on day one.")
    store = VectorStore()
    assert index_directory(store, tmp_path) == 2
    fake = FakeClient([
        tool_use("retrieve_docs", {"query": "flight class policy"}),
        tool_use("submit_answer", {
            "answer": "Economy under six hours.", "confidence": "high",
            "citations": [{"url": "policies/travel.md (chunk 0)",
                           "quote": "economy class"}]}),
    ])
    assistant = build_assistant(fake, wikipedia_tools=fake_wikipedia([]),
                                documents=store)
    assert assistant.run("What's our flight policy?").answer.startswith(
        "Economy")
    result = fake.calls[1]["messages"][-1]["content"][0]["content"]
    assert "policies/travel.md" in result


def test_report_saving_requires_approval(tmp_path):
    workspace = Workspace(tmp_path)
    fake = FakeClient([
        tool_use("save_report", {"filename": "eiffel.md",
                                 "markdown": "# Eiffel"}, id="r"),
        tool_use("submit_answer", answer(), id="a"),
    ])
    assistant = build_assistant(fake, wikipedia_tools=fake_wikipedia([]),
                                reports=workspace)
    paused = assistant.run("Research the Eiffel Tower and save a report")
    assert isinstance(paused, PausedRun)
    assert not (tmp_path / "eiffel.md").exists()
    final = assistant.resume(paused, approved=True)
    assert (tmp_path / "eiffel.md").read_text() == "# Eiffel"
    assert final.confidence == "high"


def test_report_filename_is_validated(tmp_path):
    fake = FakeClient([
        tool_use("save_report", {"filename": "../x.md",
                                 "markdown": "x"}, id="r"),
        tool_use("submit_answer", answer(), id="a"),
    ])
    assistant = build_assistant(fake, wikipedia_tools=fake_wikipedia([]),
                                reports=Workspace(tmp_path))
    paused = assistant.run("save it")
    result = assistant.resume(paused, approved=True)
    assert result.confidence == "high"
    rejection = fake.calls[1]["messages"][-1]["content"][0]
    assert rejection["is_error"] and "Invalid arguments" in \
        rejection["content"]


def test_wikipedia_text_is_marked_untrusted():
    _, read = fake_wikipedia([])
    marked = mark_untrusted(read)(title="Eiffel Tower")
    assert marked["text"].startswith(f'<untrusted source="{URL}"')


def test_planned_question_is_split_and_reviewed():
    planner = FakeClient([
        tool_use("record_plan", {"sub_questions": [
            "Burj Khalifa height", "Eiffel Tower height"],
            "needs_calculation": True}),
        tool_use("record_review", {"complete": False,
                                   "missing": "the difference"}),
    ])
    worker = FakeClient([
        tool_use("submit_answer", answer(text_="828 m and 330 m")),
        tool_use("submit_answer", answer(text_="498 m taller")),
    ])
    assistant = build_assistant(worker,
                                wikipedia_tools=fake_wikipedia([]))
    result = ask_with_plan(assistant, planner,
                           "How much taller is the Burj than the Eiffel?")
    assert result.answer == "498 m taller"
    first_prompt = worker.calls[0]["messages"][0]["content"]
    assert "1. Burj Khalifa height" in first_prompt
    assert "the difference" in worker.calls[1]["messages"][0]["content"]


class ScriptedBySystem:
    """One fake model, scripted separately per agent role."""

    def __init__(self, scripts):
        self.scripts = {k: iter(v) for k, v in scripts.items()}

    def complete(self, messages, tools=None, system=None, **kw):
        role = next(k for k in self.scripts if k in (system or ""))
        return next(self.scripts[role])


def test_team_delegates_in_parallel_and_merges_citations():
    docs = VectorStore()
    docs.add("pricing.md#0", "Pro plan costs 40 dollars a month",
             {"source": "pricing.md", "chunk": 0})
    both = {"content": [
        {"type": "tool_use", "id": "w", "name": "web_researcher",
         "input": {"task": "Eiffel Tower construction dates"}},
        {"type": "tool_use", "id": "d", "name": "documents_researcher",
         "input": {"task": "Pro plan price"}}]}
    client = ScriptedBySystem({
        "You lead a research team": [both, tool_use("submit_answer", {
            "answer": "1887-1889; Pro is $40.", "confidence": "high",
            "citations": [{"url": URL, "quote": "constructed"},
                          {"url": "pricing.md (chunk 0)",
                           "quote": "40 dollars"}]})],
        "cite the document source label": [tool_use("submit_answer", {
            "answer": "$40", "confidence": "high",
            "citations": [{"url": "pricing.md (chunk 0)",
                           "quote": "40 dollars"}]})],
        "careful research assistant": [
            tool_use("submit_answer", answer())],
    })
    team = build_team(client, wikipedia_tools=fake_wikipedia([]),
                      documents=docs)
    result = team.run("When was the Eiffel Tower built, and what does "
                      "Pro cost?")
    assert len(result.citations) == 2


def test_assistant_answering_in_text_is_nudged():
    fake = FakeClient([text("1889"), tool_use("submit_answer", answer())])
    assistant = build_assistant(fake, wikipedia_tools=fake_wikipedia([]))
    assert assistant.run("q").confidence == "high"
