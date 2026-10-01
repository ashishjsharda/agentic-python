"""Chapter 5: the research assistant, version 1."""
import httpx

from agentic.testing import FakeClient, tool_use
from research_assistant.assistant import build_assistant_v1
from research_assistant.models import ResearchAnswer
from research_assistant.sources import (USER_AGENT,
                                        make_wikipedia_tools)

PAGE_TEXT = ("The Eiffel Tower is a wrought-iron lattice tower in Paris. "
             "It was constructed from 1887 to 1889.")


def fake_wikipedia(seen):
    def handler(request):
        seen.append(request)
        params = request.url.params
        if params.get("list") == "search":
            return httpx.Response(200, json={"query": {"search": [
                {"title": "Eiffel Tower",
                 "snippet": 'The <span class="m">Eiffel</span> '
                            "Tower &amp; Paris"}]}})
        if params.get("titles") == "Nowhere":
            return httpx.Response(200, json={"query": {"pages": {
                "-1": {"title": "Nowhere", "missing": ""}}}})
        return httpx.Response(200, json={"query": {"pages": {"1": {
            "title": "Eiffel Tower", "extract": PAGE_TEXT}}}})

    http = httpx.Client(transport=httpx.MockTransport(handler),
                        headers={"User-Agent": USER_AGENT})
    return make_wikipedia_tools(http)


def test_wikipedia_tools_parse_and_clean_results():
    seen = []
    search, read = fake_wikipedia(seen)
    hits = search(query="eiffel tower")
    assert hits[0]["snippet"] == "The Eiffel Tower & Paris"
    assert hits[0]["url"] == "https://en.wikipedia.org/wiki/Eiffel_Tower"
    assert seen[0].url.params["srsearch"] == "eiffel tower"
    assert seen[0].headers["user-agent"] == USER_AGENT
    assert read(title="Eiffel Tower")["text"] == PAGE_TEXT
    try:
        read(title="Nowhere")
    except LookupError as e:
        assert "Nowhere" in str(e)


def test_assistant_v1_returns_a_validated_cited_answer():
    url = "https://en.wikipedia.org/wiki/Eiffel_Tower"
    good = {"answer": "It was built from 1887 to 1889.",
            "citations": [{"url": url, "quote": "It was constructed "
                                                "from 1887 to 1889."}],
            "confidence": "high"}
    fake = FakeClient([
        tool_use("search_wikipedia", {"query": "Eiffel Tower"}, id="s"),
        tool_use("read_wikipedia", {"title": "Eiffel Tower"}, id="r"),
        tool_use("submit_answer", {**good, "citations": []}, id="bad"),
        tool_use("submit_answer", good, id="ok"),
    ])
    agent = build_assistant_v1(fake, wikipedia_tools=fake_wikipedia([]))
    answer = agent.run("When was the Eiffel Tower built?")
    assert isinstance(answer, ResearchAnswer)
    assert answer.citations[0].url == url
    rejection = fake.calls[3]["messages"][-1]["content"][0]
    assert rejection["is_error"] and "citations" in rejection["content"]
    tool_names = [t["name"] for t in fake.calls[0]["tools"]]
    assert tool_names == ["search_wikipedia", "read_wikipedia",
                          "calculator", "submit_answer"]


def test_assistant_nudges_when_model_answers_in_plain_text():
    fake = FakeClient([
        {"content": [{"type": "text", "text": "1889"}]},
        tool_use("submit_answer", {
            "answer": "1889", "confidence": "low",
            "citations": [{"url": "u", "quote": "q"}]}),
    ])
    agent = build_assistant_v1(fake, wikipedia_tools=fake_wikipedia([]))
    assert agent.run("q").answer == "1889"
    assert "submit_answer" in fake.calls[1]["messages"][-1]["content"]
