"""Chapter 14: graders, suites, and the regression gate."""
from agentic.evals import (Case, Recorder, compare, grade, load_cases,
                           run_suite, summarize)
from agentic.testing import FakeClient, tool_use
from research_assistant.assistant import build_assistant_v1
from research_assistant.models import ResearchAnswer
from tests.test_research_v1 import fake_wikipedia

URL = "https://en.wikipedia.org/wiki/Eiffel_Tower"


def make_answer(url=URL, text="Completed in 1889.", confidence="high"):
    return ResearchAnswer.model_validate({
        "answer": text, "confidence": confidence,
        "citations": [{"url": url, "quote": "x"}]})


def test_dataset_is_well_formed():
    cases = load_cases("evals/research_cases.jsonl")
    assert len(cases) == 20
    assert len({c.id for c in cases}) == 20
    assert {"unanswerable", "false-premise", "calculation"} <= {
        c.category for c in cases}


def test_grader_checks_facts_grounding_tools_and_confidence():
    recorder = Recorder()
    recorder("tool_results", {"results": [
        {"content": f'{{"url": "{URL}", "text": "..."}}'}]})
    case = Case(id="c", question="q", category="x",
                must_include=["1,889"], requires_tool="calculator",
                expect_confidence=["low"])
    failures = grade(case, make_answer(
        url="https://made.up/page"), recorder)
    assert "cites unread source https://made.up/page" in failures
    assert "never called calculator" in failures
    assert any("confidence" in f for f in failures)
    assert not any("lacks" in f for f in failures)  # 1,889 == 1889


def test_suite_runs_real_agent_and_grounds_citations():
    def make_agent(recorder):
        fake = FakeClient([
            tool_use("read_wikipedia", {"title": "Eiffel Tower"}, id="r"),
            tool_use("submit_answer", {
                "answer": "It was completed in 1889.",
                "confidence": "high",
                "citations": [{"url": URL, "quote": "1889"}]}, id="a"),
        ])
        return build_assistant_v1(fake, fake_wikipedia([]),
                                  trace=recorder)

    case = Case(id="eiffel", question="When?", category="fact",
                must_include=["1889"], critical=True)
    results = run_suite(make_agent, [case], trials=2)
    assert all(r["passed"] for r in results)
    assert summarize(results)["by_case"] == {"eiffel": 1.0}


def test_gate_flags_drops_and_critical_regressions():
    cases = [Case(id="a", question="", category="x", critical=True),
             Case(id="b", question="", category="x")]
    baseline = {"pass_rate": 1.0, "by_case": {"a": 1.0, "b": 1.0}}
    now = {"pass_rate": 0.5, "by_case": {"a": 0.5, "b": 0.5}}
    problems = compare(now, baseline, cases)
    assert any("pass rate" in p for p in problems)
    assert "critical case a regressed" in problems
    steady = {"pass_rate": 0.97, "by_case": {"a": 1.0, "b": 0.94}}
    assert compare(steady, baseline, cases) == []
