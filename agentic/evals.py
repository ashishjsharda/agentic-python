"""Chapter 14: evaluation datasets, graders, and a regression gate."""
import json
import re
import statistics
from dataclasses import dataclass, field

URL_PATTERN = re.compile(r"https?://[^\s\"'<>\\]+")


@dataclass
class Case:
    id: str
    question: str
    category: str
    must_include: list = field(default_factory=list)
    requires_tool: str | None = None
    expect_confidence: list = field(default_factory=list)
    critical: bool = False


def load_cases(path):
    with open(path) as f:
        return [Case(**json.loads(line)) for line in f if line.strip()]


class Recorder:
    """A trace callback that keeps what the grader needs to see."""

    def __init__(self):
        self.tools_called = []
        self.urls_seen = set()
        self.tokens = 0

    def __call__(self, event, data):
        if event == "tool_call":
            self.tools_called.append(data["name"])
        elif event == "tool_results":
            for result in data["results"]:
                self.urls_seen.update(URL_PATTERN.findall(
                    result["content"]))
        elif event == "model_response":
            usage = data["usage"]
            self.tokens += (usage.get("input_tokens", 0)
                            + usage.get("output_tokens", 0))


def normalize(text):
    """Case- and thousands-separator-insensitive matching."""
    return re.sub(r"(?<=\d),(?=\d{3})", "", text.lower())


def grade(case, answer, recorder):
    """Every failed check becomes a readable reason."""
    failures = []
    text = normalize(answer.answer)
    for expected in case.must_include:
        if normalize(expected) not in text:
            failures.append(f"answer lacks {expected!r}")
    for citation in answer.citations:
        if citation.url.startswith("http") and \
                citation.url not in recorder.urls_seen:
            failures.append(f"cites unread source {citation.url}")
    if case.requires_tool and case.requires_tool not in \
            recorder.tools_called:
        failures.append(f"never called {case.requires_tool}")
    if case.expect_confidence and \
            answer.confidence not in case.expect_confidence:
        failures.append(f"confidence {answer.confidence!r} not in "
                        f"{case.expect_confidence}")
    return failures


def run_suite(make_agent, cases, trials=1):
    results = []
    for case in cases:
        for trial in range(trials):
            recorder = Recorder()
            try:
                answer = make_agent(recorder).run(case.question)
                failures = grade(case, answer, recorder)
            except Exception as e:
                failures = [f"{type(e).__name__}: {e}"]
            results.append({"id": case.id, "category": case.category,
                            "critical": case.critical, "trial": trial,
                            "passed": not failures,
                            "failures": failures,
                            "tokens": recorder.tokens})
    return results


def summarize(results):
    by_case = {}
    for r in results:
        by_case.setdefault(r["id"], []).append(r["passed"])
    categories = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r["passed"])
    return {
        "pass_rate": round(statistics.mean(r["passed"]
                                           for r in results), 3),
        "by_category": {c: round(statistics.mean(v), 3)
                        for c, v in sorted(categories.items())},
        "by_case": {c: round(statistics.mean(v), 3)
                    for c, v in sorted(by_case.items())},
        "mean_tokens": round(statistics.mean(r["tokens"]
                                             for r in results)),
    }


def compare(summary, baseline, cases, tolerance=0.05):
    """Regressions that should fail the build, as readable strings."""
    problems = []
    if summary["pass_rate"] < baseline["pass_rate"] - tolerance:
        problems.append(f"pass rate {summary['pass_rate']} < baseline "
                        f"{baseline['pass_rate']} - {tolerance}")
    for case in cases:
        was = baseline["by_case"].get(case.id)
        now = summary["by_case"].get(case.id)
        if case.critical and was == 1.0 and now is not None and now < 1:
            problems.append(f"critical case {case.id} regressed")
    return problems
