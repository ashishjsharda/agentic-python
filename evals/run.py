"""Run the research eval suite against a live model.

    python -m evals.run                      # report only
    python -m evals.run --gate               # fail on regression
    python -m evals.run --update-baseline    # accept current results
"""
import argparse
import json
import sys
from pathlib import Path

from agentic.evals import compare, load_cases, run_suite, summarize
from agentic.llm import make_client
from research_assistant.assistant import build_assistant_v1

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=HERE / "research_cases.jsonl")
    parser.add_argument("--baseline", default=HERE / "baseline.json")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    client = make_client()
    results = run_suite(
        lambda recorder: build_assistant_v1(client, trace=recorder),
        cases, trials=args.trials)
    summary = summarize(results)
    print(json.dumps(summary, indent=2))
    for r in results:
        if not r["passed"]:
            print(f"FAIL {r['id']}[{r['trial']}]: "
                  f"{'; '.join(r['failures'])}")

    baseline_path = Path(args.baseline)
    if args.update_baseline or not baseline_path.exists():
        baseline_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Baseline written to {baseline_path}")
        return 0
    problems = compare(summary, json.loads(baseline_path.read_text()),
                       cases)
    for p in problems:
        print(f"REGRESSION: {p}")
    return 1 if (args.gate and problems) else 0


if __name__ == "__main__":
    sys.exit(main())
