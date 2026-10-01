"""Chapter 7: plan-then-execute, self-critique, and replanning."""

from .llm import extract_text
from .structured import parse_json_object

PLAN_PROMPT = (
    "Break this task into 2-6 ordered, concrete steps. Respond with "
    'only JSON: {"steps": ["...", "..."]}'
)

CRITIQUE_PROMPT = (
    "Does the answer fully address the task? Respond with only JSON: "
    '{"complete": true or false, "missing": "what is missing, or empty"}'
)


def make_plan(task, client):
    prompt = f"{PLAN_PROMPT}\n\nTask: {task}"
    response = client.complete([{"role": "user", "content": prompt}])
    return parse_json_object(extract_text(response))["steps"]


def run_plan(task, steps, agent):
    notes = []
    for i, step in enumerate(steps, start=1):
        done = "\n".join(notes) or "(nothing yet)"
        result = agent.run(f"Overall task: {task}\n"
                           f"Completed so far:\n{done}\n\n"
                           f"Now do step {i}: {step}")
        notes.append(f"Step {i} ({step}): {result}")
    return notes


def critique(task, answer, client):
    response = client.complete([{"role": "user", "content": (
        f"{CRITIQUE_PROMPT}\n\nTask: {task}\n\nAnswer: {answer}")}])
    return parse_json_object(extract_text(response))


def plan_and_execute(task, client, agent, max_replans=1):
    goal = task
    for _ in range(max_replans + 1):
        notes = run_plan(task, make_plan(goal, client), agent)
        answer = notes[-1] if notes else ""
        verdict = critique(task, "\n".join(notes), client)
        if verdict.get("complete"):
            return answer
        goal = f"{task}\n\nA previous attempt missed: {verdict['missing']}"
    return answer
