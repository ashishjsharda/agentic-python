"""Chapter 8: plan multi-part questions, then check the answer."""
from pydantic import BaseModel, Field

from agentic.structured import extract


class ResearchPlan(BaseModel):
    sub_questions: list[str] = Field(
        min_length=1, max_length=5,
        description="Independent questions whose answers together "
                    "answer the original. One item if it's simple.")
    needs_calculation: bool


class Review(BaseModel):
    complete: bool = Field(description="Does the answer fully address "
                                       "every part of the question?")
    missing: str = Field(default="", description="What is missing")


def plan_question(client, question):
    return extract(client, ResearchPlan,
                   f"Plan the research for this question:\n{question}",
                   name="record_plan")


def review_answer(client, question, answer):
    return extract(client, Review,
                   f"Question: {question}\n\nAnswer: {answer.answer}\n\n"
                   f"Citations: {len(answer.citations)}",
                   name="record_review")


def ask_with_plan(assistant, client, question, max_revisions=1):
    plan = plan_question(client, question)
    prompt = question
    if len(plan.sub_questions) > 1:
        steps = "\n".join(f"{i}. {q}" for i, q in
                          enumerate(plan.sub_questions, 1))
        prompt = f"{question}\n\nResearch plan:\n{steps}"
    answer = assistant.run(prompt)
    for _ in range(max_revisions):
        review = review_answer(client, question, answer)
        if review.complete:
            break
        answer = assistant.run(f"{prompt}\n\nA previous answer missed: "
                               f"{review.missing}. Fix that.")
    return answer
