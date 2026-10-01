from agentic.agent import Agent
from agentic.hitl import ApprovalAgent, PausedRun
from agentic.multi import build_supervisor
from agentic.testing import FakeClient, text, tool_use


def test_supervisor_delegates_to_worker_agents():
    worker = Agent(FakeClient([text("Paris")]), {}, [])
    sup_client = FakeClient([
        tool_use("research", {"task": "Capital of France?"}),
        text("The capital is Paris."),
    ])
    supervisor = build_supervisor(sup_client, [
        (worker, "research", "Looks things up.")])
    assert supervisor.run("What is the capital of France?") == \
        "The capital is Paris."
    worker_reply = sup_client.calls[1]["messages"][-1]["content"][0]
    assert worker_reply["content"] == "Paris"


def test_worker_failure_becomes_a_tool_error_not_a_crash():
    stuck = Agent(FakeClient([tool_use("x", {}, id=f"i{n}")
                              for n in range(3)]), {}, [], max_steps=2)
    sup_client = FakeClient([tool_use("research", {"task": "t"}),
                             text("The researcher failed.")])
    supervisor = build_supervisor(sup_client, [
        (stuck, "research", "Looks things up.")])
    assert supervisor.run("q") == "The researcher failed."
    assert sup_client.calls[1]["messages"][-1]["content"][0]["is_error"]


def _approval_agent(client, sent):
    def send_email(to, body):
        sent.append((to, body))
        return "sent"

    schema = {"name": "send_email", "description": "Send an email.",
              "input_schema": {"type": "object", "properties": {
                  "to": {"type": "string"}, "body": {"type": "string"}},
                  "required": ["to", "body"]}}
    return ApprovalAgent(client, {"send_email": send_email}, [schema],
                         needs_approval={"send_email"})


def test_approval_pauses_serializes_and_resumes():
    sent = []
    client = FakeClient([
        tool_use("send_email", {"to": "a@b.co", "body": "hi"}),
        text("Email sent."),
    ])
    paused = _approval_agent(client, sent).run("Email a@b.co hi")
    assert isinstance(paused, PausedRun) and sent == []

    restored = PausedRun.from_json(paused.to_json())
    result = _approval_agent(client, sent).resume(restored,
                                                  approved=True)
    assert result == "Email sent."
    assert sent == [("a@b.co", "hi")]


def test_denied_call_is_reported_to_the_model():
    sent = []
    client = FakeClient([
        tool_use("send_email", {"to": "a@b.co", "body": "hi"}),
        text("Understood, I did not send it."),
    ])
    agent = _approval_agent(client, sent)
    paused = agent.run("Email a@b.co")
    assert agent.resume(paused, approved=False, note="wrong recipient") \
        == "Understood, I did not send it."
    denial = client.calls[1]["messages"][-1]["content"][0]
    assert denial["is_error"] and "wrong recipient" in denial["content"]
    assert sent == []
