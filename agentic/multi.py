"""Chapter 8: a supervisor that calls other agents as tools."""
from .agent import Agent

SUPERVISOR_SYSTEM = (
    "You coordinate specialist agents. Delegate each part of the task "
    "to the right specialist, then combine their answers. Do not "
    "invent facts the specialists did not return."
)


def agent_as_tool(agent, name, description):
    def run_worker(task):
        return agent.run(task)

    schema = {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": {"task": {
                "type": "string",
                "description": "A self-contained instruction for this "
                               "specialist, with all needed context.",
            }},
            "required": ["task"],
            "additionalProperties": False,
        },
    }
    return run_worker, schema


def build_supervisor(client, workers, system=SUPERVISOR_SYSTEM,
                     max_steps=8, trace=None, agent_cls=Agent):
    tools, schemas = {}, []
    for agent, name, description in workers:
        fn, schema = agent_as_tool(agent, name, description)
        tools[name] = fn
        schemas.append(schema)
    return agent_cls(client, tools, schemas, system=system,
                     max_steps=max_steps, trace=trace)
