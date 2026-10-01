"""Chapter 11: expose the research assistant's tools over MCP.

    python -m research_assistant.mcp_server

Any MCP host (a desktop assistant, an IDE, another agent) can now use
the same Wikipedia and calculator tools the assistant uses.
"""
from agentic.mcp import serve_stdio
from agentic.tools import Toolbox, calculator_tool

from .sources import make_wikipedia_tools


def build_toolbox():
    return Toolbox(*make_wikipedia_tools(), calculator_tool)


if __name__ == "__main__":
    serve_stdio(build_toolbox(), name="research-assistant")
