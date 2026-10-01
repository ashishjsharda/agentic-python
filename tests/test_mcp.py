"""Chapter 11: our MCP code interoperates with the official SDK."""
import io
import json
import sys

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agentic.agent import Agent
from agentic.mcp import MCPClient, MCPError, handle_message, serve_stdio
from agentic.testing import FakeClient, text, tool_use
from agentic.tools import Toolbox, calculator_tool

OUR_SERVER = [sys.executable, "-m", "research_assistant.mcp_server"]
SDK_SERVER = [sys.executable, "tests/fixtures/sdk_server.py"]


def test_server_negotiates_protocol_version():
    box = Toolbox(calculator_tool)
    info = {"name": "t", "version": "1"}
    old = handle_message({"jsonrpc": "2.0", "id": 1,
                          "method": "initialize",
                          "params": {"protocolVersion": "2025-06-18"}},
                         box, info)
    assert old["result"]["protocolVersion"] == "2025-06-18"
    unknown = handle_message({"jsonrpc": "2.0", "id": 2,
                              "method": "initialize",
                              "params": {"protocolVersion": "1999"}},
                             box, info)
    assert unknown["result"]["protocolVersion"] == "2026-07-28"
    missing = handle_message({"jsonrpc": "2.0", "id": 3,
                              "method": "nope"}, box, info)
    assert missing["error"]["code"] == -32601
    assert handle_message({"jsonrpc": "2.0",
                           "method": "notifications/initialized"},
                          box, info) is None


def test_serve_stdio_speaks_line_delimited_json():
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "calculator",
                    "arguments": {"expression": "6*7"}}},
    ]
    stdin = io.StringIO("\n".join(json.dumps(r) for r in requests))
    stdout = io.StringIO()
    serve_stdio(Toolbox(calculator_tool), "t", stdin=stdin, stdout=stdout)
    listed, called = [json.loads(line)
                      for line in stdout.getvalue().splitlines()]
    assert listed["result"]["tools"][0]["name"] == "calculator"
    assert called["result"]["content"][0]["text"] == "42"


def test_our_client_against_the_official_sdk_server():
    client = MCPClient(SDK_SERVER)
    try:
        assert [t["name"] for t in client.list_tools()] == ["add"]
        assert client.call_tool("add", {"a": 2, "b": 40}) == "42"
        with pytest.raises(MCPError):
            client.call_tool("add", {"a": "two"})
    finally:
        client.close()


def test_official_sdk_client_against_our_server():
    async def session():
        params = StdioServerParameters(command=OUR_SERVER[0],
                                       args=OUR_SERVER[1:])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as s:
                await s.initialize()
                names = [t.name for t in (await s.list_tools()).tools]
                ok = await s.call_tool("calculator",
                                       {"expression": "17*23"})
                bad = await s.call_tool("calculator", {"expr": "1"})
                return names, ok, bad

    names, ok, bad = anyio.run(session)
    assert names == ["search_wikipedia", "read_wikipedia", "calculator"]
    assert ok.content[0].text == "391" and not ok.is_error
    assert bad.is_error


def test_agent_uses_remote_mcp_tools():
    client = MCPClient(SDK_SERVER)
    try:
        box = client.toolbox(prefix="sdk_")
        fake = FakeClient([tool_use("sdk_add", {"a": 17, "b": 25}),
                           text("42")])
        agent = Agent(fake, box.functions, box.schemas)
        assert agent.run("17 + 25?") == "42"
        result = fake.calls[1]["messages"][-1]["content"][0]
        assert result["content"] == "42" and not result["is_error"]
    finally:
        client.close()
