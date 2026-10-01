"""Chapter 11: the Model Context Protocol over stdio, by hand.

MCP is JSON-RPC 2.0. Over the stdio transport, each message is one line
of JSON. A client starts the server as a subprocess, sends `initialize`,
then lists and calls tools.
"""
import itertools
import json
import subprocess
import sys

from .tools import Tool, Toolbox, execute_tool_call

SUPPORTED_VERSIONS = ["2026-07-28", "2025-11-25", "2025-06-18"]


class MCPError(Exception):
    pass


# ---------------------------------------------------------------- server

def handle_message(message, toolbox, server_info):
    """Returns the JSON-RPC response for one request, or None."""
    method, params = message.get("method"), message.get("params") or {}
    if "id" not in message:          # a notification: no response
        return None
    if method == "initialize":
        requested = params.get("protocolVersion")
        version = (requested if requested in SUPPORTED_VERSIONS
                   else SUPPORTED_VERSIONS[0])
        result = {"protocolVersion": version,
                  "capabilities": {"tools": {}},
                  "serverInfo": server_info}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": [
            {"name": s["name"], "description": s["description"],
             "inputSchema": s["input_schema"]}
            for s in toolbox.schemas]}
    elif method == "tools/call":
        schemas = {s["name"]: s for s in toolbox.schemas}
        output = execute_tool_call(
            {"name": params["name"],
             "input": params.get("arguments") or {}},
            toolbox.functions, schemas)
        is_error = isinstance(output, dict) and "error" in output
        text = output if isinstance(output, str) else json.dumps(
            output, default=str)
        result = {"content": [{"type": "text", "text": text}],
                  "isError": is_error}
    else:
        return {"jsonrpc": "2.0", "id": message["id"],
                "error": {"code": -32601,
                          "message": f"Method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": message["id"], "result": result}


def serve_stdio(toolbox, name, version="1.0.0",
                stdin=sys.stdin, stdout=sys.stdout):
    server_info = {"name": name, "version": version}
    for line in stdin:
        if not line.strip():
            continue
        response = handle_message(json.loads(line), toolbox, server_info)
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()


# ---------------------------------------------------------------- client

class MCPClient:
    def __init__(self, command, client_name="agentic-python"):
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self._ids = itertools.count(1)
        init = self.request("initialize", {
            "protocolVersion": SUPPORTED_VERSIONS[0],
            "capabilities": {},
            "clientInfo": {"name": client_name, "version": "1.0.0"},
        })
        if init["protocolVersion"] not in SUPPORTED_VERSIONS:
            self.close()
            raise MCPError(f"Unsupported protocol version: "
                           f"{init['protocolVersion']}")
        self.server_info = init.get("serverInfo", {})
        self.notify("notifications/initialized")

    def _send(self, message):
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def notify(self, method, params=None):
        self._send({"jsonrpc": "2.0", "method": method,
                    **({"params": params} if params else {})})

    def request(self, method, params=None):
        request_id = next(self._ids)
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method,
                    "params": params or {}})
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise MCPError("Server closed the connection")
            message = json.loads(line)
            if message.get("id") != request_id:
                continue             # notifications, log messages, etc.
            if "error" in message:
                raise MCPError(message["error"]["message"])
            return message["result"]

    def list_tools(self):
        return self.request("tools/list")["tools"]

    def call_tool(self, name, arguments):
        result = self.request("tools/call",
                              {"name": name, "arguments": arguments})
        text = "\n".join(c["text"] for c in result["content"]
                         if c["type"] == "text")
        if result.get("isError"):
            raise MCPError(text)
        return text

    def toolbox(self, prefix=""):
        """Every remote tool, ready to hand to an Agent."""
        box = Toolbox()
        for remote in self.list_tools():
            def call(_name=remote["name"], **arguments):
                return self.call_tool(_name, arguments)

            box.add(Tool(call, {
                "name": prefix + remote["name"],
                "description": remote.get("description", ""),
                "input_schema": remote["inputSchema"],
            }))
        return box

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=5)
