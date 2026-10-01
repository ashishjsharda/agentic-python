"""A server built with the official MCP SDK, for interop tests."""
from mcp.server.mcpserver import MCPServer

server = MCPServer("sdk-demo")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


if __name__ == "__main__":
    server.run()
