"""Real local MCP protocol fixture; never imported by production."""

import os
import sys

from mcp.server.fastmcp import FastMCP

server = FastMCP("test-server", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8000)


@server.tool()
def echo(text: str) -> dict:
    return {"text": text, "credential": os.environ.get("FIXTURE_CREDENTIAL", "absent")}


@server.tool()
def rejected() -> str:
    raise ValueError("deliberate diagnostic")


@server.tool()
def forbidden() -> str:
    return "must not be callable"


if __name__ == "__main__":
    server.run(transport="streamable-http" if len(sys.argv) > 1 else "stdio")
