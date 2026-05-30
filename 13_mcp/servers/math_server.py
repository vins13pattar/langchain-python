"""
servers/math_server.py
=======================
A minimal MCP math server using FastMCP with stdio transport.
Run directly: python servers/math_server.py

This server exposes two tools: add() and multiply().
It is used by the examples in 01_mcp_basics.py and others.
"""

from fastmcp import FastMCP

mcp = FastMCP("Math")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b. Returns an error string if b is zero."""
    if b == 0:
        return "Error: division by zero"
    return a / b


if __name__ == "__main__":
    mcp.run(transport="stdio")
