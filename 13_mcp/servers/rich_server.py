"""
servers/rich_server.py
=======================
An MCP server that demonstrates advanced tool features:
  - Structured content (JSON alongside text response)
  - Resources (expose files/data as Blobs)
  - Prompts (reusable prompt templates)
  - Progress notifications for long-running ops

Run: python servers/rich_server.py  (starts HTTP on port 8001)
"""

import asyncio
import json
from fastmcp import FastMCP, Context

mcp = FastMCP("Rich")


# ── Tools with structured content ────────────────────────────────────

@mcp.tool()
def get_stock_price(symbol: str) -> dict:
    """
    Get the current stock price and metadata for a ticker symbol.

    Args:
        symbol: Stock ticker symbol (e.g. AAPL, GOOGL).
    """
    prices = {
        "AAPL":  {"price": 189.50, "change": +1.2,  "volume": 52_000_000},
        "GOOGL": {"price": 178.30, "change": -0.8,  "volume": 18_000_000},
        "MSFT":  {"price": 415.60, "change": +2.1,  "volume": 22_000_000},
    }
    symbol = symbol.upper()
    data   = prices.get(symbol, {"price": 100.00, "change": 0.0, "volume": 1_000_000})
    data["symbol"] = symbol
    return data  # FastMCP serializes dict as structured content


@mcp.tool()
async def run_analysis(dataset: str, ctx: Context) -> str:
    """
    Run analysis on a dataset (demonstrates progress notifications).

    Args:
        dataset: Name of the dataset to analyze.
    """
    stages = ["Loading data", "Cleaning", "Computing stats", "Generating report"]
    for i, stage in enumerate(stages, 1):
        await ctx.report_progress(progress=i, total=len(stages), message=stage)
        await asyncio.sleep(0.1)  # simulate work
    return f"Analysis of '{dataset}' complete: mean=42.5, std=7.3, outliers=3"


# ── Resources ─────────────────────────────────────────────────────────

@mcp.resource("file:///data/config.json")
def get_config() -> str:
    """Application configuration file."""
    return json.dumps({
        "version":  "1.0.0",
        "debug":    False,
        "max_retries": 3,
        "timeout_seconds": 30,
    }, indent=2)


@mcp.resource("file:///data/readme.txt")
def get_readme() -> str:
    """Project README file."""
    return (
        "Rich MCP Server\n"
        "================\n"
        "This server demonstrates advanced MCP features:\n"
        "  - Structured content tools\n"
        "  - File resources\n"
        "  - Prompt templates\n"
        "  - Progress notifications\n"
    )


# ── Prompts ────────────────────────────────────────────────────────────

@mcp.prompt()
def summarize(text: str) -> str:
    """Summarize the provided text concisely."""
    return f"Please summarize the following text in 3 bullet points:\n\n{text}"


@mcp.prompt()
def code_review(code: str, language: str = "python", focus: str = "quality") -> list[dict]:
    """Perform a code review with configurable focus."""
    return [
        {
            "role": "user",
            "content": (
                f"Please review this {language} code with a focus on {focus}:\n\n"
                f"```{language}\n{code}\n```\n\n"
                "Provide specific, actionable feedback."
            )
        }
    ]


if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8001)
