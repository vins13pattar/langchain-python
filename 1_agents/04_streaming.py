"""
04_streaming.py
===============
Demonstrates STREAMING agent output — see tool calls and messages as they happen.

Concepts covered:
  - agent.stream() — yields state snapshots as the agent runs
  - stream_mode="values" — each chunk is the full state at that step
  - Detecting HumanMessage vs AIMessage vs tool calls in the stream
  - Why streaming matters: better UX for slow/multi-step agents

Without streaming the user waits silently until the entire run finishes.
With streaming you can show progress in real time — tool calls, partial
answers, and intermediate steps.
"""

import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

load_dotenv()


# ── Simulated slow tools ──────────────────────────────────────────────────────

@tool
def search_web(query: str) -> str:
    """Search the web for recent news and information.

    Args:
        query: Search query
    """
    import time
    time.sleep(0.3)   # simulate network latency
    return (
        f"Search results for '{query}': "
        "1) AI advances reshape software development (2026). "
        "2) LangChain releases new agent harness with middleware support. "
        "3) Open-source LLMs reach GPT-4 parity on coding benchmarks."
    )


@tool
def fetch_stock_price(ticker: str) -> str:
    """Fetch the current stock price for a given ticker symbol.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL, GOOGL)
    """
    import time, random
    time.sleep(0.2)
    prices = {"AAPL": 213.45, "GOOGL": 178.30, "MSFT": 425.80, "NVDA": 950.12}
    price = prices.get(ticker.upper(), random.uniform(50, 300))
    return f"{ticker.upper()}: ${price:.2f}"


# ── Agent ─────────────────────────────────────────────────────────────────────

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_web, fetch_stock_price],
    system_prompt=(
        "You are a financial research assistant. Search the web and check "
        "stock prices when asked. Show all steps clearly."
    ),
)


# ── Streaming helper ──────────────────────────────────────────────────────────

def stream_agent(question: str) -> None:
    """Run the agent and print each step as it arrives."""
    print(f"\n🧑 User: {question}")
    print("─" * 50)

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",   # each chunk = full state snapshot
    ):
        # chunk["messages"] is the complete message list so far.
        # Look at only the LAST message (what just changed this step).
        latest = chunk["messages"][-1]

        if isinstance(latest, HumanMessage):
            # This is the user's own message echoed back — skip
            pass

        elif isinstance(latest, AIMessage):
            if latest.content:
                # Model produced text
                print(f"🤖 Agent: {latest.content}")
            elif latest.tool_calls:
                # Model decided to call one or more tools
                names = [tc["name"] for tc in latest.tool_calls]
                args  = [tc["args"] for tc in latest.tool_calls]
                print(f"🔧 Calling tools: {names}")
                for name, arg in zip(names, args):
                    print(f"   └─ {name}({arg})")

        elif isinstance(latest, ToolMessage):
            # Tool has returned its result
            print(f"📦 Tool result [{latest.name}]: {latest.content[:120]}")

    print("─" * 50)
    print("✅ Done\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Agent Streaming Demo")
    print("=" * 60)

    # Example 1 — single tool call
    stream_agent("What are the latest AI news headlines?")

    # Example 2 — multiple tool calls
    stream_agent(
        "Look up the latest AI news, then get stock prices for NVDA and MSFT. "
        "Give me a short summary at the end."
    )
