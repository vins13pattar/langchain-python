"""
03_state_and_values_streaming.py
==================================
Demonstrates STATE SNAPSHOT streaming and FINAL OUTPUT access.

Concepts covered:
  - stream.values          — state snapshots after every step
  - stream.output          — final agent state after run completes
  - Custom state fields in snapshots
  - Detecting agent progress from state deltas
  - Combining stream.values + stream.messages in one run
  - stream.interleave()    — consume multiple projections together (sync)

stream.values gives you a live view of the agent's state as it evolves:
  • After each model call
  • After each tool execution
  • At the end of the run

This is useful for progress monitoring, dashboards, and debugging.
"""

import os
import uuid
from dotenv import load_dotenv

from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

load_dotenv()

print("=" * 60)
print("State & Values Streaming Demo")
print("=" * 60)


# ── Tools ─────────────────────────────────────────────────────────

@tool
def fetch_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker.

    Args:
        ticker: Stock symbol (e.g. AAPL, GOOGL)
    """
    prices = {"AAPL": 213.45, "GOOGL": 178.30, "MSFT": 425.80, "AMZN": 198.65}
    price  = prices.get(ticker.upper(), 0)
    return f"${price:.2f}" if price else f"Unknown ticker '{ticker}'"


@tool
def get_company_info(ticker: str) -> str:
    """Get basic company information for a stock ticker.

    Args:
        ticker: Stock symbol
    """
    companies = {
        "AAPL":  "Apple Inc. — Consumer electronics, iPhone, Mac. HQ: Cupertino, CA",
        "GOOGL": "Alphabet Inc. — Search, Cloud, YouTube. HQ: Mountain View, CA",
        "MSFT":  "Microsoft Corp — Software, Azure, Teams. HQ: Redmond, WA",
    }
    return companies.get(ticker.upper(), f"No data for '{ticker}'")


agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[fetch_stock_price, get_company_info],
    system_prompt="You are a stock research assistant. Be concise.",
)

INPUT = {
    "messages": [{
        "role": "user",
        "content": "Get the price and company info for AAPL and MSFT."
    }]
}


# ════════════════════════════════════════════════════════════════════
# 1. stream.values — state snapshots after each step
# ════════════════════════════════════════════════════════════════════

print("\n── 1. stream.values (state snapshots) ────────────────────")

stream = agent.stream_events(INPUT, version="v3")

snapshot_count = 0
for snapshot in stream.values:
    snapshot_count += 1
    msgs    = snapshot.get("messages", [])
    last    = msgs[-1] if msgs else None
    msg_type = type(last).__name__ if last else "None"
    content_preview = ""
    if last:
        c = last.content if isinstance(last.content, str) else str(last.content)
        content_preview = c[:60] + "…" if len(c) > 60 else c

    print(f"\n  Snapshot #{snapshot_count}:")
    print(f"    Messages in state: {len(msgs)}")
    print(f"    Last message type: {msg_type}")
    print(f"    Last content:      {content_preview!r}")

print(f"\n  Total snapshots: {snapshot_count}  (one per step: model + tool calls)")


# ════════════════════════════════════════════════════════════════════
# 2. stream.output — final agent state
# ════════════════════════════════════════════════════════════════════

print("\n── 2. stream.output (final state) ───────────────────────")

stream = agent.stream_events(INPUT, version="v3")

# Drain all projections
for _ in stream.values:
    pass

final = stream.output
print(f"\n  Final state keys:       {list(final.keys())}")
print(f"  Total messages in run:  {len(final['messages'])}")
print(f"  Final answer:           {final['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# 3. CUSTOM STATE FIELDS in snapshots
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Custom state fields in snapshots ───────────────────")


class TrackingState(AgentState):
    tool_call_count: int = 0
    tickers_looked_up: list = []


@tool
def tracked_stock_price(ticker: str, runtime: ToolRuntime) -> Command:
    """Get stock price and update tracking state.

    Args:
        ticker: Stock symbol
    """
    prices = {"AAPL": 213.45, "GOOGL": 178.30, "MSFT": 425.80}
    price  = prices.get(ticker.upper(), 0)
    result = f"${price:.2f}" if price else f"Unknown '{ticker}'"

    # Write to custom state fields
    current_count   = runtime.state.get("tool_call_count", 0)
    current_tickers = runtime.state.get("tickers_looked_up", [])

    return Command(update={
        "tool_call_count":    current_count + 1,
        "tickers_looked_up":  current_tickers + [ticker.upper()],
        "messages": [ToolMessage(
            content=result,
            tool_call_id=runtime.tool_call_id,
        )],
    })


tracking_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[tracked_stock_price],
    state_schema=TrackingState,
    checkpointer=MemorySaver(),
    system_prompt="You are a stock assistant. Track prices using the tool.",
)

config = {"configurable": {"thread_id": str(uuid.uuid4())}}
stream = tracking_agent.stream_events(
    {
        "messages": [{"role": "user", "content": "Get prices for AAPL, MSFT, and GOOGL."}],
        "tool_call_count":   0,
        "tickers_looked_up": [],
    },
    config,
    version="v3",
)

print(f"\n  Custom state progression:")
for snapshot in stream.values:
    count   = snapshot.get("tool_call_count", 0)
    tickers = snapshot.get("tickers_looked_up", [])
    if tickers:     # only show when state has changed
        print(f"    tool_call_count={count}, tickers_looked_up={tickers}")

final = stream.output
print(f"\n  Final: {final.get('tool_call_count', 0)} tool calls, tickers: {final.get('tickers_looked_up', [])}")


# ════════════════════════════════════════════════════════════════════
# 4. stream.interleave() — multiple projections in one loop (sync)
# ════════════════════════════════════════════════════════════════════

print("\n── 4. stream.interleave() ─────────────────────────────────")

stream = agent.stream_events(INPUT, version="v3")

print("\n  Interleaved stream (messages + tool_calls + values):")
for name, item in stream.interleave("messages", "tool_calls", "values"):
    if name == "messages":
        # item is a ChatModelStream
        text = ""
        try:
            # Drain silently - we'll just show the node
            pass
        except Exception:
            pass
        print(f"  [message ] node={item.node}")

    elif name == "tool_calls":
        # item is a tool call handle
        for _ in item.output_deltas:
            pass   # drain
        print(f"  [tool_call] {item.tool_name}({item.input}) → {str(item.output)[:50]}")

    elif name == "values":
        # item is a state snapshot dict
        n_msgs = len(item.get("messages", []))
        print(f"  [snapshot ] {n_msgs} messages in state")


# ════════════════════════════════════════════════════════════════════
# 5. PROGRESS MONITORING WITH stream.values
#    Detect which step the agent is on
# ════════════════════════════════════════════════════════════════════

print("\n── 5. Progress monitoring with stream.values ─────────────")

stream  = agent.stream_events(INPUT, version="v3")
step    = 0
prev_n  = 0

print(f"\n  Step-by-step progress:")
for snapshot in stream.values:
    msgs  = snapshot.get("messages", [])
    n     = len(msgs)
    diff  = n - prev_n

    if diff > 0:
        step += 1
        last = msgs[-1]
        kind = type(last).__name__
        print(f"    Step {step}: +{diff} message(s) → last={kind}")
    prev_n = n

print(f"\n  Final answer: {stream.output['messages'][-1].content}")
