"""
01_built_in_middleware.py
=========================
Demonstrates the most commonly used BUILT-IN middleware shipped with LangChain.

Concepts covered:
  - SummarizationMiddleware  — Compresses long conversation history automatically
  - ModelCallLimitMiddleware — Caps the total number of LLM calls per run
  - ToolCallLimitMiddleware  — Limits how many times a specific tool can be called
  - ToolRetryMiddleware      — Automatically retries tool calls on transient errors
  - Stacking multiple middleware in one agent
"""

import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain.tools import tool

load_dotenv()

print("=" * 60)
print("Built-In Middleware Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# SHARED TOOLS
# ════════════════════════════════════════════════════════════════════

call_count = {"search": 0}

@tool
def web_search(query: str) -> str:
    """Search the web for the latest information on a topic."""
    call_count["search"] += 1
    print(f"  [Tool] web_search called (#{call_count['search']}): '{query}'")
    return f"Search results for '{query}': Sample result from the web."


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    try:
        result = eval(expression)  # noqa: S307 – safe in demo context
        print(f"  [Tool] calculator: {expression} = {result}")
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# ════════════════════════════════════════════════════════════════════
# 1. SUMMARIZATION MIDDLEWARE
#    Automatically compresses the conversation history when it grows
#    beyond a token threshold, keeping the context window manageable.
# ════════════════════════════════════════════════════════════════════

print("\n── 1. SummarizationMiddleware ───────────────────────────────")

agent_summarized = create_agent(
    model="openai:gpt-4o-mini",
    tools=[web_search, calculator],
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",  # Model used to summarize
            max_tokens=4000,             # Threshold before summarization kicks in
        )
    ],
    system_prompt="You are a helpful research assistant.",
)

result_summarized = agent_summarized.invoke({
    "messages": [{"role": "user", "content": "Search for Python 3.13 features."}]
})

print(f"Response: {result_summarized['messages'][-1].content[:120]}...")


# ════════════════════════════════════════════════════════════════════
# 2. MODEL CALL LIMIT MIDDLEWARE
#    Stops the agent after N LLM calls to prevent runaway costs.
# ════════════════════════════════════════════════════════════════════

print("\n── 2. ModelCallLimitMiddleware ──────────────────────────────")

agent_model_limited = create_agent(
    model="openai:gpt-4o-mini",
    tools=[web_search, calculator],
    middleware=[
        ModelCallLimitMiddleware(max_calls=3)  # At most 3 LLM round-trips
    ],
    system_prompt="You are a research assistant.",
)

result_model_limited = agent_model_limited.invoke({
    "messages": [{"role": "user", "content": "What is 99 * 87?"}]
})

print(f"Response: {result_model_limited['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# 3. TOOL CALL LIMIT MIDDLEWARE
#    Limits how many times a specific tool can be invoked in one run.
# ════════════════════════════════════════════════════════════════════

print("\n── 3. ToolCallLimitMiddleware ───────────────────────────────")

agent_tool_limited = create_agent(
    model="openai:gpt-4o-mini",
    tools=[web_search, calculator],
    middleware=[
        ToolCallLimitMiddleware(limits={"web_search": 2})  # Max 2 searches per run
    ],
    system_prompt="You are a research assistant.",
)

result_tool_limited = agent_tool_limited.invoke({
    "messages": [{"role": "user", "content":
        "Search for AI news, then search for Python news."}]
})

print(f"Response: {result_tool_limited['messages'][-1].content[:120]}...")


# ════════════════════════════════════════════════════════════════════
# 4. TOOL RETRY MIDDLEWARE
#    Automatically retries a failed tool call up to N times before
#    propagating the error to the agent as a tool message.
# ════════════════════════════════════════════════════════════════════

print("\n── 4. ToolRetryMiddleware ───────────────────────────────────")

attempt = {"n": 0}

@tool
def flaky_api(endpoint: str) -> str:
    """Call an external API endpoint (may fail transiently)."""
    attempt["n"] += 1
    print(f"  [Tool] flaky_api attempt #{attempt['n']}: '{endpoint}'")
    if attempt["n"] < 3:
        raise ConnectionError("Temporary network failure — please retry.")
    return f"API response from {endpoint}: Success after {attempt['n']} attempts."


agent_retry = create_agent(
    model="openai:gpt-4o-mini",
    tools=[flaky_api],
    middleware=[
        ToolRetryMiddleware(max_retries=3)  # Retry up to 3 times
    ],
    system_prompt="You are an API integration assistant.",
)

result_retry = agent_retry.invoke({
    "messages": [{"role": "user", "content": "Call the /status endpoint."}]
})

print(f"Response: {result_retry['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# 5. STACKING MULTIPLE MIDDLEWARE
#    Middleware runs in declaration order (top → bottom). Stack them
#    to combine capabilities in a single production agent.
# ════════════════════════════════════════════════════════════════════

print("\n── 5. Stacking Multiple Middleware ──────────────────────────")

agent_stacked = create_agent(
    model="openai:gpt-4o-mini",
    tools=[web_search, calculator],
    middleware=[
        SummarizationMiddleware(model="openai:gpt-4o-mini", max_tokens=4000),
        ModelCallLimitMiddleware(max_calls=5),
        ToolCallLimitMiddleware(limits={"web_search": 3}),
        ToolRetryMiddleware(max_retries=2),
    ],
    system_prompt="You are a production-grade research assistant.",
)

result_stacked = agent_stacked.invoke({
    "messages": [{"role": "user", "content": "Search for LangChain v1.3 features."}]
})

print(f"Response: {result_stacked['messages'][-1].content[:120]}...")
print("\n✅ Built-in middleware demo complete.")
