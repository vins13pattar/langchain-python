"""
01_basic_tools.py
=================
Demonstrates the SIMPLEST ways to create tools in LangChain.

Concepts covered:
  - @tool decorator                — turn any Python function into a tool
  - Docstring as tool description  — the model reads this to decide when to use it
  - Type hints as input schema     — required for tool argument validation
  - Custom tool name               — override the function name
  - Custom tool description        — override the docstring description
  - Tool properties                — .name, .description, .args_schema
  - Calling a tool directly        — tool.invoke({...})

Tools = callable functions with well-defined inputs/outputs.
The model reads the name + description to decide WHEN to call a tool
and the type hints to know WHAT arguments to pass.
"""

import os
from dotenv import load_dotenv

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

load_dotenv()


# ════════════════════════════════════════════════════════════════════
# 1. BASIC @tool DECORATOR
# ════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Basic Tools Demo")
print("=" * 60)


# Minimal example — docstring IS the description the model reads.
# Type hints ARE required — they define the tool's input schema.

@tool
def add_numbers(a: float, b: float) -> str:
    """Add two numbers and return the result.

    Args:
        a: The first number
        b: The second number
    """
    return f"{a} + {b} = {a + b}"


@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Use this when you need to find customer records, orders,
    or product information.

    Args:
        query: Search terms to look for (2-8 words recommended)
        limit: Maximum number of results to return (default: 10)
    """
    # Simulated database results
    sample_results = [
        f"Record {i+1}: {query} match #{i+1}"
        for i in range(min(limit, 3))
    ]
    return f"Found {limit} results for '{query}':\n" + "\n".join(sample_results)


@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date and time.

    Args:
        timezone: Timezone name (e.g. 'UTC', 'US/Eastern', 'Asia/Kolkata')
    """
    from datetime import datetime, timezone as tz
    now = datetime.now(tz.utc)
    return f"Current time ({timezone}): {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"


# ── Inspect tool properties ───────────────────────────────────────
print("\n── Tool Properties ───────────────────────────────────────")
print(f"\n  add_numbers:")
print(f"    .name        = {add_numbers.name!r}")
print(f"    .description = {add_numbers.description!r}")
print(f"    .args        = {add_numbers.args}")

print(f"\n  search_database:")
print(f"    .name        = {search_database.name!r}")
print(f"    .description = {search_database.description[:60]!r}…")
print(f"    .args        = {search_database.args}")

# ── Call a tool directly (without an agent) ────────────────────────
print("\n── Direct Tool Invocation ────────────────────────────────")

result = add_numbers.invoke({"a": 42, "b": 8})
print(f"\n  add_numbers(42, 8)         → {result}")

result = search_database.invoke({"query": "Vinod", "limit": 3})
print(f"\n  search_database('Vinod', 3):\n    {result}")

result = get_current_time.invoke({})
print(f"\n  get_current_time()         → {result}")


# ════════════════════════════════════════════════════════════════════
# 2. CUSTOM TOOL NAME AND DESCRIPTION
# ════════════════════════════════════════════════════════════════════

print("\n── Custom Tool Name & Description ────────────────────────")

# Override the function name with a custom tool name
@tool("web_search")                     # ← custom name (no spaces/special chars)
def search(query: str) -> str:
    """Search the web for current information about any topic.

    Use this when you need recent facts, news, or information
    that may not be in your training data.

    Args:
        query: Search query (2-8 keywords recommended)
    """
    return f"Web search results for '{query}': [simulated result]"


# Override BOTH name and description inline
@tool(
    "safe_calculator",
    description=(
        "Performs arithmetic calculations. "
        "Use this for any math problems: addition, subtraction, "
        "multiplication, division, percentages."
    )
)
def calc(expression: str) -> str:
    """Evaluate a safe mathematical expression.

    Args:
        expression: Math expression such as '2 + 2' or '100 * 1.08'
    """
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: unsupported characters in expression"
        return f"Result: {eval(expression)}"  # noqa: S307
    except Exception as e:
        return f"Error: {e}"


print(f"\n  web_search.name        = {search.name!r}")
print(f"  safe_calculator.name   = {calc.name!r}")
print(f"  safe_calculator.desc   = {calc.description[:60]!r}…")

print(f"\n  web_search.invoke('Python 3.12'):   {search.invoke({'query': 'Python 3.12'})}")
print(f"  safe_calculator.invoke('100 * 1.08'): {calc.invoke({'expression': '100 * 1.08'})}")


# ════════════════════════════════════════════════════════════════════
# 3. GOOD vs BAD TOOL DESCRIPTIONS
# ════════════════════════════════════════════════════════════════════

print("\n── Good vs Bad Tool Descriptions ────────────────────────")

# ❌ BAD — vague description, model doesn't know when to use it
@tool
def bad_tool(input: str) -> str:
    """Does stuff with input."""
    return "result"

# ✅ GOOD — specific description, clear when to use it and why
@tool
def good_search(query: str) -> str:
    """Search the web for current news and information about a topic.

    Use this when you need:
    - Recent events or news (past 24-48 hours)
    - Current prices, statistics, or data
    - Information that may have changed since your training cutoff

    Do NOT use for general knowledge questions you can answer directly.

    Args:
        query: Search keywords (2-8 words recommended)
    """
    return f"Results for: {query}"

print(f"\n  BAD  description: {bad_tool.description!r}")
print(f"  GOOD description: {good_search.description[:80]!r}…")
print("\n  KEY RULE: The docstring IS the model's only guide for when to use the tool.")


# ════════════════════════════════════════════════════════════════════
# 4. USING TOOLS WITH AN AGENT
# ════════════════════════════════════════════════════════════════════

print("\n── Tools with an Agent ───────────────────────────────────")

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[add_numbers, search, calc, get_current_time],
    system_prompt=(
        "You are a helpful assistant. Use the available tools when needed. "
        "Show your reasoning and which tools you are calling."
    ),
)

questions = [
    "What is 1,234 multiplied by 5.67?",
    "What time is it right now?",
]

for q in questions:
    result = agent.invoke({"messages": [HumanMessage(q)]})
    print(f"\n  🧑 {q}")
    print(f"  🤖 {result['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# 5. TOOL NAMING BEST PRACTICES
# ════════════════════════════════════════════════════════════════════

print("\n── Naming Best Practices ─────────────────────────────────")
print("""
  ✅ Use snake_case:         get_weather, search_web, calculate_tax
  ✅ Use alphanumeric + _:   read_file, write_file_v2, send_email_v1
  ❌ Avoid spaces:           "Get Weather" → use get_weather
  ❌ Avoid special chars:    search-web, calc.result → use snake_case
  ❌ Avoid reserved words:   config, runtime → these are system-reserved

  Some providers (OpenAI, Anthropic) will reject tool names with spaces
  or special characters — always use snake_case for compatibility.
""")
