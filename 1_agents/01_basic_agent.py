"""
01_basic_agent.py
=================
Demonstrates the SIMPLEST possible LangChain agent using create_agent().

Concepts covered:
  - create_agent() — the recommended modern harness
  - @tool decorator — turning a Python function into an agent tool
  - system_prompt — shaping how the agent behaves
  - agent.invoke() — running the agent and reading the result

Agent = Model + Harness
  The harness (create_agent) manages the loop:
    1. Model receives messages + tools
    2. Model decides to call a tool  ──► Tool executes
    3. Result fed back to model      ──► Repeat until done
    4. Model returns final answer
"""

import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()  # Loads OPENAI_API_KEY (or GOOGLE_API_KEY etc.) from .env


# ── 1. Define Tools ──────────────────────────────────────────────────────────
# Any Python callable decorated with @tool becomes an agent tool.
# The docstring IS the tool description — the model reads it to decide when
# to call the tool and what arguments to pass.

@tool
def calculator(operation: str, num1: float, num2: float) -> str:
    """Perform a basic arithmetic operation on two numbers.

    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        num1:      First number
        num2:      Second number
    """
    ops = {
        "add":      lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b,
        "divide":   lambda a, b: a / b if b != 0 else "Error: division by zero",
    }
    if operation not in ops:
        return f"Unknown operation '{operation}'. Use: add, subtract, multiply, divide"
    result = ops[operation](num1, num2)
    return f"{num1} {operation} {num2} = {result}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.

    Args:
        city: Name of the city (e.g. 'London', 'Tokyo')
    """
    # In a real app you'd call a weather API here.
    weather_db = {
        "london":  "☁️  Cloudy, 15°C",
        "tokyo":   "☀️  Sunny,  28°C",
        "new york": "🌧️  Rainy,  18°C",
        "sydney":  "⛅  Partly cloudy, 22°C",
    }
    return weather_db.get(city.lower(), f"Weather data not available for '{city}'")


# ── 2. Create the Agent ──────────────────────────────────────────────────────
# Pass a "provider:model" string — no manual ChatOpenAI instantiation needed.
# create_agent() sets up the entire loop internally.

agent = create_agent(
    model="openai:gpt-4o-mini",          # provider:model  (change to your preferred model)
    tools=[calculator, get_weather],      # tools the model can call
    system_prompt=(
        "You are a helpful assistant with access to a calculator and weather tool. "
        "Use the tools when needed. Always show your reasoning."
    ),
)


# ── 3. Invoke the Agent ──────────────────────────────────────────────────────
# agent.invoke() takes a dict with a "messages" key.
# The result is also a dict; the final answer is in result["messages"][-1].content

def ask(question: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print(result)
    return result["messages"][-1].content


if __name__ == "__main__":
    print("=" * 60)
    print("Basic LangChain Agent Demo")
    print("=" * 60)

    questions = [
        "What is 1234 multiplied by 56?",
        "What's the weather like in Tokyo?",
        "If I have 500 dollars and spend 137.50, how much is left?",
    ]

    for q in questions:
        print(f"\n🧑 User: {q}")
        answer = ask(q)
        print(f"🤖 Agent: {answer}")
