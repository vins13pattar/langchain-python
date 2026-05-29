"""
03_structured_output.py
=======================
Demonstrates returning STRUCTURED (typed, validated) data from an agent.

Concepts covered:
  - response_format= — tell the agent to fill a Pydantic schema
  - result["structured_response"] — typed access to the validated output
  - Pydantic BaseModel — defines the exact shape of the agent's answer
  - Combining tools + structured output in one agent

Why this matters:
  Without structured output you get a raw string.
  With response_format= you get a Python object whose fields are guaranteed
  to exist and be of the correct type — safe to use in downstream code.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

load_dotenv()


# ── Pydantic schemas (the "response format") ─────────────────────────────────

class WeatherReport(BaseModel):
    """Structured weather report returned by the agent."""
    city: str                           = Field(description="City name")
    temperature_celsius: float          = Field(description="Temperature in °C")
    condition: str                      = Field(description="Weather condition, e.g. 'Sunny'")
    recommendation: str                 = Field(description="What to wear / bring")


class ResearchSummary(BaseModel):
    """Structured research summary."""
    topic: str                          = Field(description="The topic researched")
    key_points: List[str]               = Field(description="3-5 bullet point findings")
    confidence: float                   = Field(description="Confidence score 0.0–1.0")
    follow_up_questions: List[str]      = Field(description="Suggested next questions")
    sources_needed: Optional[str]       = Field(
        default=None,
        description="Types of sources that would improve this research"
    )


# ── Tools ────────────────────────────────────────────────────────────────────

@tool
def fetch_weather_data(city: str) -> str:
    """Fetch raw weather data for a city.

    Args:
        city: City name
    """
    raw_data = {
        "london":  "temp=15, condition=cloudy, humidity=80%",
        "tokyo":   "temp=28, condition=sunny, humidity=55%",
        "dubai":   "temp=41, condition=sunny, humidity=30%",
        "reykjavik": "temp=5, condition=windy, humidity=70%",
    }
    return raw_data.get(city.lower(), f"No data for {city}")


@tool
def search_knowledge_base(query: str) -> str:
    """Search internal knowledge base for information.

    Args:
        query: Search query string
    """
    # Simulated KB results
    return (
        f"Knowledge base results for '{query}': "
        "LangChain is a framework for building LLM applications. "
        "It supports chains, agents, tools, and retrieval-augmented generation. "
        "Key features include: composable chains, tool integration, memory, "
        "and structured output. Version 0.3+ uses create_agent() as the primary "
        "agent factory instead of older AgentExecutor patterns."
    )


# ── Agent 1: Weather with structured output ───────────────────────────────────

weather_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[fetch_weather_data],
    response_format=WeatherReport,    # ← structured output schema
    system_prompt=(
        "You are a weather assistant. Fetch weather data using the tool, "
        "then fill in the WeatherReport schema accurately."
    ),
)

# ── Agent 2: Research summariser with structured output ───────────────────────

research_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_knowledge_base],
    response_format=ResearchSummary,
    system_prompt=(
        "You are a research assistant. Search the knowledge base and produce "
        "a structured, accurate research summary."
    ),
)


if __name__ == "__main__":
    print("=" * 60)
    print("Structured Output Agent Demo")
    print("=" * 60)

    # ── Example 1: Weather report ─────────────────────────────────────────────
    print("\n📍 Example 1 — Structured Weather Report")
    result = weather_agent.invoke({
        "messages": [{"role": "user", "content": "What's the weather like in Tokyo?"}]
    })

    # Access structured response — a validated WeatherReport object
    report: WeatherReport = result["structured_response"]

    print(f"\nCity:            {report.city}")
    print(f"Temperature:     {report.temperature_celsius}°C")
    print(f"Condition:       {report.condition}")
    print(f"Recommendation:  {report.recommendation}")
    print(f"\nRaw type: {type(report).__name__}")   # WeatherReport — not a string!

    # ── Example 2: Research summary ───────────────────────────────────────────
    print("\n" + "─" * 60)
    print("\n📚 Example 2 — Structured Research Summary")
    result = research_agent.invoke({
        "messages": [{"role": "user", "content": "Tell me about LangChain agents"}]
    })

    summary: ResearchSummary = result["structured_response"]

    print(f"\nTopic:      {summary.topic}")
    print(f"Confidence: {summary.confidence:.0%}")
    print("\nKey Points:")
    for i, point in enumerate(summary.key_points, 1):
        print(f"  {i}. {point}")
    print("\nFollow-up Questions:")
    for q in summary.follow_up_questions:
        print(f"  • {q}")
    if summary.sources_needed:
        print(f"\nSources Needed: {summary.sources_needed}")
