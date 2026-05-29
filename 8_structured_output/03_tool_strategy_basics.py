"""
03_tool_strategy_basics.py
==========================
Demonstrates explicit use of ToolStrategy for schema extraction using model
tool-calling capabilities.

Concepts covered:
  - Explicit ToolStrategy configuration
  - Using ToolStrategy with Pydantic BaseModel schemas
  - Tool-calling based extraction under the hood
  - Accessing the validated final result
"""

import os
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

load_dotenv()

print("=" * 60)
print("ToolStrategy (Tool-Calling Structured Output) Demo")
print("=" * 60)

TEXT_TO_PARSE = "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"
print(f"Input text: '{TEXT_TO_PARSE}'\n")


# ════════════════════════════════════════════════════════════════════
# 1. DEFINE SCHEMA AND TOOLSTRATEGY
# ════════════════════════════════════════════════════════════════════

class ProductReview(BaseModel):
    """Analysis of a product review."""
    rating: int = Field(description="The rating of the product from 1 to 5", ge=1, le=5)
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the review")
    key_points: list[str] = Field(description="The key points of the review. Lowercase, 1-3 words each.")

# Explicitly configure a ToolStrategy
# ToolStrategy forces the agent to use tool calling to extract the schema
tool_strategy = ToolStrategy(
    schema=ProductReview
)

agent = create_agent(
    model="openai:gpt-4o-mini",
    response_format=tool_strategy,
    system_prompt="You are a helpful product review parser."
)

print("Invoking agent...")
result = agent.invoke({
    "messages": [{"role": "user", "content": TEXT_TO_PARSE}]
})

structured_res = result["structured_response"]

print(f"\nResult Class: {type(structured_res).__name__}")
print(f"Parsed Review:")
print(f"  Rating:     {structured_res.rating}/5")
print(f"  Sentiment:  {structured_res.sentiment}")
print(f"  Key Points: {structured_res.key_points}\n")
