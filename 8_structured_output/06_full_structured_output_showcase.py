"""
06_full_structured_output_showcase.py
======================================
A production-grade case study demonstrating a support ticket classification and
triage agent that uses tools and returns structured output.

Concepts covered:
  - Structured output schemas with validation rules
  - Using tools AND structured output in the same agent
  - Union types or complex schemas in response_format
  - Complete execution flow, inputs, outputs, and validation
"""

import os
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain.agents.structured_output import ToolStrategy

load_dotenv()

# ── 1. Tools ──────────────────────────────────────────────────────────

@tool
def lookup_customer_order(order_id: str) -> str:
    """Lookup order shipping status and history.

    Args:
        order_id: The order identifier (e.g. 'ORD-98765')
    """
    orders = {
        "ord-98765": "📦 Order ORD-98765: Shipped via FedEx on 2026-05-27. Delayed due to transit storms in Chicago.",
        "ord-11223": "📦 Order ORD-11223: Delivered to front porch on 2026-05-28. Signed by 'V. Pattar'.",
    }
    return orders.get(order_id.lower(), f"❌ Order {order_id} not found in database.")


# ── 2. Define Structured Response Schema ──────────────────────────────

class TicketTriage(BaseModel):
    """Analysis and structured triage of an incoming support ticket."""
    order_id: Optional[str] = Field(None, description="The order ID found in the ticket (e.g. ORD-xxxxx)")
    category: Literal["shipping", "billing", "technical", "refund", "general"] = Field(
        description="The primary category of the user's issue"
    )
    priority: Literal["low", "medium", "high", "critical"] = Field(
        description="Priority of the issue based on customer sentiment and urgency"
    )
    sentiment: Literal["frustrated", "neutral", "satisfied"] = Field(
        description="Detected emotional state of the customer"
    )
    summary: str = Field(description="A concise 1-sentence summary of the core complaint")
    action_items: List[str] = Field(description="Concrete next actions for the support agent to take")


print("=" * 60)
print("Production Triage & Structured Output Showcase")
print("=" * 60)

# Configure ToolStrategy with custom message and default error handling
response_strategy = ToolStrategy(
    schema=TicketTriage,
    tool_message_content="📊 Support ticket successfully triaged and added to Zendesk queue."
)

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[lookup_customer_order],
    response_format=response_strategy,
    system_prompt=(
        "You are an automated support triage agent. Look up orders if an order ID "
        "is present in the ticket, then fill out the TicketTriage schema completely and accurately."
    )
)

TICKET_TEXT = (
    "Subject: Where is my order?! "
    "Body: Hi, I ordered a mechanical keyboard last week (Order ID: ORD-98765) and "
    "it was supposed to arrive yesterday. It still hasn't arrived! I need this "
    "keyboard for my coding job. Please refund my shipping or tell me where it is!"
)

print(f"Incoming Ticket:\n  {TICKET_TEXT}\n")
print("Processing ticket and invoking agent...")

result = agent.invoke({
    "messages": [{"role": "user", "content": TICKET_TEXT}]
})

print("\n── 1. History Steps ──")
for msg in result["messages"]:
    msg_type = type(msg).__name__
    if msg_type == "ToolMessage":
        print(f"  [{msg_type}]: {msg.content}")
    elif msg_type == "AIMessage" and msg.tool_calls:
        print(f"  [{msg_type}]: Requests tool {msg.tool_calls[0]['name']}({msg.tool_calls[0]['args']})")

print("\n── 2. Extracted Structured Response ──")
triage: TicketTriage = result["structured_response"]

print(f"  Order ID:     {triage.order_id}")
print(f"  Category:     {triage.category}")
print(f"  Priority:     {triage.priority}")
print(f"  Sentiment:    {triage.sentiment}")
print(f"  Summary:      {triage.summary}")
print(f"  Action Items:")
for item in triage.action_items:
    print(f"    - {item}")

print("=" * 60)
