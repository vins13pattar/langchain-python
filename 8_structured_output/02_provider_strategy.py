"""
02_provider_strategy.py
=======================
Demonstrates explicit use of ProviderStrategy for native, provider-enforced
structured outputs.

Concepts covered:
  - Explicit ProviderStrategy configuration
  - Enabling 'strict' schema adherence (supported by OpenAI/xAI)
  - Native extraction using JSON Schema dictionary formats
  - Accessing the validated result
"""

import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy

load_dotenv()

print("=" * 60)
print("ProviderStrategy (Native Structured Output) Demo")
print("=" * 60)

TEXT_TO_PARSE = "Extract contact: John Doe, john@example.com, (555) 123-4567"


# ════════════════════════════════════════════════════════════════════
# 1. PROVIDERSTRATEGY WITH strict=True
# ════════════════════════════════════════════════════════════════════

print("── 1. Using ProviderStrategy with strict=True ───────────────")

class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")

# Configure ProviderStrategy with strict schema adherence
provider_strategy = ProviderStrategy(
    schema=ContactInfo,
    strict=True   # Ensures strict validation if supported (e.g. OpenAI Structured Outputs)
)

agent_strict = create_agent(
    model="openai:gpt-4o-mini",
    response_format=provider_strategy,
    system_prompt="You extract contact information."
)

result_strict = agent_strict.invoke({
    "messages": [{"role": "user", "content": TEXT_TO_PARSE}]
})

structured_res = result_strict["structured_response"]
print(f"Result Class: {type(structured_res).__name__}")
print(f"Parsed Data:")
print(f"  Name:  {structured_res.name}")
print(f"  Email: {structured_res.email}")
print(f"  Phone: {structured_res.phone}\n")


# ════════════════════════════════════════════════════════════════════
# 2. PROVIDERSTRATEGY WITH RAW JSON SCHEMA
# ════════════════════════════════════════════════════════════════════

print("─" * 60)
print("2. Using ProviderStrategy with raw JSON Schema")
print("─" * 60)

# You can also pass a raw JSON Schema directly as the target representation
contact_info_json_schema = {
    "title": "ContactInfo",
    "type": "object",
    "description": "Contact information for a person.",
    "properties": {
        "name": {"type": "string", "description": "The name of the person"},
        "email": {"type": "string", "description": "The email address of the person"},
        "phone": {"type": "string", "description": "The phone number of the person"}
    },
    "required": ["name", "email", "phone"]
}

provider_strategy_json = ProviderStrategy(
    schema=contact_info_json_schema
)

agent_json = create_agent(
    model="openai:gpt-4o-mini",
    response_format=provider_strategy_json,
    system_prompt="You extract contact information."
)

result_json = agent_json.invoke({
    "messages": [{"role": "user", "content": TEXT_TO_PARSE}]
})

structured_res_json = result_json["structured_response"]
print(f"Result Class: {type(structured_res_json).__name__}")
print(f"Parsed JSON Schema Data:")
print(f"  Name:  {structured_res_json.get('name')}")
print(f"  Email: {structured_res_json.get('email')}")
print(f"  Phone: {structured_res_json.get('phone')}\n")
