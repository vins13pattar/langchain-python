"""
01_auto_strategy.py
===================
Demonstrates AUTO-STRATEGY selection in create_agent() by passing a schema
type directly to response_format.

Concepts covered:
  - Direct schema passing (response_format=ContactInfo)
  - Auto-selection of ProviderStrategy vs ToolStrategy
  - Pydantic BaseModel structure
  - Python standard dataclass structure
  - TypedDict structure
  - Accessing the final parsed data in result["structured_response"]
"""

import os
from dataclasses import dataclass
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain.agents import create_agent

load_dotenv()

print("=" * 60)
print("Response Format Auto-Strategy Demo")
print("=" * 60)

TEXT_TO_PARSE = "Extract info from: John Doe, john@example.com, (555) 123-4567"
print(f"Input text: '{TEXT_TO_PARSE}'\n")


# ════════════════════════════════════════════════════════════════════
# 1. AUTO-STRATEGY WITH PYDANTIC BASEMODEL
# ════════════════════════════════════════════════════════════════════

print("── 1. Using Pydantic BaseModel ─────────────────────────────")

class ContactPydantic(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The full name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")

agent_pydantic = create_agent(
    model="openai:gpt-4o-mini",
    response_format=ContactPydantic,  # Auto-selects ProviderStrategy
    system_prompt="You extract contact information."
)

result_pydantic = agent_pydantic.invoke({
    "messages": [{"role": "user", "content": TEXT_TO_PARSE}]
})

structured_res = result_pydantic["structured_response"]
print(f"Result Class: {type(structured_res).__name__}")
print(f"Parsed Pydantic Data:")
print(f"  Name:  {structured_res.name}")
print(f"  Email: {structured_res.email}")
print(f"  Phone: {structured_res.phone}\n")


# ════════════════════════════════════════════════════════════════════
# 2. AUTO-STRATEGY WITH DATACLASS
# ════════════════════════════════════════════════════════════════════

print("─" * 60)
print("2. Using Python standard dataclass")
print("─" * 60)

@dataclass
class ContactDataclass:
    """Contact information for a person."""
    name: str  # The full name
    email: str  # The email address
    phone: str  # The phone number

agent_dataclass = create_agent(
    model="openai:gpt-4o-mini",
    response_format=ContactDataclass,  # Auto-selects ProviderStrategy
    system_prompt="You extract contact information."
)

result_dataclass = agent_dataclass.invoke({
    "messages": [{"role": "user", "content": TEXT_TO_PARSE}]
})

structured_res_dc = result_dataclass["structured_response"]
print(f"Result Class: {type(structured_res_dc).__name__}")
print(f"Parsed Dataclass Data:")
print(f"  Name:  {structured_res_dc.get('name') if isinstance(structured_res_dc, dict) else getattr(structured_res_dc, 'name', '')}")
print(f"  Email: {structured_res_dc.get('email') if isinstance(structured_res_dc, dict) else getattr(structured_res_dc, 'email', '')}")
print(f"  Phone: {structured_res_dc.get('phone') if isinstance(structured_res_dc, dict) else getattr(structured_res_dc, 'phone', '')}\n")


# ════════════════════════════════════════════════════════════════════
# 3. AUTO-STRATEGY WITH TYPEDDICT
# ════════════════════════════════════════════════════════════════════

print("─" * 60)
print("3. Using TypedDict")
print("─" * 60)

class ContactTypedDict(TypedDict):
    """Contact information for a person."""
    name: str
    email: str
    phone: str

agent_typeddict = create_agent(
    model="openai:gpt-4o-mini",
    response_format=ContactTypedDict,  # Auto-selects ProviderStrategy
    system_prompt="You extract contact information."
)

result_typeddict = agent_typeddict.invoke({
    "messages": [{"role": "user", "content": TEXT_TO_PARSE}]
})

structured_res_td = result_typeddict["structured_response"]
print(f"Result Class: {type(structured_res_td).__name__}")
print(f"Parsed TypedDict Data:")
print(f"  Name:  {structured_res_td.get('name')}")
print(f"  Email: {structured_res_td.get('email')}")
print(f"  Phone: {structured_res_td.get('phone')}\n")
