"""
structured_output_overview.py — LangChain Structured Output: all key concepts in one file
Covers: auto strategy (Pydantic/dataclass/TypedDict), ProviderStrategy, ToolStrategy,
        error handling/retries, tools + structured output together
"""

from dataclasses import dataclass
from typing import List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy

load_dotenv()

def section(title): print(f"\n{'─'*55}\n{title}\n{'─'*55}")

TEXT = "Extract info from: John Doe, john@example.com, (555) 123-4567"


# ════════════════════════════════════════════════════════════════════
# 1. AUTO STRATEGY — pass schema directly, LangChain picks the best method
# ════════════════════════════════════════════════════════════════════
section("1. AUTO STRATEGY")

class ContactPydantic(BaseModel):
    """Contact information."""
    name:  str = Field(description="Full name")
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number")

@dataclass
class ContactDataclass:
    name:  str
    email: str
    phone: str

class ContactTypedDict(TypedDict):
    name:  str
    email: str
    phone: str

for schema, label in [
    (ContactPydantic, "Pydantic"),
    (ContactDataclass, "Dataclass"),
    (ContactTypedDict, "TypedDict"),
]:
    agent = create_agent(
        model="openai:gpt-4o-mini",
        response_format=schema,  # auto-selects strategy
        system_prompt="You extract contact information.",
    )
    r = agent.invoke({"messages": [{"role": "user", "content": TEXT}]})
    result = r["structured_response"]
    if hasattr(result, "name"):
        name = result.name
    else:
        name = result.get("name") if isinstance(result, dict) else "?"
    print(f"{label}: name={name!r}  type={type(result).__name__}")


# ════════════════════════════════════════════════════════════════════
# 2. PROVIDER STRATEGY — native JSON mode / strict schema
# ════════════════════════════════════════════════════════════════════
section("2. PROVIDER STRATEGY")

class ContactInfo(BaseModel):
    name:  str = Field(description="Full name")
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number")

# With strict=True (OpenAI Structured Outputs)
agent_strict = create_agent(
    model="openai:gpt-4o-mini",
    response_format=ProviderStrategy(schema=ContactInfo, strict=True),
    system_prompt="You extract contact information.",
)
r = agent_strict.invoke({"messages": [{"role": "user", "content": TEXT}]})
res = r["structured_response"]
print(f"Strict: name={res.name}  email={res.email}  phone={res.phone}")

# With raw JSON Schema dict
json_schema = {
    "title": "Contact", "type": "object",
    "properties": {
        "name":  {"type": "string", "description": "Full name"},
        "email": {"type": "string", "description": "Email address"},
        "phone": {"type": "string", "description": "Phone number"},
    },
    "required": ["name", "email", "phone"],
}
agent_json = create_agent(
    model="openai:gpt-4o-mini",
    response_format=ProviderStrategy(schema=json_schema),
    system_prompt="You extract contact information.",
)
r = agent_json.invoke({"messages": [{"role": "user", "content": TEXT}]})
res = r["structured_response"]
print(f"JSON schema: name={res.get('name')}  email={res.get('email')}")


# ════════════════════════════════════════════════════════════════════
# 3. TOOL STRATEGY — structured output via tool calling
# ════════════════════════════════════════════════════════════════════
section("3. TOOL STRATEGY")

class PersonProfile(BaseModel):
    """Extracted person profile."""
    full_name:  str             = Field(description="Full name")
    occupation: str             = Field(description="Job title or role")
    skills:     List[str]       = Field(description="Technical skills (list)")
    experience_years: int       = Field(description="Years of professional experience")
    location:   Optional[str]   = Field(None, description="City or country if mentioned")

agent_tool = create_agent(
    model="openai:gpt-4o-mini",
    response_format=ToolStrategy(
        schema=PersonProfile,
        tool_message_content="Profile extraction complete.",  # custom ToolMessage
    ),
    system_prompt="Extract all person profile information from the text.",
)

bio = "Meet Ananya, a senior Python developer from Bengaluru with 6 years of experience specialising in FastAPI, AWS, and ML pipelines."
r = agent_tool.invoke({"messages": [{"role": "user", "content": bio}]})
profile: PersonProfile = r["structured_response"]
print(f"Name: {profile.full_name}  Occupation: {profile.occupation}  Exp: {profile.experience_years}y")
print(f"Skills: {profile.skills}  Location: {profile.location}")


# ════════════════════════════════════════════════════════════════════
# 4. ERROR HANDLING & RETRIES
# ════════════════════════════════════════════════════════════════════
section("4. ERROR HANDLING & RETRIES")

class ProductReview(BaseModel):
    """Structured product review."""
    rating:    int            = Field(description="Rating 1-5", ge=1, le=5)
    sentiment: Literal["positive", "neutral", "negative"] = Field(description="Overall sentiment")
    pros:      List[str]      = Field(description="List of positive points")
    cons:      List[str]      = Field(description="List of negative points")
    summary:   str            = Field(description="One-sentence summary")

agent_review = create_agent(
    model="openai:gpt-4o-mini",
    response_format=ToolStrategy(schema=ProductReview, max_attempts=3),
    system_prompt="Extract a structured product review from the customer feedback.",
)

feedback = "This laptop is amazing! The battery lasts all day and the screen is gorgeous. However the keyboard feels a bit mushy and it heats up during video calls. Overall 4 stars."
r = agent_review.invoke({"messages": [{"role": "user", "content": feedback}]})
review: ProductReview = r["structured_response"]
print(f"Rating: {review.rating}/5  Sentiment: {review.sentiment}")
print(f"Pros: {review.pros}")
print(f"Cons: {review.cons}")
print(f"Summary: {review.summary}")


# ════════════════════════════════════════════════════════════════════
# 5. TOOLS + STRUCTURED OUTPUT — agent uses tools THEN fills schema
# ════════════════════════════════════════════════════════════════════
section("5. TOOLS + STRUCTURED OUTPUT")

@tool
def lookup_order(order_id: str) -> str:
    """Look up order status. Args: order_id: Order ID e.g. ORD-98765."""
    orders = {
        "ord-98765": "Shipped via FedEx on 2026-05-27. Delayed due to transit storms.",
        "ord-11223": "Delivered on 2026-05-28. Signed by 'V. Pattar'.",
    }
    return orders.get(order_id.lower(), f"Order {order_id} not found.")

class TicketTriage(BaseModel):
    """Support ticket triage result."""
    order_id:     Optional[str]                                              = Field(None, description="Order ID if mentioned")
    category:     Literal["shipping","billing","technical","refund","general"] = Field(description="Issue category")
    priority:     Literal["low","medium","high","critical"]                  = Field(description="Priority level")
    sentiment:    Literal["frustrated","neutral","satisfied"]                = Field(description="Customer sentiment")
    summary:      str                                                        = Field(description="One-sentence summary")
    action_items: List[str]                                                  = Field(description="Next steps for support")

triage_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[lookup_order],
    response_format=ToolStrategy(schema=TicketTriage, tool_message_content="Ticket triaged successfully."),
    system_prompt="Triage support tickets. Look up order info if an order ID is present.",
)

ticket = "Hi, I ordered a keyboard last week (ORD-98765) and it hasn't arrived! I need it for work. Please help or refund!"
r = triage_agent.invoke({"messages": [{"role": "user", "content": ticket}]})
triage: TicketTriage = r["structured_response"]
print(f"Order: {triage.order_id}  Category: {triage.category}  Priority: {triage.priority}")
print(f"Sentiment: {triage.sentiment}  Summary: {triage.summary}")
for item in triage.action_items:
    print(f"  • {item}")

print("""
Strategy guide:
  Auto (schema directly)  → LangChain picks best method automatically
  ProviderStrategy        → Native JSON mode, strict=True for OpenAI Structured Outputs
  ToolStrategy            → Works on all providers, supports tool use + custom error msgs
  
  result["structured_response"] → always the parsed, typed object
""")
