"""
01_pii_middleware.py
====================
Demonstrates the PIIMiddleware built-in guardrail with all four PII handling
strategies: redact, mask, hash, and block.

Concepts covered:
  - PIIMiddleware with strategy="redact"  — Replace PII with [REDACTED_TYPE]
  - PIIMiddleware with strategy="mask"    — Partially obscure (e.g. ****-1234)
  - PIIMiddleware with strategy="hash"    — Deterministic hash replacement
  - PIIMiddleware with strategy="block"   — Raise exception on detection
  - Custom regex detector (api_key pattern)
  - apply_to_input / apply_to_output / apply_to_tool_results flags
  - Stacking multiple PIIMiddleware instances for different PII types
"""

import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.tools import tool

load_dotenv()

print("=" * 60)
print("PIIMiddleware — All Strategies Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# SHARED TOOL
# ════════════════════════════════════════════════════════════════════

@tool
def customer_service_tool(query: str) -> str:
    """Process a customer service query and return a response."""
    return f"Customer query processed: {query}"


@tool
def email_tool(recipient: str, message: str) -> str:
    """Send an email message to a recipient."""
    print(f"  [Tool] email_tool: to={recipient}, msg='{message[:50]}'")
    return f"Email sent to {recipient}."


# ════════════════════════════════════════════════════════════════════
# 1. REDACT STRATEGY — Replace PII with [REDACTED_{PII_TYPE}]
# ════════════════════════════════════════════════════════════════════

print("\n── 1. strategy='redact' (email) ─────────────────────────────")

agent_redact = create_agent(
    model="openai:gpt-4o-mini",
    tools=[customer_service_tool],
    middleware=[
        PIIMiddleware(
            "email",
            strategy="redact",       # → [REDACTED_EMAIL]
            apply_to_input=True,     # Check user messages before model call
            apply_to_output=False,   # Don't check model responses
        )
    ],
    system_prompt="You are a customer support assistant.",
)

result_redact = agent_redact.invoke({
    "messages": [{"role": "user", "content":
        "My email is john.doe@example.com — please process my support request."}]
})
print(f"Response: {result_redact['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# 2. MASK STRATEGY — Partially obscure (last 4 digits visible)
# ════════════════════════════════════════════════════════════════════

print("\n── 2. strategy='mask' (credit_card) ────────────────────────")

agent_mask = create_agent(
    model="openai:gpt-4o-mini",
    tools=[customer_service_tool],
    middleware=[
        PIIMiddleware(
            "credit_card",
            strategy="mask",         # → ****-****-****-5100
            apply_to_input=True,
        )
    ],
    system_prompt="You are a payment support assistant.",
)

result_mask = agent_mask.invoke({
    "messages": [{"role": "user", "content":
        "I have a problem with my card 5105-1051-0510-5100. Please help."}]
})
print(f"Response: {result_mask['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# 3. HASH STRATEGY — Replace with deterministic hash
# ════════════════════════════════════════════════════════════════════

print("\n── 3. strategy='hash' (ip address) ─────────────────────────")

agent_hash = create_agent(
    model="openai:gpt-4o-mini",
    tools=[customer_service_tool],
    middleware=[
        PIIMiddleware(
            "ip",
            strategy="hash",         # → a8f5f167f44f4964e6c998dee827110c
            apply_to_input=True,
        )
    ],
    system_prompt="You are a network security assistant.",
)

result_hash = agent_hash.invoke({
    "messages": [{"role": "user", "content":
        "My device IP is 192.168.1.42 — is it flagged in your system?"}]
})
print(f"Response: {result_hash['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# 4. BLOCK STRATEGY — Raise an exception immediately
# ════════════════════════════════════════════════════════════════════

print("\n── 4. strategy='block' (api_key — custom regex) ─────────────")

agent_block = create_agent(
    model="openai:gpt-4o-mini",
    tools=[customer_service_tool],
    middleware=[
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",  # Custom regex for OpenAI-style keys
            strategy="block",                  # Raise immediately
            apply_to_input=True,
        )
    ],
    system_prompt="You are a security assistant.",
)

try:
    result_block = agent_block.invoke({
        "messages": [{"role": "user", "content":
            "My API key is sk-AbCdEfGhIjKlMnOpQrStUvWxYz123456 — can you use it?"}]
    })
    print(f"Response: {result_block['messages'][-1].content[:80]}")
except Exception as e:
    print(f"🚫 Blocked — exception raised: {type(e).__name__}: {e}")


# ════════════════════════════════════════════════════════════════════
# 5. APPLY TO OUTPUT — Redact PII from model responses too
# ════════════════════════════════════════════════════════════════════

print("\n── 5. apply_to_output=True (redact from model responses) ────")

@tool
def get_customer_profile(customer_id: str) -> str:
    """Retrieve full customer profile including contact info."""
    return (
        f"Customer {customer_id}: Jane Smith, "
        "email: jane.smith@private.com, IP: 10.0.0.5, "
        "card: 4111-1111-1111-1111"
    )

agent_output = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_customer_profile],
    middleware=[
        PIIMiddleware("email",       strategy="redact", apply_to_input=True, apply_to_output=True),
        PIIMiddleware("credit_card", strategy="mask",   apply_to_input=True, apply_to_output=True),
        PIIMiddleware("ip",          strategy="redact", apply_to_input=True, apply_to_output=True),
    ],
    system_prompt="You are a customer lookup assistant. Return the profile as-is.",
)

result_output = agent_output.invoke({
    "messages": [{"role": "user", "content": "Get the profile for customer C-1001."}]
})
print(f"Response (PII redacted from output): {result_output['messages'][-1].content[:200]}")


# ════════════════════════════════════════════════════════════════════
# 6. STACKING MULTIPLE PIIMiddleware — Different types, different strategies
# ════════════════════════════════════════════════════════════════════

print("\n── 6. Stacking Multiple PIIMiddleware Instances ─────────────")

agent_stacked = create_agent(
    model="openai:gpt-4o-mini",
    tools=[customer_service_tool, email_tool],
    middleware=[
        PIIMiddleware("email",       strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask",   apply_to_input=True),
        PIIMiddleware("ip",          strategy="hash",   apply_to_input=True),
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{20,}",
            strategy="block",
            apply_to_input=True,
        ),
    ],
    system_prompt="You are a secure customer service agent.",
)

result_stacked = agent_stacked.invoke({
    "messages": [{"role": "user", "content":
        "Hello! My email is alice@example.com and I'm connecting from 203.0.113.5. "
        "Help me with my account."}]
})
print(f"Response: {result_stacked['messages'][-1].content[:120]}")

print("\n" + "═" * 60)
print("PIIMiddleware Strategy Reference:")
print("  redact  → [REDACTED_EMAIL], [REDACTED_CREDIT_CARD], etc.")
print("  mask    → ****-****-****-1234 (last segment visible)")
print("  hash    → deterministic SHA hash string")
print("  block   → raises exception immediately on detection")
print("═" * 60)
print("\n✅ PIIMiddleware demo complete.")
