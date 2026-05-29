"""
04_pii_detection_and_guardrails.py
===================================
Demonstrates middleware used as guardrails — blocking, redacting, or
flagging dangerous content before it reaches tools or the model.

Concepts covered:
  - PIIDetectionMiddleware (built-in) — Redact PII from prompts/responses
  - Custom content guardrail          — Block prompts matching forbidden patterns
  - Input validation middleware        — Reject malformed tool arguments early
  - Output sanitization middleware     — Clean up model responses before returning
"""

import re
import os
from typing import Any, Optional
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import BaseMiddleware, PIIDetectionMiddleware
from langchain.tools import tool

load_dotenv()

print("=" * 60)
print("PII Detection & Guardrails Middleware Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# TOOLS
# ════════════════════════════════════════════════════════════════════

@tool
def store_record(name: str, email: str, phone: str) -> str:
    """Store a customer record to the database."""
    print(f"  [Tool] store_record: name={name}, email={email}, phone={phone}")
    return f"Record stored for {name}."


@tool
def generate_report(topic: str) -> str:
    """Generate a detailed report on a given topic."""
    return f"Detailed report on '{topic}': [Report content here]"


# ════════════════════════════════════════════════════════════════════
# 1. BUILT-IN PII DETECTION MIDDLEWARE
#    Automatically detects and redacts PII (emails, phones, SSNs, etc.)
#    from messages before they are sent to the model and from tool inputs.
# ════════════════════════════════════════════════════════════════════

print("\n── 1. PIIDetectionMiddleware (Built-in) ─────────────────────")

agent_pii = create_agent(
    model="openai:gpt-4o-mini",
    tools=[store_record],
    middleware=[
        PIIDetectionMiddleware(
            redact=True,           # Replace PII with [REDACTED]
            raise_on_detect=False, # Don't raise — just redact silently
        )
    ],
    system_prompt="You are a customer data assistant.",
)

result_pii = agent_pii.invoke({
    "messages": [{"role": "user", "content":
        "Store a record for Jane Doe, email jane.doe@secret.com, phone 555-987-6543."}]
})
print(f"Response: {result_pii['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# 2. CUSTOM CONTENT GUARDRAIL
#    Block the agent from running if the user's prompt matches a
#    forbidden topic (e.g. competitor names, abusive keywords).
# ════════════════════════════════════════════════════════════════════

class ContentGuardrailMiddleware(BaseMiddleware):
    """Blocks agent execution if the input contains forbidden content."""

    FORBIDDEN_PATTERNS = [
        r"\b(competitor|rival|hack|bypass|jailbreak)\b",
    ]

    def before_agent(self, state: dict) -> Optional[dict]:
        messages = state.get("messages", [])
        for msg in messages:
            content = getattr(msg, "content", "") or ""
            for pattern in self.FORBIDDEN_PATTERNS:
                if re.search(pattern, content, re.IGNORECASE):
                    print(f"  [Guardrail] 🚫 Blocked — forbidden pattern detected: '{pattern}'")
                    # Inject a refusal message into state instead of running the agent
                    from langchain_core.messages import AIMessage
                    return {
                        **state,
                        "messages": messages + [
                            AIMessage(
                                content="I'm sorry, I cannot help with that request."
                            )
                        ],
                    }
        return None  # Allow the agent to proceed normally


print("\n── 2. ContentGuardrailMiddleware (Custom) ───────────────────")

agent_guardrail = create_agent(
    model="openai:gpt-4o-mini",
    tools=[generate_report],
    middleware=[ContentGuardrailMiddleware()],
    system_prompt="You are a business intelligence assistant.",
)

# Safe request — passes
result_safe = agent_guardrail.invoke({
    "messages": [{"role": "user", "content": "Generate a report on global AI trends."}]
})
print(f"✅ Safe request: {result_safe['messages'][-1].content[:100]}")

# Forbidden request — blocked
result_blocked = agent_guardrail.invoke({
    "messages": [{"role": "user", "content": "Generate a report to hack into competitor systems."}]
})
print(f"🚫 Blocked request: {result_blocked['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# 3. INPUT VALIDATION MIDDLEWARE
#    Validate tool call arguments BEFORE executing the tool.
#    If validation fails, return an error message to the agent to retry.
# ════════════════════════════════════════════════════════════════════

class InputValidationMiddleware(BaseMiddleware):
    """Validates tool inputs against business rules before execution."""

    RULES = {
        "store_record": lambda args: (
            "@" in args.get("email", "") and
            len(args.get("name", "")) >= 2
        )
    }

    def wrap_tool_call(self, tool_call: dict, call_tool, **kwargs) -> Any:
        name  = tool_call.get("name", "")
        args  = tool_call.get("args", {})
        rule  = self.RULES.get(name)

        if rule and not rule(args):
            print(f"  [Validation] ⚠️  Invalid args for '{name}': {args}")
            return (
                f"Error: Invalid arguments for {name}. "
                "Please provide a valid name (≥ 2 chars) and email (must contain @)."
            )

        print(f"  [Validation] ✅ Args valid for '{name}'")
        return call_tool(tool_call, **kwargs)


print("\n── 3. InputValidationMiddleware (Custom) ────────────────────")

agent_validated = create_agent(
    model="openai:gpt-4o-mini",
    tools=[store_record],
    middleware=[InputValidationMiddleware()],
    system_prompt="You are a customer data assistant.",
)

# Valid record
result_valid = agent_validated.invoke({
    "messages": [{"role": "user", "content":
        "Store record: name=Alice Smith, email=alice@example.com, phone=555-0001"}]
})
print(f"✅ Valid record: {result_valid['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# 4. OUTPUT SANITIZATION MIDDLEWARE
#    Scrub sensitive patterns from the final agent response before
#    delivering it to the caller.
# ════════════════════════════════════════════════════════════════════

class OutputSanitizationMiddleware(BaseMiddleware):
    """Redacts API keys and secrets from the agent's final response."""

    SECRET_PATTERNS = [
        (r"sk-[A-Za-z0-9]{20,}", "[REDACTED_API_KEY]"),
        (r"\b[A-Z0-9]{32}\b", "[REDACTED_TOKEN]"),
    ]

    def after_agent(self, state: dict) -> Optional[dict]:
        messages = state.get("messages", [])
        if not messages:
            return None
        last = messages[-1]
        content = getattr(last, "content", "") or ""
        for pattern, replacement in self.SECRET_PATTERNS:
            content = re.sub(pattern, replacement, content)
        # Return updated state with sanitized message
        sanitized = last.copy(update={"content": content})
        return {**state, "messages": messages[:-1] + [sanitized]}


print("\n── 4. OutputSanitizationMiddleware (Custom) ─────────────────")

@tool
def get_config() -> str:
    """Retrieve system configuration including API credentials."""
    return "Config loaded. API Key: sk-AbCdEfGhIjKlMnOpQrStUv123456"


agent_sanitized = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_config],
    middleware=[OutputSanitizationMiddleware()],
    system_prompt="You are a system configuration assistant.",
)

result_sanitized = agent_sanitized.invoke({
    "messages": [{"role": "user", "content": "Show me the current system configuration."}]
})
print(f"🔒 Sanitized response: {result_sanitized['messages'][-1].content[:150]}")

print("\n✅ PII & Guardrails demo complete.")
