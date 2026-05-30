"""
02_model_context_messages.py
==============================
Demonstrates how to engineer the MESSAGE LIST sent to the LLM using
@wrap_model_call — injecting additional context transiently (without
modifying persistent state) from State, Store, and Runtime Context.

Concepts covered:
  - @wrap_model_call hook signature and request.override()
  - Injecting file context from State (transient, per-call)
  - Injecting user writing style from Store (transient, per-call)
  - Injecting compliance rules from Runtime Context (transient)
  - Transient vs persistent message updates (key distinction)
  - Combining multiple injection middlewares in one agent
"""

import os
from dataclasses import dataclass, field
from typing import Callable
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool
from langgraph.store.memory import InMemoryStore

load_dotenv()

print("=" * 60)
print("Model Context — Message Injection (wrap_model_call)")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# TOOLS
# ════════════════════════════════════════════════════════════════════

@tool
def compose_email(to: str, subject: str) -> str:
    """Draft a professional email to a recipient."""
    return f"Draft email created for {to} with subject '{subject}'."


@tool
def search_documents(query: str) -> str:
    """Search through available documents."""
    return f"Found 3 documents matching '{query}': [Doc A, Doc B, Doc C]."


# ════════════════════════════════════════════════════════════════════
# 1. STATE-BASED MESSAGE INJECTION
#    Inject metadata about uploaded files stored in session state.
#    This is TRANSIENT — it adds messages for this LLM call only;
#    it does NOT persist them to the conversation history.
# ════════════════════════════════════════════════════════════════════

@wrap_model_call
def inject_file_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Inject context about files the user has uploaded this session."""
    uploaded_files = request.state.get("uploaded_files", [])

    if uploaded_files:
        descriptions = [
            f"  - {f['name']} ({f['type']}): {f['summary']}"
            for f in uploaded_files
        ]
        file_context = (
            "Files available in this conversation:\n"
            + "\n".join(descriptions)
            + "\nReference these files when answering."
        )
        print(f"  [inject_file_context] Injecting {len(uploaded_files)} file(s)")
        messages = [*request.messages, {"role": "user", "content": file_context}]
        request  = request.override(messages=messages)
    else:
        print("  [inject_file_context] No uploaded files in state")

    return handler(request)


print("\n── 1. State-Based Message Injection (uploaded files) ────────")

agent_files = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_documents],
    middleware=[inject_file_context],
    system_prompt="You are a document analysis assistant.",
)

# Without files
result_no_files = agent_files.invoke({
    "messages": [{"role": "user", "content": "Help me find the budget document."}]
})
print(f"No files: {result_no_files['messages'][-1].content[:100]}")

# With uploaded files in state
result_with_files = agent_files.invoke({
    "messages":       [{"role": "user", "content": "Summarize the uploaded documents."}],
    "uploaded_files": [
        {"name": "budget_q4.xlsx", "type": "spreadsheet",
         "summary": "Q4 budget breakdown by department"},
        {"name": "roadmap.pdf",    "type": "pdf",
         "summary": "2026 product roadmap with milestones"},
    ],
})
print(f"With files: {result_with_files['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# 2. STORE-BASED MESSAGE INJECTION
#    Inject the user's personal email writing style from long-term
#    memory so the LLM mimics their tone when drafting emails.
# ════════════════════════════════════════════════════════════════════

@dataclass
class EmailCtx:
    user_id: str


@wrap_model_call
def inject_writing_style(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Inject user's email writing style from Store."""
    user_id = request.runtime.context.user_id
    store   = request.runtime.store

    if store:
        style_mem = store.get(("writing_style",), user_id)
        if style_mem:
            style = style_mem.value
            style_guide = (
                "Your writing style guide:\n"
                f"  Tone:           {style.get('tone', 'professional')}\n"
                f"  Typical greeting: \"{style.get('greeting', 'Hi')}\"\n"
                f"  Typical sign-off: \"{style.get('sign_off', 'Best regards')}\"\n"
                f"  Example email:\n{style.get('example_email', '(none)')}"
            )
            print(f"  [inject_writing_style] Style loaded for user={user_id}")
            # Append at the END — models pay more attention to final messages
            messages = [*request.messages, {"role": "user", "content": style_guide}]
            request  = request.override(messages=messages)
        else:
            print(f"  [inject_writing_style] No style found for user={user_id}")

    return handler(request)


print("\n── 2. Store-Based Message Injection (writing style) ─────────")

store = InMemoryStore()
store.put(("writing_style",), "USR-1", {
    "tone":          "warm and friendly",
    "greeting":      "Hey there",
    "sign_off":      "Cheers",
    "example_email": "Hey team, just a quick reminder about our Friday sync — hope to see you all there! Cheers, Alex",
})

agent_style = create_agent(
    model="openai:gpt-4o-mini",
    tools=[compose_email],
    context_schema=EmailCtx,
    store=store,
    middleware=[inject_writing_style],
    system_prompt="You are an email drafting assistant.",
)

result_style = agent_style.invoke(
    {"messages": [{"role": "user", "content":
        "Draft an email to the team about the project update meeting next Monday."}]},
    context=EmailCtx(user_id="USR-1"),
)
print(f"Styled email: {result_style['messages'][-1].content[:200]}")


# ════════════════════════════════════════════════════════════════════
# 3. RUNTIME CONTEXT–BASED MESSAGE INJECTION
#    Inject compliance rules tailored to the user's jurisdiction
#    and industry — loaded from the invocation-time context.
# ════════════════════════════════════════════════════════════════════

@dataclass
class ComplianceCtx:
    user_jurisdiction:    str
    industry:             str
    compliance_frameworks: list[str]


@wrap_model_call
def inject_compliance_rules(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Inject jurisdiction-specific compliance rules from Runtime Context."""
    ctx       = request.runtime.context
    frameworks = ctx.compliance_frameworks

    rules = []
    if "GDPR" in frameworks:
        rules += [
            "- Must obtain explicit consent before processing personal data.",
            "- Users have the right to data deletion (right to be forgotten).",
        ]
    if "HIPAA" in frameworks:
        rules += [
            "- Cannot share patient health information without authorization.",
            "- Must use secure, encrypted communication channels.",
        ]
    if "CCPA" in frameworks:
        rules += [
            "- Must disclose data collection practices to California residents.",
            "- Users can opt out of sale of their personal information.",
        ]
    if ctx.industry == "finance":
        rules.append("- Cannot provide investment advice without explicit disclaimers.")

    if rules:
        compliance_msg = (
            f"Compliance requirements for {ctx.user_jurisdiction} "
            f"({', '.join(frameworks)}):\n" + "\n".join(rules)
        )
        print(f"  [inject_compliance_rules] {len(rules)} rules for {ctx.user_jurisdiction}")
        messages = [*request.messages, {"role": "user", "content": compliance_msg}]
        request  = request.override(messages=messages)
    else:
        print("  [inject_compliance_rules] No applicable rules")

    return handler(request)


print("\n── 3. Runtime Context–Based Injection (compliance rules) ────")

agent_compliance = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    context_schema=ComplianceCtx,
    middleware=[inject_compliance_rules],
    system_prompt="You are a business operations assistant.",
)

result_gdpr = agent_compliance.invoke(
    {"messages": [{"role": "user", "content":
        "How should we store customer email addresses for our newsletter?"}]},
    context=ComplianceCtx(
        user_jurisdiction="EU",
        industry="e-commerce",
        compliance_frameworks=["GDPR"],
    ),
)
print(f"GDPR: {result_gdpr['messages'][-1].content[:180]}")

result_hipaa = agent_compliance.invoke(
    {"messages": [{"role": "user", "content":
        "Can we send patient data to our analytics platform?"}]},
    context=ComplianceCtx(
        user_jurisdiction="USA",
        industry="healthcare",
        compliance_frameworks=["HIPAA"],
    ),
)
print(f"HIPAA: {result_hipaa['messages'][-1].content[:180]}")


# ════════════════════════════════════════════════════════════════════
# 4. STACKED INJECTIONS — file context + compliance in one agent
# ════════════════════════════════════════════════════════════════════

print("\n── 4. Stacked Injections (file context + compliance) ─────────")

agent_stacked = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_documents],
    context_schema=ComplianceCtx,
    middleware=[inject_file_context, inject_compliance_rules],  # Both run in order
    system_prompt="You are a compliance-aware document assistant.",
)

result_stacked = agent_stacked.invoke(
    {
        "messages":       [{"role": "user", "content":
                           "Search for customer data policy documents."}],
        "uploaded_files": [{"name": "policy_draft.docx", "type": "docx",
                           "summary": "Draft data retention policy v2.1"}],
    },
    context=ComplianceCtx(
        user_jurisdiction="EU",
        industry="fintech",
        compliance_frameworks=["GDPR"],
    ),
)
print(f"Stacked: {result_stacked['messages'][-1].content[:180]}")

print("\n" + "═" * 60)
print("Transient vs Persistent — Key Distinction:")
print("  wrap_model_call → TRANSIENT: modifies what the LLM sees")
print("  for ONE call only. Does NOT update stored state.")
print("  before_model/after_model → PERSISTENT: can update state")
print("  using Command, saved for all future turns.")
print("═" * 60)
print("\n✅ Message injection demo complete.")
