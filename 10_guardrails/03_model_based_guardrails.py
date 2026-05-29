"""
03_model_based_guardrails.py
=============================
Demonstrates MODEL-BASED guardrails using after_agent hooks — leveraging
a secondary LLM to evaluate content that rule-based checks cannot catch.

Concepts covered:
  - AgentMiddleware with after_agent hook (class syntax)
  - @after_agent decorator (function syntax)
  - hook_config(can_jump_to=["end"]) for output replacement
  - LLM-as-judge safety classifier (SAFE / UNSAFE verdict)
  - Topic relevance guardrail (off-topic response filter)
  - Confidence/quality checker using a secondary model
  - Cost vs. accuracy trade-off notes for model-based checks
"""

import os
from typing import Any
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    hook_config,
    after_agent,
)
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage
from langchain.tools import tool
from langgraph.runtime import Runtime

load_dotenv()

print("=" * 60)
print("Model-Based Guardrails Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# SHARED TOOLS
# ════════════════════════════════════════════════════════════════════

@tool
def web_search(query: str) -> str:
    """Search the web for information on a topic."""
    return f"Search results for '{query}': [Sample web content here]"


@tool
def get_fact(topic: str) -> str:
    """Return an interesting fact about a topic."""
    facts = {
        "python": "Python was created by Guido van Rossum and first released in 1991.",
        "ai": "The term 'Artificial Intelligence' was coined by John McCarthy in 1956.",
    }
    return facts.get(topic.lower(), f"No pre-loaded fact for '{topic}'.")


# ════════════════════════════════════════════════════════════════════
# HELPER — lightweight LLM judge
# ════════════════════════════════════════════════════════════════════

def _judge(prompt: str, model: str = "openai:gpt-4o-mini") -> str:
    """Call a secondary LLM and return the response content."""
    judge_model = init_chat_model(model)
    result = judge_model.invoke([{"role": "user", "content": prompt}])
    return result.content.strip().upper()


# ════════════════════════════════════════════════════════════════════
# 1. SAFETY GUARDRAIL — CLASS SYNTAX
#    Uses a secondary LLM to classify the final response as
#    SAFE or UNSAFE before it reaches the user.
# ════════════════════════════════════════════════════════════════════

class SafetyGuardrailMiddleware(AgentMiddleware):
    """Model-based guardrail: Use an LLM to evaluate final response safety."""

    def __init__(self, safety_model: str = "openai:gpt-4o-mini"):
        super().__init__()
        self._model_name = safety_model

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return None

        verdict = _judge(
            f"Evaluate if this AI response is safe and appropriate. "
            f"Respond with only 'SAFE' or 'UNSAFE'.\n\n"
            f"Response: {last.content}",
            model=self._model_name,
        )

        print(f"  [SafetyGuardrail] Judge verdict: {verdict}")
        if "UNSAFE" in verdict:
            # Replace the unsafe response
            safe_reply = AIMessage(
                content=(
                    "I'm sorry, I cannot provide that response. "
                    "Please rephrase your question."
                )
            )
            return {**state, "messages": state["messages"][:-1] + [safe_reply]}

        return None  # SAFE — pass through unchanged


print("\n── 1. SafetyGuardrailMiddleware (class syntax) ──────────────")

agent_safety = create_agent(
    model="openai:gpt-4o-mini",
    tools=[web_search],
    middleware=[SafetyGuardrailMiddleware()],
    system_prompt="You are a helpful assistant.",
)

result_s = agent_safety.invoke({
    "messages": [{"role": "user", "content": "Tell me an interesting fact about Python."}]
})
print(f"Response: {result_s['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# 2. SAFETY GUARDRAIL — DECORATOR SYNTAX
#    Same behavior as above, using the @after_agent decorator.
# ════════════════════════════════════════════════════════════════════

@after_agent(can_jump_to=["end"])
def safety_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Decorator-style model-based safety guardrail."""
    if not state["messages"]:
        return None
    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        return None

    verdict = _judge(
        f"Is this response safe and appropriate? Reply only with 'SAFE' or 'UNSAFE'.\n\n"
        f"Response: {last.content}"
    )

    print(f"  [safety_guardrail fn] Judge verdict: {verdict}")
    if "UNSAFE" in verdict:
        safe_reply = AIMessage(content="I cannot provide that response.")
        return {**state, "messages": state["messages"][:-1] + [safe_reply]}
    return None


print("\n── 2. safety_guardrail (decorator syntax) ───────────────────")

agent_safety_dec = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_fact],
    middleware=[safety_guardrail],
    system_prompt="You are a helpful assistant.",
)

result_dec = agent_safety_dec.invoke({
    "messages": [{"role": "user", "content": "What is a fact about AI?"}]
})
print(f"Response: {result_dec['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# 3. TOPIC RELEVANCE GUARDRAIL
#    Checks whether the model's response stays on topic relative to
#    the agent's stated purpose (e.g. a cooking assistant).
# ════════════════════════════════════════════════════════════════════

class TopicRelevanceGuardrail(AgentMiddleware):
    """Ensure the response is relevant to the agent's designated domain."""

    def __init__(self, domain: str, model: str = "openai:gpt-4o-mini"):
        super().__init__()
        self.domain  = domain
        self._model  = model

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None
        last = state["messages"][-1]
        if not isinstance(last, AIMessage):
            return None

        verdict = _judge(
            f"Is the following response relevant to the topic of '{self.domain}'? "
            f"Reply with only 'RELEVANT' or 'OFF-TOPIC'.\n\n"
            f"Response: {last.content}",
            model=self._model,
        )

        print(f"  [TopicRelevance] Domain='{self.domain}', verdict={verdict}")
        if "OFF-TOPIC" in verdict:
            off_topic_reply = AIMessage(
                content=(
                    f"I'm designed to assist with {self.domain} topics only. "
                    "Please ask a relevant question."
                )
            )
            return {**state, "messages": state["messages"][:-1] + [off_topic_reply]}
        return None


print("\n── 3. TopicRelevanceGuardrail ───────────────────────────────")

agent_topic = create_agent(
    model="openai:gpt-4o-mini",
    tools=[web_search],
    middleware=[TopicRelevanceGuardrail(domain="cooking and recipes")],
    system_prompt="You are a cooking assistant. Help users with recipes and food questions.",
)

# On-topic
result_on = agent_topic.invoke({
    "messages": [{"role": "user", "content": "How do I make pasta carbonara?"}]
})
print(f"✅ On-topic: {result_on['messages'][-1].content[:100]}")

# Off-topic
result_off = agent_topic.invoke({
    "messages": [{"role": "user", "content": "What is the latest news in AI?"}]
})
print(f"🔀 Off-topic: {result_off['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# 4. OUTPUT QUALITY CHECKER
#    Evaluates whether the response is sufficiently detailed and
#    informative, prompting the model to retry if too vague.
# ════════════════════════════════════════════════════════════════════

class OutputQualityGuardrail(AgentMiddleware):
    """Reject vague or low-quality responses using an LLM judge."""

    def __init__(self, min_quality: str = "HIGH", model: str = "openai:gpt-4o-mini"):
        super().__init__()
        self.min_quality = min_quality
        self._model = model

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if not state["messages"]:
            return None
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or len(last.content) < 10:
            return None

        verdict = _judge(
            f"Rate the quality of this response as 'HIGH', 'MEDIUM', or 'LOW'. "
            f"A high-quality response is detailed, accurate, and helpful. "
            f"Reply with only one word.\n\nResponse: {last.content}",
            model=self._model,
        )

        print(f"  [QualityGuard] Quality verdict: {verdict}")
        if "LOW" in verdict:
            low_quality_reply = AIMessage(
                content=(
                    "I was unable to provide a complete answer. "
                    "Could you please clarify your question so I can help better?"
                )
            )
            return {**state, "messages": state["messages"][:-1] + [low_quality_reply]}
        return None


print("\n── 4. OutputQualityGuardrail ─────────────────────────────────")

agent_quality = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_fact],
    middleware=[OutputQualityGuardrail()],
    system_prompt=(
        "You are a knowledgeable assistant. "
        "Provide thorough, well-explained answers."
    ),
)

result_q = agent_quality.invoke({
    "messages": [{"role": "user", "content": "Explain how LangChain agents work."}]
})
print(f"Response: {result_q['messages'][-1].content[:150]}")

print("\n" + "═" * 60)
print("Model-Based vs Deterministic Trade-offs:")
print("  Deterministic: fast, cheap, 100% predictable — misses nuance")
print("  Model-based:   catches subtle violations — slower, costs tokens")
print("  Best practice: deterministic FIRST, model-based LAST in stack")
print("═" * 60)
print("\n✅ Model-based guardrails demo complete.")
