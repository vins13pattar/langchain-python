"""
01_subagents.py
================
Demonstrates the SUBAGENTS multi-agent pattern — a central supervisor
agent coordinates specialized subagents by calling them as tools.

Concepts covered:
  - Wrapping a subagent as a @tool the main agent can call
  - Tool per agent — distinct tool for each specialist
  - Subagent state isolation (stateless by design)
  - Synchronous invocation — main agent waits for subagent
  - Parallel tool calls — multiple subagents in one turn
  - Custom subagent inputs using ToolRuntime + AgentState
  - Custom subagent outputs using Command + InjectedToolCallId

Key characteristics:
  - All routing passes through the main (supervisor) agent
  - Subagents don't remember past interactions
  - Context isolation: each subagent has a clean context window
  - Main agent can call multiple subagents in a single turn
"""

import asyncio
from typing import Annotated
from dotenv import load_dotenv

from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime, InjectedToolCallId
from langchain.messages import ToolMessage
from langgraph.types import Command
from langgraph.store.memory import InMemoryStore

load_dotenv()

print("=" * 60)
print("Subagents Pattern")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# SPECIALIST SUBAGENTS
# Each subagent is a standalone agent with its own tools and prompt.
# They are stateless — no memory of past invocations.
# ════════════════════════════════════════════════════════════════════

# Research subagent — domain: information gathering
research_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],   # Uses model knowledge only (add real tools in production)
    system_prompt=(
        "You are a research specialist. Given a topic, provide concise, "
        "accurate research findings (3-5 key facts). "
        "Always include your final findings in your response — "
        "the supervisor only sees your final message."
    ),
)

# Writer subagent — domain: content creation
writer_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    system_prompt=(
        "You are a professional writer. Given research findings and requirements, "
        "produce well-structured, engaging content. "
        "The supervisor only sees your final message, so include the complete draft."
    ),
)

# Reviewer subagent — domain: quality assurance
reviewer_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    system_prompt=(
        "You are a content reviewer. Review provided content for accuracy, "
        "clarity, and quality. Give a short review score (1-10) and 2-3 "
        "specific improvement suggestions. Keep it concise."
    ),
)


# ════════════════════════════════════════════════════════════════════
# PART 1: BASIC TOOL-PER-AGENT PATTERN
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Basic Subagent-as-Tool Pattern ────────────────────────")

@tool("research", description="Research a topic and return key findings")
def call_research_agent(topic: str) -> str:
    """Invoke the research specialist subagent."""
    result = research_agent.invoke({
        "messages": [{"role": "user", "content": f"Research: {topic}"}]
    })
    return result["messages"][-1].content


@tool("write_content", description="Write content based on research findings and requirements")
def call_writer_agent(research_findings: str, content_requirements: str) -> str:
    """Invoke the writer specialist subagent."""
    query = f"Research:\n{research_findings}\n\nRequirements:\n{content_requirements}"
    result = writer_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content


@tool("review_content", description="Review content for quality and accuracy")
def call_reviewer_agent(content: str) -> str:
    """Invoke the reviewer specialist subagent."""
    result = reviewer_agent.invoke({
        "messages": [{"role": "user", "content": f"Review this content:\n\n{content}"}]
    })
    return result["messages"][-1].content


# Main (supervisor) agent — coordinates the specialists
supervisor = create_agent(
    model="openai:gpt-4o-mini",
    tools=[call_research_agent, call_writer_agent, call_reviewer_agent],
    system_prompt=(
        "You are a content production supervisor. "
        "For content tasks: first research the topic, then write based on findings, "
        "then review the draft. Synthesize the results for the user."
    ),
)

result = supervisor.invoke({
    "messages": [{"role": "user", "content":
        "Write a brief overview of quantum computing for a general audience."}]
})
print(f"\nResult:\n{result['messages'][-1].content[:400]}")


# ════════════════════════════════════════════════════════════════════
# PART 2: SUBAGENT INPUTS — ToolRuntime + AgentState
# Inject state (conversation history) into subagent context.
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Custom Subagent Inputs (ToolRuntime + AgentState) ──────")

class CustomState(AgentState):
    domain: str          # Extra state the main agent tracks


@tool("research_with_context", description="Research a topic using conversation context")
def call_research_with_context(query: str, runtime: ToolRuntime[None, CustomState]) -> str:
    """Inject state context into subagent call."""
    state  = runtime.state
    domain = state.get("domain", "general")

    # Pass main agent conversation + domain context to subagent
    subagent_input = {
        "messages": [
            {"role": "system", "content": f"You are a research specialist in {domain}."},
            {"role": "user",   "content": query},
        ]
    }
    result = research_agent.invoke(subagent_input)
    content = result["messages"][-1].content
    print(f"  [Research] domain={domain!r}, result snippet: {content[:80]}")
    return content


supervisor2 = create_agent(
    model="openai:gpt-4o-mini",
    tools=[research_with_context],
    system_prompt="You are a research coordinator.",
)

result2 = supervisor2.invoke({
    "messages": [{"role": "user", "content":
        "What are the key principles of machine learning?"}],
    "domain": "artificial intelligence",
})
print(f"\nResult (with context): {result2['messages'][-1].content[:200]}")


# ════════════════════════════════════════════════════════════════════
# PART 3: SUBAGENT OUTPUTS — Command + InjectedToolCallId
# Pass additional state back from subagent to main agent.
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Custom Subagent Outputs (Command + InjectedToolCallId) ─")

class StateWithDraft(AgentState):
    last_draft: str      # State key updated by the writer subagent


@tool("write_and_store", description="Write content and store the draft in shared state")
def call_writer_store_state(
    requirements: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Write content and store it in the main agent's state via Command."""
    result = writer_agent.invoke({
        "messages": [{"role": "user", "content": requirements}]
    })
    draft = result["messages"][-1].content

    # Command updates both the tool message AND the last_draft state key
    return Command(update={
        "last_draft": draft,
        "messages": [
            ToolMessage(
                content=f"Draft written ({len(draft)} chars). Preview: {draft[:100]}...",
                tool_call_id=tool_call_id,
            )
        ]
    })


supervisor3 = create_agent(
    model="openai:gpt-4o-mini",
    tools=[write_and_store],
    system_prompt="You are a writing coordinator.",
)

result3 = supervisor3.invoke({
    "messages": [{"role": "user", "content":
        "Write a 2-sentence definition of neural networks."}],
    "last_draft": "",
})
print(f"\nState last_draft: {result3.get('last_draft', '')[:120]}")
print(f"Final response:   {result3['messages'][-1].content[:150]}")


# ════════════════════════════════════════════════════════════════════
# PART 4: PARALLEL SUBAGENTS
# The main agent can invoke multiple subagents in a single turn
# via parallel tool calls.
# ════════════════════════════════════════════════════════════════════

print("\n── 4. Parallel Subagent Execution ───────────────────────────")
print("  (Agent invokes research + write + review in parallel turns)")

async def demo_parallel():
    # Create async versions of the subagent tools
    python_agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[],
        system_prompt="You are a Python expert. Summarize Python's key strengths in 2 sentences.",
    )
    js_agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[],
        system_prompt="You are a JavaScript expert. Summarize JS's key strengths in 2 sentences.",
    )
    rust_agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[],
        system_prompt="You are a Rust expert. Summarize Rust's key strengths in 2 sentences.",
    )

    @tool("analyze_python", description="Get Python's strengths from the Python expert")
    def analyze_python(query: str) -> str:
        r = python_agent.invoke({"messages": [{"role": "user", "content": query}]})
        return r["messages"][-1].content

    @tool("analyze_javascript", description="Get JavaScript's strengths from the JS expert")
    def analyze_javascript(query: str) -> str:
        r = js_agent.invoke({"messages": [{"role": "user", "content": query}]})
        return r["messages"][-1].content

    @tool("analyze_rust", description="Get Rust's strengths from the Rust expert")
    def analyze_rust(query: str) -> str:
        r = rust_agent.invoke({"messages": [{"role": "user", "content": query}]})
        return r["messages"][-1].content

    parallel_supervisor = create_agent(
        model="openai:gpt-4o-mini",
        tools=[analyze_python, analyze_javascript, analyze_rust],
        system_prompt=(
            "You coordinate language experts. For multi-language comparisons, "
            "invoke all relevant expert agents (you can call multiple tools at once), "
            "then synthesize their responses."
        ),
    )

    r = parallel_supervisor.invoke({
        "messages": [{"role": "user", "content":
            "Compare Python, JavaScript, and Rust for web development."}]
    })
    print(f"\nParallel result: {r['messages'][-1].content[:300]}")

asyncio.run(demo_parallel())

print("\n" + "═" * 60)
print("Subagents Pattern Summary:")
print("  @tool wraps subagent.invoke() → returns final message content")
print("  ToolRuntime[None, State] → inject state into subagent call")
print("  Command(update={...})    → push state from subagent to supervisor")
print("  InjectedToolCallId       → construct ToolMessage for Command")
print("  Multiple tools           → parallel calls in single turn")
print("  Stateless by design      → clean context window per call")
print("═" * 60)
print("\n✅ Subagents pattern demo complete.")
