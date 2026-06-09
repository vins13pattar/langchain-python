"""
multi_agent_overview.py — Multi-Agent Patterns: all key concepts in one file
Covers: subagents (tool-per-agent), dispatch, handoffs, skills, router, full showcase
"""

import asyncio
from typing import Annotated
from dotenv import load_dotenv

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import dynamic_prompt, ModelRequest, wrap_model_call
from langchain.tools import tool, ToolRuntime, InjectedToolCallId
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

load_dotenv()

def section(title): print(f"\n{'─'*55}\n{title}\n{'─'*55}")


# ════════════════════════════════════════════════════════════════════
# 1. SUBAGENTS — specialist agents as tools (supervisor pattern)
# ════════════════════════════════════════════════════════════════════
section("1. SUBAGENTS (supervisor pattern)")

# Each specialist is a standalone create_agent()
research_agent = create_agent(
    model="openai:gpt-4o-mini", tools=[],
    system_prompt="You are a research specialist. Give 3-5 key facts on the topic. Final answer only.",
)
writer_agent = create_agent(
    model="openai:gpt-4o-mini", tools=[],
    system_prompt="You are a professional writer. Produce well-structured content. Final answer only.",
)
reviewer_agent = create_agent(
    model="openai:gpt-4o-mini", tools=[],
    system_prompt="You are a content reviewer. Give a score (1-10) and 2 improvements. Be concise.",
)

# Wrap each subagent as a regular @tool
@tool("research", description="Research a topic and return key findings")
def call_research(topic: str) -> str:
    return research_agent.invoke({"messages": [{"role": "user", "content": f"Research: {topic}"}]})["messages"][-1].content

@tool("write_content", description="Write content given research and requirements")
def call_writer(research_findings: str, requirements: str) -> str:
    q = f"Research:\n{research_findings}\n\nRequirements:\n{requirements}"
    return writer_agent.invoke({"messages": [{"role": "user", "content": q}]})["messages"][-1].content

@tool("review_content", description="Review content for quality and accuracy")
def call_reviewer(content: str) -> str:
    return reviewer_agent.invoke({"messages": [{"role": "user", "content": f"Review:\n{content}"}]})["messages"][-1].content

supervisor = create_agent(
    model="openai:gpt-4o-mini",
    tools=[call_research, call_writer, call_reviewer],
    system_prompt="You are a content supervisor. Research → write → review for content tasks.",
)
r = supervisor.invoke({"messages": [HumanMessage("Write a 2-sentence overview of quantum computing.")]})
print("Subagent result:", r["messages"][-1].content[:200])


# ════════════════════════════════════════════════════════════════════
# 2. CUSTOM INPUTS — inject state into subagent via ToolRuntime
# ════════════════════════════════════════════════════════════════════
section("2. SUBAGENT INPUTS (ToolRuntime + state)")

class CustomState(AgentState):
    domain: str  # extra state the supervisor tracks

@tool("research_with_context", description="Research a topic using conversation context")
def call_research_ctx(query: str, runtime: ToolRuntime[None, CustomState]) -> str:
    domain = runtime.state.get("domain", "general")
    result = research_agent.invoke({
        "messages": [
            {"role": "system", "content": f"You are a research specialist in {domain}."},
            {"role": "user",   "content": query},
        ]
    })
    return result["messages"][-1].content

supervisor2 = create_agent(model="openai:gpt-4o-mini", tools=[call_research_ctx], system_prompt="Research coordinator.")
r = supervisor2.invoke({"messages": [HumanMessage("What are the key principles of ML?")], "domain": "artificial intelligence"})
print("Context-aware:", r["messages"][-1].content[:150])


# ════════════════════════════════════════════════════════════════════
# 3. CUSTOM OUTPUTS — Command + InjectedToolCallId to write state
# ════════════════════════════════════════════════════════════════════
section("3. SUBAGENT OUTPUTS (Command + InjectedToolCallId)")

class StateWithDraft(AgentState):
    last_draft: str

@tool("write_and_store", description="Write content and store the draft in shared state")
def call_writer_store(requirements: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    draft = writer_agent.invoke({"messages": [{"role": "user", "content": requirements}]})["messages"][-1].content
    return Command(update={
        "last_draft": draft,
        "messages": [ToolMessage(content=f"Draft ready ({len(draft)} chars). Preview: {draft[:80]}...", tool_call_id=tool_call_id)],
    })

supervisor3 = create_agent(model="openai:gpt-4o-mini", tools=[call_writer_store], system_prompt="Writing coordinator.")
r = supervisor3.invoke({"messages": [HumanMessage("Write a 2-sentence definition of neural networks.")], "last_draft": ""})
print("Stored draft:", r.get("last_draft", "")[:100])
print("Final reply:", r["messages"][-1].content[:100])


# ════════════════════════════════════════════════════════════════════
# 4. HANDOFFS — state drives behavior/routing dynamically
# ════════════════════════════════════════════════════════════════════
section("4. HANDOFFS (state-driven routing)")

AGENT_PROMPTS = {
    "general":   "You are a general support agent. Route specialists as needed.",
    "billing":   "You are a billing specialist. Handle refunds and billing queries.",
    "technical": "You are a technical specialist. Diagnose and fix device issues.",
}

AGENT_TOOLS_MAP = {
    "general":   {"transfer_to_billing", "transfer_to_technical"},
    "billing":   {"process_refund", "transfer_to_technical"},
    "technical": {"run_diagnostics", "transfer_to_billing"},
}

class HandoffState(AgentState):
    active_agent: str

@tool
def transfer_to_billing(reason: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Transfer to billing specialist. Args: reason."""
    return Command(update={"active_agent": "billing", "messages": [ToolMessage(content="Transferred to Billing.", tool_call_id=tool_call_id)]})

@tool
def transfer_to_technical(reason: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Transfer to technical specialist. Args: reason."""
    return Command(update={"active_agent": "technical", "messages": [ToolMessage(content="Transferred to Technical.", tool_call_id=tool_call_id)]})

@tool
def process_refund(amount: float, reason: str) -> str:
    """Process a billing refund. Args: amount, reason."""
    print(f"  [Tool] process_refund(${amount})")
    return f"Refund of ${amount:.2f} processed."

@tool
def run_diagnostics(device_id: str) -> str:
    """Run device diagnostics. Args: device_id."""
    return f"Diagnostics for {device_id}: No hardware faults."

all_handoff_tools = [transfer_to_billing, transfer_to_technical, process_refund, run_diagnostics]

@dynamic_prompt
def routing_prompt(request: ModelRequest) -> str:
    return AGENT_PROMPTS.get(request.runtime.state.get("active_agent", "general"), AGENT_PROMPTS["general"])

@wrap_model_call
def filter_tools_by_agent(request: ModelRequest, handler) -> object:
    active = request.runtime.state.get("active_agent", "general")
    allowed = AGENT_TOOLS_MAP.get(active, set())
    filtered = [t for t in request.tools if t.name in allowed]
    print(f"  [Filter] active={active!r}  tools={[t.name for t in filtered]}")
    return handler(request.override(tools=filtered))

handoff_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=all_handoff_tools,
    middleware=[routing_prompt, filter_tools_by_agent],
    checkpointer=MemorySaver(),
    system_prompt="Customer service agent.",
)

cfg = {"configurable": {"thread_id": "handoff-1"}}
for msg in ["I was charged twice for my subscription.", "Yes, I want a $9.99 refund please."]:
    r = handoff_agent.invoke({"messages": [{"role": "user", "content": msg}], "active_agent": "general"}, config=cfg)
    print(f"  [{r.get('active_agent','?')}] {r['messages'][-1].content[:100]}")


# ════════════════════════════════════════════════════════════════════
# 5. SKILLS — shared capability libraries
# ════════════════════════════════════════════════════════════════════
section("5. SKILLS (shared tool libraries)")

# Skills are reusable sets of tools across multiple agents
WRITING_SKILLS = []

@tool("improve_style", description="Improve the writing style and readability")
def improve_style(text: str) -> str:
    return writer_agent.invoke({"messages": [{"role": "user", "content": f"Improve style:\n{text}"}]})["messages"][-1].content

@tool("summarize", description="Summarize text in 1-2 sentences")
def summarize(text: str) -> str:
    return writer_agent.invoke({"messages": [{"role": "user", "content": f"Summarize in 2 sentences:\n{text}"}]})["messages"][-1].content

WRITING_SKILLS = [improve_style, summarize]

# Compose: specialized agent using skill set
blog_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[call_research, *WRITING_SKILLS],
    system_prompt="You are a blog writing agent. Research, draft, improve style.",
)
r = blog_agent.invoke({"messages": [HumanMessage("Write a summary about Python programming.")]})
print("Skill-based agent:", r["messages"][-1].content[:150])


# ════════════════════════════════════════════════════════════════════
# 6. ROUTER — classify intent and dispatch to specialist
# ════════════════════════════════════════════════════════════════════
section("6. ROUTER (classify → dispatch)")

# Intent routing via model
def classify_intent(user_message: str) -> str:
    """Classify user intent into one of: research, write, review."""
    classifier = create_agent(
        model="openai:gpt-4o-mini", tools=[],
        system_prompt="Classify the user intent as EXACTLY one of: research, write, review. Reply with only the category.",
    )
    r = classifier.invoke({"messages": [{"role": "user", "content": user_message}]})
    return r["messages"][-1].content.strip().lower()

INTENT_AGENTS = {
    "research": create_agent(model="openai:gpt-4o-mini", tools=[call_research], system_prompt="Research specialist."),
    "write":    create_agent(model="openai:gpt-4o-mini", tools=[call_writer], system_prompt="Writing specialist."),
    "review":   create_agent(model="openai:gpt-4o-mini", tools=[call_reviewer], system_prompt="Review specialist."),
}

def route_request(user_message: str) -> str:
    intent = classify_intent(user_message)
    agent = INTENT_AGENTS.get(intent, supervisor)
    print(f"  [Router] classified as: {intent!r}")
    r = agent.invoke({"messages": [{"role": "user", "content": user_message}]})
    return r["messages"][-1].content

for q in ["Research the history of Python language.", "Write a haiku about coding."]:
    result = route_request(q)
    print(f"  Q: {q[:50]}")
    print(f"  A: {result[:100]}")


# ════════════════════════════════════════════════════════════════════
# 7. PARALLEL SUBAGENTS — async parallel calls in one turn
# ════════════════════════════════════════════════════════════════════
section("7. PARALLEL SUBAGENTS (async)")

python_agent = create_agent(model="openai:gpt-4o-mini", tools=[], system_prompt="Python expert. 2-sentence summary.")
js_agent     = create_agent(model="openai:gpt-4o-mini", tools=[], system_prompt="JavaScript expert. 2-sentence summary.")

@tool("analyze_python", description="Get Python strengths from the Python expert")
def analyze_python(query: str) -> str:
    return python_agent.invoke({"messages": [{"role": "user", "content": query}]})["messages"][-1].content

@tool("analyze_javascript", description="Get JS strengths from the JS expert")
def analyze_javascript(query: str) -> str:
    return js_agent.invoke({"messages": [{"role": "user", "content": query}]})["messages"][-1].content

parallel_supervisor = create_agent(
    model="openai:gpt-4o-mini",
    tools=[analyze_python, analyze_javascript],
    system_prompt="Language comparison supervisor. Call all experts simultaneously, then synthesize.",
)
r = parallel_supervisor.invoke({"messages": [HumanMessage("Compare Python vs JavaScript for web dev.")]})
print("Parallel result:", r["messages"][-1].content[:200])


print("""
Multi-Agent Patterns:
  Subagents     → specialists wrapped as @tools; supervisor orchestrates
  Handoffs      → state variable (current_step/active_agent) drives routing
  Skills        → reusable tool libraries shared across multiple agents
  Router        → classify intent → dispatch to specialist agent
  Parallel      → model calls multiple tools (agents) in one turn

  Subagent in/out:
    ToolRuntime[None, State] → read supervisor state in subagent tool
    Command(update={...})    → write state back from subagent to supervisor
    InjectedToolCallId       → construct ToolMessage for Command updates
""")
