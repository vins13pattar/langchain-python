"""
05_full_longterm_memory_showcase.py
=====================================
Production-ready showcase: a Personal AI Assistant that learns and
remembers across sessions using all three memory types.

System design:
  ┌──────────────────────────────────────────────────────────────────┐
  │               Personal AI Assistant                              │
  ├──────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  Per-invocation context: UserContext(user_id, session_id)       │
  │                                                                  │
  │  On EVERY invocation:                                            │
  │    1. load_memory()  → pull semantic + episodic + procedural     │
  │    2. Answer with personalized context                           │
  │    3. save_interaction() → episode + any new facts/rules        │
  │                                                                  │
  │  InMemoryStore namespaces:                                       │
  │    (user_id, "semantic")  → factual knowledge                   │
  │    (user_id, "episodes")  → past interaction history            │
  │    (user_id, "rules")     → behavioral rules (procedural)       │
  │    (user_id, "sessions")  → session-level metadata              │
  │                                                                  │
  │  Scenarios:                                                      │
  │    A. First session — no memory, learns from conversation        │
  │    B. Second session — recalls facts and episodes                │
  │    C. User gives behavior instruction — saved as rule            │
  │    D. Multi-user — isolated namespaces per user                 │
  │    E. Memory search — semantic search across all memory types   │
  └──────────────────────────────────────────────────────────────────┘
"""

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from typing_extensions import TypedDict
from collections.abc import Sequence
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langgraph.store.base import IndexConfig
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

print("=" * 60)
print("Personal AI Assistant — Full Long-Term Memory Showcase")
print("=" * 60)

# ── Embeddings (for semantic search) ──────────────────────────────
openai_emb = OpenAIEmbeddings(model="text-embedding-3-small")

def embed_fn(texts: Sequence[str]) -> list[list[float]]:
    return openai_emb.embed_documents(list(texts))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ── Shared persistent store ────────────────────────────────────────
store = InMemoryStore(index=IndexConfig(embed=embed_fn, dims=1536))

# ── Short-term memory (conversation within a session) ─────────────
checkpointer = MemorySaver()


# ════════════════════════════════════════════════════════════════════
# CONTEXT SCHEMA
# ════════════════════════════════════════════════════════════════════

@dataclass
class PersonalContext:
    user_id:    str
    session_id: str = "default"


# ════════════════════════════════════════════════════════════════════
# TYPE DEFS
# ════════════════════════════════════════════════════════════════════

class UserFact(TypedDict):
    key:      str
    content:  str
    category: str   # identity | skills | style | project | tooling | context


class Episode(TypedDict):
    summary:    str
    topic:      str
    outcome:    Optional[str]


class BehaviorRule(TypedDict):
    rule:     str
    priority: int   # 1-10


# ════════════════════════════════════════════════════════════════════
# MEMORY TOOLS
# ════════════════════════════════════════════════════════════════════

@tool
def load_memory(query: str, runtime: ToolRuntime[PersonalContext]) -> str:
    """Load memory relevant to the current question.
    Searches semantic facts, episodic history, and procedural rules.
    Call this at the START of every conversation.
    """
    assert runtime.store is not None
    uid = runtime.context.user_id

    # Semantic: relevant facts
    sem = runtime.store.search((uid, "semantic"), query=query, limit=5)

    # Episodic: relevant past interactions
    epi = runtime.store.search((uid, "episodes"), query=query, limit=3)

    # Procedural: all behavior rules
    rules = runtime.store.search((uid, "rules"), limit=20)
    rules_sorted = sorted(rules, key=lambda r: r.value.get("priority", 0), reverse=True)

    parts = []
    if sem:
        facts = "\n".join(f"  [{r.value.get('category','?')}] {r.value['content']}" for r in sem)
        parts.append(f"What I know about you:\n{facts}")

    if epi:
        history = "\n".join(f"  [{r.value.get('timestamp','?')[:10]}] {r.value['summary']}" for r in epi)
        parts.append(f"Relevant past interactions:\n{history}")

    if rules_sorted:
        rule_lines = "\n".join(f"  [P{r.value['priority']}] {r.value['rule']}" for r in rules_sorted)
        parts.append(f"How I should respond (your preferences):\n{rule_lines}")

    if not parts:
        print(f"  [Memory] No memories found for {uid!r}")
        return "No previous context found. This appears to be a new user."

    print(f"  [Memory] Loaded: {len(sem)} facts, {len(epi)} episodes, {len(rules_sorted)} rules")
    return "\n\n".join(parts)


@tool
def save_fact(fact: UserFact, runtime: ToolRuntime[PersonalContext]) -> str:
    """Save a semantic fact about the user (background, skills, preferences, project info).
    Call when the user shares information about themselves.
    """
    assert runtime.store is not None
    uid = runtime.context.user_id
    runtime.store.put((uid, "semantic"), fact["key"], {
        "content": fact["content"], "category": fact["category"],
    })
    print(f"  [Memory:Write] fact [{fact['key']}]: {fact['content'][:60]}")
    return f"Noted: {fact['content'][:60]}"


@tool
def save_interaction(episode: Episode, runtime: ToolRuntime[PersonalContext]) -> str:
    """Save an episodic memory of this interaction.
    Call after each meaningful exchange to build interaction history.
    """
    assert runtime.store is not None
    uid      = runtime.context.user_id
    event_id = f"evt_{uuid.uuid4().hex[:8]}"
    runtime.store.put((uid, "episodes"), event_id, {
        "summary":    episode["summary"],
        "topic":      episode["topic"],
        "outcome":    episode.get("outcome"),
        "session":    runtime.context.session_id,
        "timestamp":  datetime.now().isoformat(),
    })
    print(f"  [Memory:Write] episode {event_id}: {episode['summary'][:60]}")
    return f"Interaction recorded: {episode['summary'][:60]}"


@tool
def save_behavior_rule(rule: BehaviorRule, runtime: ToolRuntime[PersonalContext]) -> str:
    """Save a behavioral rule. Call when the user:
    - Gives explicit instructions ("Always do X")
    - Corrects your behavior ("Stop doing Y, do Z instead")
    - States a preference ("I prefer X format")
    """
    assert runtime.store is not None
    uid     = runtime.context.user_id
    rule_id = f"rule_{uuid.uuid4().hex[:6]}"
    runtime.store.put((uid, "rules"), rule_id, {
        "rule": rule["rule"], "priority": rule["priority"],
        "session": runtime.context.session_id,
    })
    print(f"  [Memory:Write] rule P{rule['priority']}: {rule['rule'][:60]}")
    return f"I'll remember: {rule['rule'][:60]}"


@tool
def forget(key: str, namespace: str, runtime: ToolRuntime[PersonalContext]) -> str:
    """Delete a specific memory item.
    namespace: 'semantic' | 'episodes' | 'rules'
    """
    assert runtime.store is not None
    uid = runtime.context.user_id
    runtime.store.delete((uid, namespace), key)
    print(f"  [Memory:Delete] ({uid},{namespace})/{key}")
    return f"Memory {key!r} deleted from {namespace!r}."


# ════════════════════════════════════════════════════════════════════
# PERSONAL ASSISTANT AGENT
# ════════════════════════════════════════════════════════════════════

personal_assistant = create_agent(
    model="openai:gpt-4o-mini",
    tools=[load_memory, save_fact, save_interaction, save_behavior_rule, forget],
    store=store,
    checkpointer=checkpointer,   # short-term: remember within a session
    context_schema=PersonalContext,
    system_prompt=(
        "You are a Personal AI Assistant with long-term memory.\n\n"
        "ALWAYS start by calling load_memory() with the user's question to retrieve context.\n\n"
        "During the conversation:\n"
        "- Call save_fact() when the user shares background info, skills, or preferences\n"
        "- Call save_behavior_rule() when the user gives instructions about HOW you should respond\n"
        "- Call save_interaction() after each meaningful exchange to record history\n\n"
        "Use loaded memory to personalize every response. Reference what you remember."
    ),
)


def chat(user_id: str, session_id: str, message: str) -> str:
    """Send a message to the personal assistant."""
    cfg = {"configurable": {"thread_id": f"{user_id}:{session_id}"}}
    result = personal_assistant.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=cfg,
        context=PersonalContext(user_id=user_id, session_id=session_id),
    )
    return result["messages"][-1].content


# ════════════════════════════════════════════════════════════════════
# SCENARIO A: First Session — learns from conversation
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("SCENARIO A — First Session (no prior memory)")
print("─" * 60)

r_a1 = chat("alice", "session_1",
    "Hi! I'm Alice, a Senior ML engineer at TechCorp. "
    "I mainly work with Python and PyTorch. I prefer concise answers with code examples.")
print(f"\n  Turn 1:\n  {r_a1[:200]}")

r_a2 = chat("alice", "session_1",
    "I'm building a real-time recommendation engine. "
    "How should I structure my FAISS index for 10M items?")
print(f"\n  Turn 2:\n  {r_a2[:200]}")


# ════════════════════════════════════════════════════════════════════
# SCENARIO B: Second Session — recalls from long-term memory
# Different thread_id (new session) but same store (persistent memory)
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("SCENARIO B — Second Session (recalls long-term memory)")
print("─" * 60)

r_b1 = chat("alice", "session_2",
    "Hey, I'm back. What do you remember about me?")
print(f"\n  Turn 1 (recall check):\n  {r_b1[:300]}")

r_b2 = chat("alice", "session_2",
    "How's my recommendation engine project going from where we left off?")
print(f"\n  Turn 2 (project continuity):\n  {r_b2[:200]}")


# ════════════════════════════════════════════════════════════════════
# SCENARIO C: Behavioral Instructions — user teaches the agent
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("SCENARIO C — User Teaches Behavioral Rules")
print("─" * 60)

instructions = [
    "Please always add type hints to Python code you show me.",
    "Never use deprecated APIs. Always show the modern equivalent.",
    "When comparing approaches, use a table format.",
]

for instr in instructions:
    r = chat("alice", "session_3", instr)
    print(f"\n  Instruction: {instr[:70]}")
    print(f"  Response:    {r[:80]}")

# Verify rules were saved
all_rules = store.search(("alice", "rules"), limit=20)
print(f"\n  Stored rules ({len(all_rules)} total):")
for r in sorted(all_rules, key=lambda x: x.value.get("priority", 0), reverse=True):
    print(f"    [P{r.value['priority']}] {r.value['rule'][:70]}")


# ════════════════════════════════════════════════════════════════════
# SCENARIO D: Multi-User — completely isolated namespaces
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("SCENARIO D — Multi-User (isolated namespaces)")
print("─" * 60)

users = [
    ("bob",   "session_1", "I'm Bob, a DevOps engineer. I use Kubernetes and Terraform daily."),
    ("carol", "session_1", "I'm Carol, a data scientist specializing in NLP with Hugging Face."),
]

for user_id, session_id, intro in users:
    r = chat(user_id, session_id, intro)
    print(f"\n  [{user_id}] {r[:120]}")

# Verify namespace isolation
alice_facts = store.search(("alice", "semantic"), limit=20)
bob_facts   = store.search(("bob", "semantic"),   limit=20)
carol_facts = store.search(("carol", "semantic"),  limit=20)

print(f"\n  Memory isolation:")
print(f"    alice: {len(alice_facts)} semantic facts")
print(f"    bob:   {len(bob_facts)} semantic facts")
print(f"    carol: {len(carol_facts)} semantic facts")


# ════════════════════════════════════════════════════════════════════
# SCENARIO E: Memory Inspection — check what's stored
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("SCENARIO E — Memory Inspection (full store view)")
print("─" * 60)

for uid in ["alice", "bob", "carol"]:
    sem   = store.search((uid, "semantic"),  limit=50)
    epi   = store.search((uid, "episodes"),  limit=50)
    rules = store.search((uid, "rules"),     limit=50)
    print(f"\n  [{uid}]")
    print(f"    Semantic facts: {len(sem)}")
    print(f"    Episodes:       {len(epi)}")
    print(f"    Rules:          {len(rules)}")

    if sem:
        top = sem[0]
        print(f"    Top fact: [{top.value.get('category','?')}] {top.value['content'][:60]}")

    if epi:
        latest = max(epi, key=lambda e: e.value.get("timestamp", ""))
        print(f"    Latest episode: {latest.value['summary'][:60]}")


# ════════════════════════════════════════════════════════════════════
# MEMORY SEARCH — semantic search across memory
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("Memory Search Demo")
print("─" * 60)

search_queries = [
    ("alice", "semantic",  "machine learning expertise"),
    ("alice", "episodes",  "vector search and FAISS"),
    ("alice", "rules",     "how to format responses"),
]

for uid, namespace, query in search_queries:
    results = store.search((uid, namespace), query=query, limit=2)
    print(f"\n  [{uid}/{namespace}] query={query!r}")
    for r in results:
        val = r.value
        content = val.get("content") or val.get("summary") or val.get("rule") or str(val)
        print(f"    [{r.key}] {content[:70]}")


print("\n" + "═" * 60)
print("Personal AI Assistant Showcase Summary:")
print("  InMemoryStore(index=IndexConfig(...)) → semantic search")
print("  MemorySaver checkpointer             → short-term (per session)")
print("  3 namespaces: semantic + episodes + rules")
print("  load_memory()       → pull all relevant context on every turn")
print("  save_fact()         → learn user background and preferences")
print("  save_interaction()  → build episodic history")
print("  save_behavior_rule()→ learn how to respond")
print("  forget()            → delete stale memories")
print("  context=PersonalContext(user_id, session_id) → per-request scope")
print("  thread_id = user_id:session_id → both short + long term work")
print("═" * 60)
print("\n✅ Full long-term memory showcase complete.")
