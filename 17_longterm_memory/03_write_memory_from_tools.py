"""
03_write_memory_from_tools.py
==============================
Demonstrates writing to long-term memory from agent tools.
The agent learns about users and persists knowledge across sessions.

Concepts covered:
  - TypedDict — structured schema for what the LLM saves
  - save_user_info() — official docs pattern for writing memories
  - Upsert semantics — put() overwrites existing key
  - Saving structured facts (profile, preferences, goals)
  - Saving episodic memories (events, interactions, history)
  - Saving procedural knowledge (rules, behaviors, instructions)
  - Deleting stale memories
  - Memory extraction — LLM extracts facts from conversation
  - Write + read cycle — save then verify
  - Multi-key writes — updating multiple fields atomically

Key difference from read tools:
  - Read: runtime.store.get() / .search()
  - Write: runtime.store.put(namespace, key, dict)
"""

import uuid
from dataclasses import dataclass
from typing import Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore

load_dotenv()

print("=" * 60)
print("Writing Long-Term Memory from Tools")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# PART 1: BASIC WRITE — save user info (official docs pattern)
# Exactly as shown in the LangChain documentation.
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Basic Write — save_user_info (official docs pattern) ──")

@dataclass
class UserContext:
    user_id: str


# TypedDict defines the structure the LLM uses to fill in data
class UserInfo(TypedDict):
    name: str


store1 = InMemoryStore()


@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[UserContext]) -> str:
    """Save user info."""
    assert runtime.store is not None
    store   = runtime.store
    user_id = runtime.context.user_id
    store.put(("users",), user_id, dict(user_info))
    return "Successfully saved user info."


agent1 = create_agent(
    model="openai:gpt-4o-mini",
    tools=[save_user_info],
    store=store1,
    context_schema=UserContext,
    system_prompt="You are a helpful assistant. When users introduce themselves, save their information.",
)

agent1.invoke(
    {"messages": [{"role": "user", "content": "My name is John Smith"}]},
    context=UserContext(user_id="user_123"),
)

# Verify directly from the store
item = store1.get(("users",), "user_123")
print(f"\n  Saved: {item.value}")


# ════════════════════════════════════════════════════════════════════
# PART 2: RICH STRUCTURED WRITES — TypedDict schemas
# Different TypedDicts for different types of information.
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Structured Writes with TypedDict schemas ───────────────")

class FullProfile(TypedDict):
    name:       str
    role:       str
    company:    str
    experience: str  # "junior" | "mid" | "senior" | "expert"
    location:   str


class CodingPreferences(TypedDict):
    primary_language:  str
    secondary_languages: list[str]
    preferred_ide:     str
    test_framework:    str
    code_style:        str   # "functional" | "oop" | "mixed"


class CommunicationStyle(TypedDict):
    verbosity:    str   # "concise" | "detailed"
    format:       str   # "prose" | "bullets" | "mixed"
    examples:     bool  # likes code examples?
    depth:        str   # "beginner" | "intermediate" | "expert"


store2 = InMemoryStore()


@tool
def save_user_profile(profile: FullProfile, runtime: ToolRuntime[UserContext]) -> str:
    """Save the user's professional profile. Call when user shares their background."""
    assert runtime.store is not None
    uid = runtime.context.user_id
    runtime.store.put((uid,), "profile", dict(profile))
    print(f"  [Store] Saved profile for {uid}: {profile['name']}")
    return f"Profile saved for {profile['name']}."


@tool
def save_coding_preferences(prefs: CodingPreferences, runtime: ToolRuntime[UserContext]) -> str:
    """Save the user's coding preferences. Call when user shares tech stack or tool preferences."""
    assert runtime.store is not None
    uid = runtime.context.user_id
    runtime.store.put((uid,), "coding_prefs", dict(prefs))
    print(f"  [Store] Saved coding prefs for {uid}: lang={prefs['primary_language']}")
    return "Coding preferences saved."


@tool
def save_communication_style(style: CommunicationStyle, runtime: ToolRuntime[UserContext]) -> str:
    """Save how the user prefers to communicate. Call when user expresses style preferences."""
    assert runtime.store is not None
    uid = runtime.context.user_id
    runtime.store.put((uid,), "comm_style", dict(style))
    print(f"  [Store] Saved comm style for {uid}: {style['verbosity']}/{style['format']}")
    return "Communication style preferences saved."


agent2 = create_agent(
    model="openai:gpt-4o-mini",
    tools=[save_user_profile, save_coding_preferences, save_communication_style],
    store=store2,
    context_schema=UserContext,
    system_prompt=(
        "You are an adaptive assistant that learns about users. "
        "When users share background information, coding preferences, or how they like to communicate, "
        "save it using the appropriate tool. Extract the relevant information and call the right tool."
    ),
)

intro_message = (
    "Hi! I'm Alice Chen, a Senior Python developer at TechCorp based in Singapore. "
    "I use Python mainly, with some Go on the side. My IDE is VS Code. "
    "I prefer concise answers with code examples, at an expert level. "
    "I use pytest for testing and write functional-style code."
)

result2 = agent2.invoke(
    {"messages": [{"role": "user", "content": intro_message}]},
    context=UserContext(user_id="alice"),
)
print(f"\n  Agent: {result2['messages'][-1].content[:100]}")

# Verify writes
for key in ["profile", "coding_prefs", "comm_style"]:
    item = store2.get(("alice",), key)
    if item:
        print(f"  Stored [{key}]: {item.value}")


# ════════════════════════════════════════════════════════════════════
# PART 3: EPISODIC MEMORY — storing events and interactions
# Each event gets a unique key so history accumulates.
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Episodic Memory (event history) ───────────────────────")

class InteractionEvent(TypedDict):
    topic:      str
    summary:    str
    sentiment:  str   # "positive" | "neutral" | "negative"
    follow_up:  Optional[str]


store3 = InMemoryStore()


@tool
def save_interaction_memory(event: InteractionEvent, runtime: ToolRuntime[UserContext]) -> str:
    """Save a memory of this interaction for future reference.
    Use at the end of a meaningful conversation to capture what was discussed.
    """
    assert runtime.store is not None
    uid = runtime.context.user_id
    # Use unique key so history accumulates (not overwritten)
    event_id = f"evt_{uuid.uuid4().hex[:8]}"
    runtime.store.put((uid, "history"), event_id, {
        "topic":     event["topic"],
        "summary":   event["summary"],
        "sentiment": event["sentiment"],
        "follow_up": event.get("follow_up"),
    })
    print(f"  [Store] Episodic event {event_id}: topic={event['topic']!r}")
    return f"Interaction saved (ID: {event_id})."


agent3 = create_agent(
    model="openai:gpt-4o-mini",
    tools=[save_interaction_memory],
    store=store3,
    context_schema=UserContext,
    system_prompt=(
        "You are a helpful assistant with memory. "
        "After answering any substantive question, save an episodic memory "
        "of the interaction using save_interaction_memory."
    ),
)

for msg in [
    "How do I implement a binary search tree in Python?",
    "What's the best way to handle database migrations?",
]:
    agent3.invoke(
        {"messages": [{"role": "user", "content": msg}]},
        context=UserContext(user_id="alice"),
    )

# Check accumulated history
history = store3.search(("alice", "history"), limit=10)
print(f"\n  Episodic history: {len(history)} events")
for h in history:
    print(f"    [{h.key}] topic={h.value['topic']!r}, sentiment={h.value['sentiment']!r}")


# ════════════════════════════════════════════════════════════════════
# PART 4: PROCEDURAL MEMORY — storing rules and behavior
# Agent learns what behaviors to use with this user.
# ════════════════════════════════════════════════════════════════════

print("\n── 4. Procedural Memory (rules and behaviors) ───────────────")

class BehaviorRule(TypedDict):
    rule:        str
    context:     str   # when does this rule apply
    priority:    int   # 1-10, higher = more important


store4 = InMemoryStore()


@tool
def add_behavior_rule(rule: BehaviorRule, runtime: ToolRuntime[UserContext]) -> str:
    """Add a rule about how to behave with this user.
    Call when the user expresses preferences, corrects your behavior, or gives instructions.
    """
    assert runtime.store is not None
    uid     = runtime.context.user_id
    rule_id = f"rule_{uuid.uuid4().hex[:6]}"
    runtime.store.put((uid, "rules"), rule_id, dict(rule))
    print(f"  [Store] Rule {rule_id}: {rule['rule'][:60]}")
    return f"Behavior rule saved: {rule['rule'][:60]}"


@tool
def get_behavior_rules(runtime: ToolRuntime[UserContext]) -> str:
    """Retrieve all behavioral rules for this user, sorted by priority."""
    assert runtime.store is not None
    uid   = runtime.context.user_id
    rules = runtime.store.search((uid, "rules"), limit=20)
    if not rules:
        return "No behavior rules set for this user."
    sorted_rules = sorted(rules, key=lambda r: r.value.get("priority", 0), reverse=True)
    lines = [
        f"  [P{r.value['priority']}] {r.value['rule']}"
        for r in sorted_rules
    ]
    return "Active rules:\n" + "\n".join(lines)


agent4 = create_agent(
    model="openai:gpt-4o-mini",
    tools=[add_behavior_rule, get_behavior_rules],
    store=store4,
    context_schema=UserContext,
    system_prompt=(
        "You are an adaptive assistant. When users give instructions about how to communicate "
        "or correct your behavior, save it as a behavior rule. "
        "At the start of each conversation, check get_behavior_rules to apply known rules."
    ),
)

feedback_msgs = [
    "Please always provide code examples when explaining programming concepts.",
    "Don't use overly formal language — keep it casual and friendly.",
    "When I ask about Python, always mention type hints.",
]

for msg in feedback_msgs:
    agent4.invoke(
        {"messages": [{"role": "user", "content": msg}]},
        context=UserContext(user_id="alice"),
    )

# Show stored rules
rules = store4.search(("alice", "rules"), limit=10)
print(f"\n  Stored {len(rules)} behavior rules:")
for r in rules:
    print(f"    [{r.value.get('priority', '?')}] {r.value['rule'][:70]}")


# ════════════════════════════════════════════════════════════════════
# PART 5: DELETE — removing stale memories
# ════════════════════════════════════════════════════════════════════

print("\n── 5. Deleting Stale Memories ───────────────────────────────")

store5 = InMemoryStore()
store5.put(("alice",), "old_project", {"project": "Legacy Rails app", "status": "done"})
store5.put(("alice",), "current_project", {"project": "FastAPI service", "status": "active"})

before = store5.get(("alice",), "old_project")
print(f"\n  Before delete: {before.value}")

store5.delete(("alice",), "old_project")

after = store5.get(("alice",), "old_project")
print(f"  After delete:  {after!r}")

current = store5.get(("alice",), "current_project")
print(f"  Current still exists: {current.value['project']!r}")

print("\n" + "═" * 60)
print("Write Memory in Tools Summary:")
print("  TypedDict schema          → structured data the LLM fills in")
print("  runtime.store.put(ns,k,d) → write/upsert a memory")
print("  Unique key per event      → accumulate episodic history")
print("  Fixed key per profile     → overwrite/update profile data")
print("  Behavior rules            → procedural memory for agent behavior")
print("  runtime.store.delete()    → remove stale memories")
print("  Write + read cycle        → put() then verify with get()")
print("═" * 60)
print("\n✅ Writing long-term memory from tools demo complete.")
