"""
04_memory_types.py
===================
Demonstrates the three canonical memory types from the LangChain
conceptual guide, implemented with InMemoryStore.

Memory types:
  - Semantic memory:   factual knowledge (user profile, preferences)
  - Episodic memory:   past events / interaction history
  - Procedural memory: rules and behaviors the agent should follow

Concepts covered:
  - Semantic memory: facts about user stored under (user_id, "semantic")
  - Episodic memory: time-stamped events, unique key per event
  - Procedural memory: rules with priority ordering
  - Memory extraction: LLM extracts facts from conversation text
  - Memory recall: semantic search to build personalized context
  - Cross-memory synthesis: combine all three types for a rich prompt
"""

import json
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

load_dotenv()

print("=" * 60)
print("Memory Types — Semantic, Episodic, Procedural")
print("=" * 60)

openai_emb = OpenAIEmbeddings(model="text-embedding-3-small")

def embed_fn(texts: Sequence[str]) -> list[list[float]]:
    return openai_emb.embed_documents(list(texts))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


@dataclass
class UserContext:
    user_id: str


# ════════════════════════════════════════════════════════════════════
# PART 1: SEMANTIC MEMORY — factual knowledge about the user
# Stored under: (user_id, "semantic") namespace
# Key:   descriptive string ("profile", "skills", "preferences")
# Value: {"content": natural language fact, "category": str}
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Semantic Memory (factual knowledge) ───────────────────")

semantic_store = InMemoryStore(index=IndexConfig(embed=embed_fn, dims=1536))

initial_facts = [
    ("profile",     {"content": "Alice Chen is a Senior Python developer at TechCorp, Singapore. 10 years exp.", "category": "identity"}),
    ("expertise",   {"content": "Expert in machine learning, PyTorch, and distributed systems.", "category": "skills"}),
    ("preferences", {"content": "Prefers concise bullet-point answers with code examples.", "category": "style"}),
    ("goals",       {"content": "Building a real-time recommendation system using collaborative filtering.", "category": "project"}),
    ("tools",       {"content": "Uses VS Code, pytest, FastAPI, PostgreSQL, Redis.", "category": "tooling"}),
]

for key, value in initial_facts:
    semantic_store.put(("alice", "semantic"), key, value)

print(f"  Seeded {len(initial_facts)} semantic facts")


class SemanticFact(TypedDict):
    key:      str
    content:  str
    category: str


@tool
def save_fact(fact: SemanticFact, runtime: ToolRuntime[UserContext]) -> str:
    """Save a semantic fact about the user (skill, preference, project info)."""
    assert runtime.store is not None
    uid = runtime.context.user_id
    runtime.store.put((uid, "semantic"), fact["key"], {
        "content": fact["content"], "category": fact["category"]
    })
    print(f"  [Semantic] [{fact['key']}]: {fact['content'][:50]}")
    return f"Fact saved: {fact['content'][:50]}"


@tool
def recall_facts(query: str, runtime: ToolRuntime[UserContext]) -> str:
    """Retrieve semantic facts relevant to the current topic."""
    assert runtime.store is not None
    uid     = runtime.context.user_id
    results = runtime.store.search((uid, "semantic"), query=query, limit=4)
    if not results:
        return "No relevant facts found."
    return "Known facts:\n" + "\n".join(
        f"  [{r.value['category']}] {r.value['content']}" for r in results
    )


semantic_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[save_fact, recall_facts],
    store=semantic_store,
    context_schema=UserContext,
    system_prompt=(
        "You are an assistant with semantic memory. "
        "Use recall_facts before answering. Use save_fact when users share info."
    ),
)

r = semantic_agent.invoke(
    {"messages": [{"role": "user", "content": "What PyTorch features suit my recommendation system?"}]},
    context=UserContext(user_id="alice"),
)
print(f"\n  Semantic-aware answer: {r['messages'][-1].content[:200]}")


# ════════════════════════════════════════════════════════════════════
# PART 2: EPISODIC MEMORY — past events and interactions
# Stored under: (user_id, "episodes") namespace
# Key:   unique event ID per interaction
# Value: {"summary", "topic", "event_type", "timestamp"}
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Episodic Memory (interaction history) ─────────────────")

episodic_store = InMemoryStore(index=IndexConfig(embed=embed_fn, dims=1536))

past_events = [
    ("Explained async/await and asyncio.gather() in Python",            "python_async",   "learning"),
    ("Debugged a LangGraph StateGraph edge condition bug",               "langgraph_debug","debugging"),
    ("Designed a FAISS-based vector similarity search pipeline",         "faiss_design",  "architecture"),
    ("Reviewed FastAPI endpoint for JWT authentication vulnerability",   "security_review","code_review"),
]

for i, (summary, topic, event_type) in enumerate(past_events, 1):
    episodic_store.put(("alice", "episodes"), f"evt_{uuid.uuid4().hex[:6]}", {
        "summary": summary, "topic": topic, "event_type": event_type,
        "session": i, "timestamp": f"2024-0{i}-15",
    })

print(f"  Seeded {len(past_events)} past episodes")


class Episode(TypedDict):
    summary:    str
    topic:      str
    event_type: str


@tool
def save_episode(episode: Episode, runtime: ToolRuntime[UserContext]) -> str:
    """Save an episodic memory of this interaction. Call after meaningful exchanges."""
    assert runtime.store is not None
    uid = runtime.context.user_id
    event_id = f"evt_{uuid.uuid4().hex[:8]}"
    runtime.store.put((uid, "episodes"), event_id, {
        **dict(episode), "timestamp": datetime.now().isoformat(),
    })
    print(f"  [Episodic] {event_id}: {episode['summary'][:50]}")
    return f"Episode saved: {episode['summary'][:50]}"


@tool
def recall_past_interactions(query: str, runtime: ToolRuntime[UserContext]) -> str:
    """Search past interaction history for relevant episodes."""
    assert runtime.store is not None
    uid     = runtime.context.user_id
    results = runtime.store.search((uid, "episodes"), query=query, limit=3)
    if not results:
        return "No relevant past interactions found."
    return "Past interactions:\n" + "\n".join(
        f"  [{r.value.get('timestamp','?')[:10]}] {r.value['summary'][:80]}"
        for r in results
    )


episodic_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[save_episode, recall_past_interactions],
    store=episodic_store,
    context_schema=UserContext,
    system_prompt=(
        "You have episodic memory. Use recall_past_interactions before answering — "
        "build on past context instead of repeating. Use save_episode after meaningful exchanges."
    ),
)

r2 = episodic_agent.invoke(
    {"messages": [{"role": "user", "content": "Can you help me debug a LangGraph issue?"}]},
    context=UserContext(user_id="alice"),
)
print(f"\n  Episodic-aware answer: {r2['messages'][-1].content[:200]}")

episodes = episodic_store.search(("alice", "episodes"), limit=20)
print(f"  Total episodes: {len(episodes)}")


# ════════════════════════════════════════════════════════════════════
# PART 3: PROCEDURAL MEMORY — rules and behaviors
# Stored under: (user_id, "rules") namespace
# Key:   unique rule ID
# Value: {"rule", "priority", "source"}
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Procedural Memory (rules and behaviors) ───────────────")

procedural_store = InMemoryStore()

initial_rules = [
    ("rule_1", {"rule": "Always include type hints in Python code examples.", "priority": 9, "source": "explicit_instruction"}),
    ("rule_2", {"rule": "Use pytest fixtures when showing test examples.",     "priority": 8, "source": "explicit_instruction"}),
    ("rule_3", {"rule": "Prefer async/await over threading patterns.",         "priority": 7, "source": "inferred"}),
]

for key, value in initial_rules:
    procedural_store.put(("alice", "rules"), key, value)

print(f"  Seeded {len(initial_rules)} procedural rules")


class BehaviorRule(TypedDict):
    rule:     str
    priority: int
    source:   str


@tool
def save_behavior_rule(rule: BehaviorRule, runtime: ToolRuntime[UserContext]) -> str:
    """Save a behavioral rule. Call when user gives explicit instructions or corrects behavior."""
    assert runtime.store is not None
    uid     = runtime.context.user_id
    rule_id = f"rule_{uuid.uuid4().hex[:6]}"
    runtime.store.put((uid, "rules"), rule_id, dict(rule))
    print(f"  [Procedural] P{rule['priority']}: {rule['rule'][:60]}")
    return f"Rule saved: {rule['rule'][:60]}"


@tool
def get_active_rules(runtime: ToolRuntime[UserContext]) -> str:
    """Retrieve all active behavioral rules sorted by priority. Call at conversation start."""
    assert runtime.store is not None
    uid   = runtime.context.user_id
    rules = runtime.store.search((uid, "rules"), limit=20)
    if not rules:
        return "No behavior rules set."
    sorted_rules = sorted(rules, key=lambda r: r.value.get("priority", 0), reverse=True)
    return "Active rules:\n" + "\n".join(
        f"  [P{r.value['priority']}] {r.value['rule']}" for r in sorted_rules
    )


procedural_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[save_behavior_rule, get_active_rules],
    store=procedural_store,
    context_schema=UserContext,
    system_prompt=(
        "At the start of EVERY conversation, call get_active_rules to load behavioral rules. "
        "Follow all rules. When user gives instructions, call save_behavior_rule."
    ),
)

r3 = procedural_agent.invoke(
    {"messages": [{"role": "user", "content":
        "Always show error handling (try/except) in every code example."}]},
    context=UserContext(user_id="alice"),
)
print(f"\n  Procedural agent: {r3['messages'][-1].content[:150]}")

all_rules = procedural_store.search(("alice", "rules"), limit=20)
print(f"  Active rules ({len(all_rules)} total):")
for r in sorted(all_rules, key=lambda x: x.value.get("priority", 0), reverse=True):
    print(f"    [P{r.value['priority']}] {r.value['rule'][:70]}")


# ════════════════════════════════════════════════════════════════════
# PART 4: MEMORY EXTRACTION — LLM extracts facts from text
# Background extraction after a conversation, rather than relying
# on the agent to proactively call save tools.
# ════════════════════════════════════════════════════════════════════

print("\n── 4. Memory Extraction from Conversation Text ──────────────")

import re

EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    """Extract structured memories from this conversation.

Conversation:
{conversation}

Extract and respond ONLY with valid JSON (no markdown):
{{
  "semantic_facts": [
    {{"key": "descriptive_key", "content": "fact", "category": "identity|skills|style|project|tooling"}}
  ],
  "episodic_summary": "one-sentence summary of the interaction",
  "procedural_rules": [
    {{"rule": "behavior rule", "priority": 7}}
  ]
}}"""
)

extraction_chain = EXTRACTION_PROMPT | llm | StrOutputParser()

conversation_sample = """\
User: Hi, I work on a FastAPI project and need help with dependency injection.
      I'm a backend developer with 5 years of Python and Go experience.
      I prefer short answers with practical code examples.
Assistant: I'll keep my answers concise with code. FastAPI's Depends() is your friend..."""

raw = extraction_chain.invoke({"conversation": conversation_sample})
cleaned = re.sub(r'```json\s*|\s*```', '', raw.strip()).strip()

try:
    extracted = json.loads(cleaned)
    print(f"\n  Extracted semantic facts: {len(extracted.get('semantic_facts',[]))}")
    for f in extracted.get("semantic_facts", []):
        print(f"    [{f['category']}] {f['key']}: {f['content'][:60]}")
    print(f"\n  Episodic summary: {extracted.get('episodic_summary','?')[:80]}")
    print(f"\n  Procedural rules: {len(extracted.get('procedural_rules',[]))}")
    for r in extracted.get("procedural_rules", []):
        print(f"    [P{r['priority']}] {r['rule'][:60]}")

    # Persist extracted memories
    ext_store = InMemoryStore()
    for f in extracted.get("semantic_facts", []):
        ext_store.put(("new_user", "semantic"), f["key"], {
            "content": f["content"], "category": f["category"]
        })
    print(f"\n  Persisted {len(extracted.get('semantic_facts',[]))} extracted facts")

except json.JSONDecodeError as e:
    print(f"  Extraction parse error: {e}\n  Raw: {raw[:100]}")


# ════════════════════════════════════════════════════════════════════
# PART 5: CROSS-MEMORY SYNTHESIS
# Build a personalized system prompt by combining all three memory types.
# This is how you create a deeply personalized agent.
# ════════════════════════════════════════════════════════════════════

print("\n── 5. Cross-Memory Synthesis (personalized system prompt) ───")

unified_store = InMemoryStore(index=IndexConfig(embed=embed_fn, dims=1536))

# Populate all three memory types
unified_store.put(("alice", "semantic"),  "profile", {"content": "Senior Python/ML dev, expert level.", "category": "identity"})
unified_store.put(("alice", "semantic"),  "style",   {"content": "Prefers concise answers with typed Python examples.", "category": "style"})
unified_store.put(("alice", "episodes"), "evt_1",   {"summary": "Discussed FAISS vector search design.", "timestamp": "2024-01-10"})
unified_store.put(("alice", "episodes"), "evt_2",   {"summary": "Helped debug an async race condition.", "timestamp": "2024-01-15"})
unified_store.put(("alice", "rules"),    "rule_1",  {"rule": "Always include type hints.", "priority": 9})
unified_store.put(("alice", "rules"),    "rule_2",  {"rule": "Show pytest examples for testing.", "priority": 7})


def build_personalized_prompt(user_id: str, query: str) -> str:
    """Synthesize a rich system prompt from all three memory types."""
    # Semantic: most relevant facts for this query
    sem_results  = unified_store.search((user_id, "semantic"), query=query, limit=3)
    semantic_ctx = "\n".join(f"  - {r.value['content']}" for r in sem_results)

    # Episodic: most relevant past interactions
    epi_results  = unified_store.search((user_id, "episodes"), query=query, limit=2)
    episodic_ctx = "\n".join(f"  - {r.value['summary']}" for r in epi_results)

    # Procedural: all rules (sorted by priority)
    all_rules    = unified_store.search((user_id, "rules"), limit=20)
    rules_sorted = sorted(all_rules, key=lambda r: r.value.get("priority", 0), reverse=True)
    rules_ctx    = "\n".join(f"  - [P{r.value['priority']}] {r.value['rule']}" for r in rules_sorted)

    return f"""You are a personalized assistant for user {user_id!r}.

About the user:
{semantic_ctx}

Past interactions (relevant to current query):
{episodic_ctx}

Behavioral rules (follow these):
{rules_ctx}"""


query = "How do I write unit tests for an async FastAPI endpoint?"
personalized_prompt = build_personalized_prompt("alice", query)
print(f"\n  Personalized system prompt preview:\n{personalized_prompt[:400]}")

# Use the personalized prompt
from langchain.chat_models import init_chat_model
fast_llm = init_chat_model("openai:gpt-4o-mini")
response = fast_llm.invoke(
    [{"role": "system", "content": personalized_prompt},
     {"role": "user",   "content": query}]
)
print(f"\n  Personalized answer: {response.content[:250]}")

print("\n" + "=" * 60)
print("Memory Types Summary:")
print("  Semantic:    (uid,'semantic') → facts about user (searchable)")
print("  Episodic:    (uid,'episodes') → unique key per event → history")
print("  Procedural:  (uid,'rules')    → priority-ordered rules")
print("  Extraction:  LLM extracts structured memories from conversation")
print("  Synthesis:   combine all 3 → rich personalized system prompt")
print("=" * 60)
print("\nAll three types complement each other:")
print("  Semantic  = WHO the user is")
print("  Episodic  = WHAT happened before")
print("  Procedural = HOW the agent should behave")
print("\n✅ Memory types demo complete.")
