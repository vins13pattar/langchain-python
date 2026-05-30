"""
04_skills.py
=============
Demonstrates the SKILLS multi-agent pattern — a single agent loads
specialized prompts and knowledge on-demand. The agent stays in control
while skills provide domain-specific context as needed.

Concepts covered:
  - Skills as tools that return specialized context/instructions
  - On-demand loading — skills are only loaded when needed
  - Context injection via tool results that become conversation history
  - Skills with ToolRuntime for state-aware loading
  - Skills composition — loading multiple skills for complex tasks
  - Stateful skills — skill context persists across turns (reuse)
  - Progressive disclosure — skills can load sub-skills

Key difference from Subagents:
  - Subagents:   create a NEW agent with clean context
  - Skills:      the SAME agent loads more context into its window
  
Tradeoffs vs Subagents:
  - Skills: fewer model calls but higher token usage per call
  - Subagents: more model calls but context isolation (lower tokens per call)
  - Skills better for: few domains, repeat requests, conversational flow
  - Subagents better for: many domains, parallel work, context isolation
"""

from dotenv import load_dotenv

from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

load_dotenv()

print("=" * 60)
print("Skills Pattern")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# SKILL KNOWLEDGE BASE
# In production: load from files, vector DBs, or APIs
# ════════════════════════════════════════════════════════════════════

SKILLS = {
    "python": {
        "name": "Python Programming",
        "context": """
## Python Expert Context

Key strengths: readable syntax, rich stdlib, huge ecosystem (PyPI).
Performance: GIL limits true parallelism; use multiprocessing/async for I/O.
Best practices:
- Use type hints (Python 3.10+: X | Y union syntax)
- Prefer f-strings over .format()
- Virtual envs: venv or uv
- Testing: pytest with fixtures
- Package management: pyproject.toml + uv or pip

Web: FastAPI (async, auto-docs), Django (batteries-included), Flask (micro)
Data science: NumPy, Pandas, scikit-learn, PyTorch/TensorFlow
Async: asyncio, httpx for async HTTP
        """,
        "tools": ["run_python_snippet", "search_pypi"],
    },
    "kubernetes": {
        "name": "Kubernetes / DevOps",
        "context": """
## Kubernetes Expert Context

Core objects: Pod, Deployment, Service, ConfigMap, Secret, Ingress.
Key concepts: Namespaces, RBAC, NetworkPolicy, HPA (autoscaling).
kubectl essentials:
  kubectl get pods -n <ns>
  kubectl describe pod <name>
  kubectl logs <pod> --previous
  kubectl exec -it <pod> -- bash
Deployment strategy: RollingUpdate (default), Recreate.
Helm: package manager — helm install <name> <chart>.
Troubleshooting: check events (kubectl get events), resource limits, image pull errors.
        """,
        "tools": ["run_kubectl", "search_helm_charts"],
    },
    "security": {
        "name": "Security & Compliance",
        "context": """
## Security Expert Context

OWASP Top 10: Injection, Broken Auth, XSS, IDOR, Misconfig, etc.
API security: authenticate (JWT/OAuth2), authorize (RBAC/ABAC), rate limit, validate input.
Secrets management: never hardcode — use env vars, Vault, AWS Secrets Manager.
Dependency security: scan with pip-audit, snyk, dependabot.
Compliance: GDPR (data minimization, right to erasure), SOC2, PCI-DSS.
Encryption: TLS 1.3 in transit, AES-256 at rest. Key rotation policy.
        """,
        "tools": ["run_security_scan", "check_cve"],
    },
    "sql": {
        "name": "SQL & Databases",
        "context": """
## SQL/Database Expert Context

Query optimization: use EXPLAIN ANALYZE, avoid SELECT *, add indexes.
Indexing: B-tree for equality/range, GIN for arrays/JSONB, partial indexes.
Common patterns:
- CTE (WITH): readable, non-recursive CTEs are inline views
- Window functions: ROW_NUMBER(), RANK(), LAG(), LEAD()
- Upsert: INSERT ... ON CONFLICT DO UPDATE
N+1 problem: use JOINs or batch loading instead of loops.
Transactions: ACID, isolation levels (READ COMMITTED default in PG).
Migrations: always reversible, idempotent, test on staging first.
        """,
        "tools": ["execute_sql_query", "explain_query"],
    },
}


# ════════════════════════════════════════════════════════════════════
# PART 1: BASIC SKILL LOADING
# Skills are tools that return specialist context as a string.
# That string becomes a ToolMessage in the conversation history,
# giving the agent domain expertise for subsequent responses.
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Basic Skill Loading ────────────────────────────────────")

@tool
def load_python_skill() -> str:
    """Load Python programming expertise and best practices into context."""
    skill = SKILLS["python"]
    print(f"  [Skill] Loading: {skill['name']}")
    return skill["context"].strip()


@tool
def load_kubernetes_skill() -> str:
    """Load Kubernetes and DevOps expertise into context."""
    skill = SKILLS["kubernetes"]
    print(f"  [Skill] Loading: {skill['name']}")
    return skill["context"].strip()


@tool
def load_security_skill() -> str:
    """Load security and compliance expertise into context."""
    skill = SKILLS["security"]
    print(f"  [Skill] Loading: {skill['name']}")
    return skill["context"].strip()


@tool
def load_sql_skill() -> str:
    """Load SQL and database optimization expertise into context."""
    skill = SKILLS["sql"]
    print(f"  [Skill] Loading: {skill['name']}")
    return skill["context"].strip()


skills_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[load_python_skill, load_kubernetes_skill, load_security_skill, load_sql_skill],
    system_prompt=(
        "You are a technical assistant. Before answering domain-specific questions, "
        "load the appropriate skill to get expert context. "
        "Available skills: Python, Kubernetes, Security, SQL. "
        "Load the relevant skill first, then answer using that expertise."
    ),
)

result1 = skills_agent.invoke({
    "messages": [{"role": "user", "content":
        "How should I handle database connection pooling in a FastAPI app?"}]
})
print(f"\nResult: {result1['messages'][-1].content[:300]}")


# ════════════════════════════════════════════════════════════════════
# PART 2: DYNAMIC SKILL SELECTOR
# Single `load_skill` tool that accepts a skill name — reduces
# number of tools and enables dynamic skill registries.
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Dynamic Skill Selector ────────────────────────────────")

@tool
def load_skill(skill_name: str) -> str:
    """Load a specific skill by name to gain domain expertise.

    Available skills: python, kubernetes, security, sql
    """
    skill = SKILLS.get(skill_name.lower())
    if not skill:
        available = list(SKILLS.keys())
        return f"Skill '{skill_name}' not found. Available: {available}"
    print(f"  [Skill] Loading: {skill['name']}")
    return f"## {skill['name']} Expertise Loaded\n{skill['context'].strip()}"


@tool
def list_available_skills() -> str:
    """List all available skills and their descriptions."""
    lines = [f"- {name}: {info['name']}" for name, info in SKILLS.items()]
    return "Available skills:\n" + "\n".join(lines)


dynamic_skills_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[list_available_skills, load_skill],
    system_prompt=(
        "You are a technical assistant. For domain-specific questions, "
        "first use list_available_skills to see what's available, "
        "then use load_skill to load the relevant expertise. "
        "You can load multiple skills for cross-domain questions."
    ),
)

result2 = dynamic_skills_agent.invoke({
    "messages": [{"role": "user", "content":
        "How do I securely deploy a Python API to Kubernetes?"}]
})
print(f"\nResult: {result2['messages'][-1].content[:300]}")


# ════════════════════════════════════════════════════════════════════
# PART 3: STATEFUL SKILLS — reuse across turns
# Once a skill is loaded into conversation history, the agent
# reuses it in subsequent turns without reloading (saves a model call).
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Stateful Skills (reuse across turns) ──────────────────")

stateful_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[load_skill],
    checkpointer=MemorySaver(),    # persist conversation + loaded skills
    system_prompt=(
        "You are a technical assistant. Load skills as needed. "
        "Once a skill is loaded, it persists in conversation history — "
        "don't reload it unless asked for a different domain."
    ),
)

cfg = {"configurable": {"thread_id": "skills-stateful"}}

# Turn 1: loads Python skill
r1 = stateful_agent.invoke(
    {"messages": [{"role": "user", "content": "What's the best way to write async code in Python?"}]},
    config=cfg,
)
print(f"\nTurn 1: {r1['messages'][-1].content[:120]}")

# Turn 2: skill already in history — no reload needed (saves 1 model call)
r2 = stateful_agent.invoke(
    {"messages": [{"role": "user", "content": "How do I use asyncio.gather()?"}]},
    config=cfg,
)
print(f"Turn 2 (reused skill): {r2['messages'][-1].content[:120]}")

# Turn 3: different domain — loads SQL skill
r3 = stateful_agent.invoke(
    {"messages": [{"role": "user", "content": "Now explain SQL window functions."}]},
    config=cfg,
)
print(f"Turn 3 (new skill): {r3['messages'][-1].content[:120]}")


# ════════════════════════════════════════════════════════════════════
# PART 4: SKILLS + STORE — persistent skill cache
# Store loaded skill content in long-term memory so it survives
# across sessions without reloading.
# ════════════════════════════════════════════════════════════════════

print("\n── 4. Skills with Store (cross-session cache) ───────────────")

skill_store = InMemoryStore()

@tool
def load_skill_cached(skill_name: str, runtime: ToolRuntime) -> str:
    """Load a skill, checking Store cache first to avoid repeated loading."""
    store  = runtime.store if runtime else None
    cached = store.get(("skills",), skill_name) if store else None

    if cached:
        print(f"  [Skill] Cache HIT: {skill_name}")
        return f"(cached) {cached.value['context'][:100]}..."

    skill = SKILLS.get(skill_name.lower())
    if not skill:
        return f"Skill '{skill_name}' not found."

    context = f"## {skill['name']}\n{skill['context'].strip()}"
    if store:
        store.put(("skills",), skill_name, {"context": context, "name": skill["name"]})
        print(f"  [Skill] Cache MISS — loaded and cached: {skill_name}")
    return context


cached_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[load_skill_cached],
    store=skill_store,
    checkpointer=MemorySaver(),
    system_prompt="Load skills before answering domain questions.",
)

cfg2 = {"configurable": {"thread_id": "skills-cached"}}
for msg in [
    "What are Python type hints?",     # MISS → loads and caches
    "What is Python's GIL?",           # HIT  → reads from cache
]:
    r = cached_agent.invoke(
        {"messages": [{"role": "user", "content": msg}]},
        config=cfg2,
    )
    print(f"\n  Q: {msg}")
    print(f"  A: {r['messages'][-1].content[:100]}")

print("\n" + "═" * 60)
print("Skills Pattern Summary:")
print("  Skills = tools that return specialist context as strings")
print("  Context becomes ToolMessage → part of agent conversation")
print("  Single load_skill(name) → dynamic registry, fewer tools")
print("  Stateful (MemorySaver) → skill loaded once, reused every turn")
print("  Store cache → persists skills across sessions (no reload)")
print("  Multiple skills loadable → cross-domain expertise in one agent")
print("  Trade-off: fewer model calls vs. higher token usage per call")
print("═" * 60)
print("\n✅ Skills pattern demo complete.")
