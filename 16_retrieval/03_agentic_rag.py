"""
03_agentic_rag.py
==================
Demonstrates Agentic RAG — an LLM agent decides WHEN and HOW to
retrieve information during its reasoning. The agent has retrieval
as a tool and uses it only when needed.

Concepts covered:
  - Retrieval as a @tool — agent calls it when needed
  - fetch_url tool — agentic web retrieval (URL fetching)
  - Vector store as a @tool — semantic search over local knowledge base
  - Multi-source agentic RAG — web + local KB + SQL
  - llms.txt agentic documentation assistant (from docs example)
  - When to use agentic vs 2-step RAG
  - Agent with retrieval memory — persisting retrieved context

Key difference from 2-Step RAG:
  - 2-Step:   ALWAYS retrieves before generating (1 LLM call)
  - Agentic:  Agent DECIDES when to retrieve (multiple LLM calls)
  - 2-Step:   predictable, fast, suitable for known retrieval need
  - Agentic:  flexible, handles "I already know this" cases
"""

import os
import requests
from typing import Optional
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model

load_dotenv()

print("=" * 60)
print("Agentic RAG — LLM Agent with Retrieval Tools")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# BUILD KNOWLEDGE BASE
# ════════════════════════════════════════════════════════════════════

DOCS = [
    Document(page_content="LangChain is a Python and JavaScript framework for building LLM applications. It provides abstractions for agents, tools, memory, and chains. Version 1.x introduced create_agent as the primary agent builder.", metadata={"source": "langchain_docs", "topic": "langchain"}),
    Document(page_content="create_agent() is LangChain's primary function for creating agents. Parameters: model (str or LLM), tools (list), system_prompt (str), checkpointer, store, middleware. Returns a compiled LangGraph graph.", metadata={"source": "create_agent_api", "topic": "langchain"}),
    Document(page_content="LangGraph is built for stateful, multi-actor applications. Key concepts: StateGraph, AgentState, Command, Send, interrupt. LangGraph underpins create_agent and provides streaming, persistence, and interrupt support.", metadata={"source": "langgraph_docs", "topic": "langgraph"}),
    Document(page_content="FAISS (Facebook AI Similarity Search) enables efficient vector similarity search. It supports multiple index types: Flat (exact), IVF (approximate, faster), HNSW (graph-based). Use FAISS.from_documents() to index LangChain Documents.", metadata={"source": "faiss_guide", "topic": "vectorstore"}),
    Document(page_content="OpenAI embeddings: text-embedding-3-small (1536 dims, cost-effective), text-embedding-3-large (3072 dims, highest quality), text-embedding-ada-002 (1536 dims, legacy). Use embed_query() for single text, embed_documents() for batches.", metadata={"source": "openai_embeddings", "topic": "embeddings"}),
    Document(page_content="RAG architectures: 2-Step RAG (always retrieve, predictable, fast), Agentic RAG (agent decides when to retrieve, flexible), Hybrid RAG (retrieve + validate + self-correct, highest quality). Choose based on latency vs. flexibility tradeoffs.", metadata={"source": "rag_architectures", "topic": "rag"}),
    Document(page_content="Middleware in LangChain agents wraps model calls and tool calls. Built-in: HumanInTheLoopMiddleware, ModelCallLimit, ToolRetry, PIIDetection, LLMToolSelector, ModelFallback. Custom middleware uses @wrap_model_call or @wrap_tool_call decorators.", metadata={"source": "middleware_guide", "topic": "middleware"}),
    Document(page_content="Memory in LangChain: Short-term memory uses MemorySaver (in-memory) or SqliteSaver (persistent). Long-term memory uses Store (InMemoryStore or SqliteStore). Messages are stored per thread_id in the configurable dict.", metadata={"source": "memory_guide", "topic": "memory"}),
]

splitter  = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
chunks    = splitter.split_documents(DOCS)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vs         = FAISS.from_documents(chunks, embeddings)

print(f"\nKnowledge base: {len(DOCS)} docs → {len(chunks)} chunks")


# ════════════════════════════════════════════════════════════════════
# PART 1: BASIC AGENTIC RAG — retrieval as a tool
# The agent calls search_knowledge_base() only when it needs info.
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Basic Agentic RAG (vector search as tool) ─────────────")

@tool
def search_knowledge_base(query: str, k: int = 3) -> str:
    """Search the knowledge base for relevant information.
    Use this when you need specific technical information about LangChain,
    LangGraph, embeddings, vector stores, or RAG architectures.
    """
    docs = vs.similarity_search(query, k=k)
    if not docs:
        return "No relevant information found in the knowledge base."

    results = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", f"doc_{i}")
        results.append(f"[{src}]\n{doc.page_content}")

    print(f"  [KB Search] query={query!r} → {len(docs)} results")
    return "\n\n".join(results)


rag_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_knowledge_base],
    system_prompt=(
        "You are a technical assistant with access to a LangChain knowledge base. "
        "Use search_knowledge_base to find information when needed. "
        "If you already know the answer confidently, you don't need to search. "
        "Always cite your sources when using retrieved information."
    ),
)

queries = [
    "What is 2 + 2?",                          # Agent should answer without retrieval
    "What parameters does create_agent take?",  # Agent should search KB
    "How does FAISS indexing work?",            # Agent should search KB
]

for q in queries:
    result = rag_agent.invoke({"messages": [{"role": "user", "content": q}]})
    print(f"\n  Q: {q}")
    print(f"  A: {result['messages'][-1].content[:180]}")


# ════════════════════════════════════════════════════════════════════
# PART 2: FETCH_URL — Web Retrieval Tool
# The agent fetches live web content for up-to-date information.
# Mirrors the exact pattern from the official LangChain docs.
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Web Retrieval Tool (fetch_url) ────────────────────────")

@tool
def fetch_url(url: str) -> str:
    """Fetch text content from a URL. Use for up-to-date web information.
    Returns the first 2000 characters of the page content.
    """
    print(f"  [WebFetch] {url}")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (educational RAG demo)"}
        response = requests.get(url, timeout=10.0, headers=headers)
        response.raise_for_status()
        # Clean the text (strip HTML tags simply)
        text = response.text
        # Remove obvious HTML tags for readability
        import re
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:2000]
    except Exception as e:
        return f"Error fetching {url}: {e}"


web_rag_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[fetch_url],
    system_prompt=(
        "You are a helpful assistant that can fetch web pages when needed. "
        "Use fetch_url to get up-to-date information from the web. "
        "Cite URLs when using fetched content. Only fetch if necessary."
    ),
)

# The agent can fetch content from any URL
result2 = web_rag_agent.invoke({
    "messages": [{"role": "user", "content":
        "What is the current version of Python? Check python.org if needed."}]
})
print(f"\n  Q: What is the current version of Python?")
print(f"  A: {result2['messages'][-1].content[:200]}")


# ════════════════════════════════════════════════════════════════════
# PART 3: llms.txt DOCUMENTATION ASSISTANT
# Mirrors the extended example from the LangChain docs:
# Agent loads llms.txt index, then fetches specific doc pages.
# ════════════════════════════════════════════════════════════════════

print("\n── 3. llms.txt Documentation Assistant ──────────────────────")

ALLOWED_DOMAINS = ["https://langchain-ai.github.io/", "https://docs.langchain.com/"]
LLMS_TXT_URL    = "https://langchain-ai.github.io/langgraph/llms.txt"

@tool
def fetch_documentation(url: str) -> str:
    """Fetch and return documentation content from an allowed domain.
    Only allowed domains: langchain-ai.github.io, docs.langchain.com
    """
    if not any(url.startswith(d) for d in ALLOWED_DOMAINS):
        return f"Error: URL not allowed. Must start with: {ALLOWED_DOMAINS}"
    print(f"  [DocFetch] {url}")
    try:
        response = requests.get(url, timeout=15.0)
        response.raise_for_status()
        import re
        text = re.sub(r'<[^>]+>', ' ', response.text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:3000]
    except Exception as e:
        return f"Error fetching documentation: {e}"


# Fetch llms.txt index upfront (no LLM call needed for this)
try:
    llms_txt_content = requests.get(LLMS_TXT_URL, timeout=10).text[:3000]
    print(f"  Loaded llms.txt index ({len(llms_txt_content)} chars)")
except Exception:
    llms_txt_content = "Documentation index unavailable."

docs_agent_prompt = f"""
You are an expert LangGraph technical assistant.
For questions about LangGraph APIs, usage, or behavior — consult the documentation.

You can access official documentation from:
{llms_txt_content[:500]}...

Instructions:
1. If unsure about LangGraph specifics, use fetch_documentation to get the answer.
2. Always cite the documentation URL you consulted.
3. Do not fetch URLs outside the allowed domains.
"""

docs_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[fetch_documentation],
    system_prompt=docs_agent_prompt,
)

# Note: API call to fetch docs may be slow/fail in demo — wrapped in try/except
try:
    doc_result = docs_agent.invoke({
        "messages": [{"role": "user", "content":
            "How do I use interrupts in LangGraph to pause agent execution?"}]
    })
    print(f"\n  Q: How do I use interrupts in LangGraph?")
    print(f"  A: {doc_result['messages'][-1].content[:250]}")
except Exception as e:
    print(f"\n  (Documentation fetch demo skipped: {e})")


# ════════════════════════════════════════════════════════════════════
# PART 4: MULTI-SOURCE AGENTIC RAG
# Agent has access to multiple retrieval tools:
#   1. Local vector store KB
#   2. Web fetcher
#   3. Simulated SQL query
# It chooses the right source for each query.
# ════════════════════════════════════════════════════════════════════

print("\n── 4. Multi-Source Agentic RAG ──────────────────────────────")

@tool
def query_database(sql_query: str) -> str:
    """Query the internal database for structured data.
    Returns sales, usage, or operational data.
    Supports simple SELECT queries on: users, events, subscriptions tables.
    """
    print(f"  [DB Query] {sql_query}")
    # Simulated database results
    mock_data = {
        "users": "| id | name | plan | created_at |\n|1|Alice|pro|2024-01|\n|2|Bob|free|2024-02|",
        "events": "| date | event | count |\n|2024-01|signup|152|\n|2024-01|activation|89|",
        "subscriptions": "| plan | count | mrr |\n|pro|450|$9000|\n|enterprise|12|$36000|",
    }
    table = next((v for k, v in mock_data.items() if k in sql_query.lower()), "No data found.")
    return f"Query result:\n{table}"


multi_source_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_knowledge_base, fetch_url, query_database],
    system_prompt=(
        "You are an enterprise assistant with three information sources:\n"
        "1. search_knowledge_base: Technical docs on LangChain/RAG (local KB)\n"
        "2. fetch_url: Live web pages for current information\n"
        "3. query_database: Internal business data (users, events, subscriptions)\n\n"
        "Choose the most appropriate source for each query. "
        "Combine sources if needed. Cite sources in your answer."
    ),
)

multi_queries = [
    "How many pro subscribers do we have and what's our MRR?",
    "What's the difference between FAISS and Chroma for vector storage?",
]

for q in multi_queries:
    result = multi_source_agent.invoke({"messages": [{"role": "user", "content": q}]})
    print(f"\n  Q: {q}")
    print(f"  A: {result['messages'][-1].content[:200]}")


# ════════════════════════════════════════════════════════════════════
# PART 5: AGENTIC RAG WITH MULTI-HOP REASONING
# Agent can make multiple retrieval calls in sequence,
# building understanding progressively.
# ════════════════════════════════════════════════════════════════════

print("\n── 5. Multi-Hop Agentic RAG ─────────────────────────────────")

multi_hop_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_knowledge_base],
    system_prompt=(
        "You are a research agent. For complex questions, make multiple "
        "search_knowledge_base calls to gather complete information from "
        "different angles before synthesizing your final answer."
    ),
)

complex_query = ("Compare 2-step RAG vs agentic RAG, explaining their architectures "
                 "and when to use each. Also explain how memory fits into each approach.")
result5 = multi_hop_agent.invoke({
    "messages": [{"role": "user", "content": complex_query}]
})
print(f"\n  Complex Q: {complex_query[:80]}...")
print(f"  A: {result5['messages'][-1].content[:300]}")

print("\n" + "═" * 60)
print("Agentic RAG Summary:")
print("  @tool search_knowledge_base → agent calls WHEN needed")
print("  @tool fetch_url             → live web content retrieval")
print("  @tool fetch_documentation   → domain-restricted doc fetcher")
print("  @tool query_database        → structured data source")
print("  Multi-source: agent picks the right tool per query")
print("  Multi-hop:    agent chains multiple retrieval calls")
print("  vs 2-Step: more flexible, but more LLM calls + variable latency")
print("═" * 60)
print("\n✅ Agentic RAG demo complete.")
