"""
05_full_retrieval_showcase.py
==============================
Production-ready showcase: a Smart Q&A Assistant that intelligently
routes queries to the best RAG architecture.

System design:
  ┌────────────────────────────────────────────────────────────────┐
  │                   Smart Q&A Assistant                           │
  ├────────────────────────────────────────────────────────────────┤
  │                                                                │
  │  Entry: user question                                          │
  │     ↓                                                          │
  │  [ROUTER] Classify → direct | two_step | agentic | hybrid      │
  │     ↓                                                          │
  │  [PATH A] Direct answer (no retrieval)                         │
  │     ─ General knowledge / math / greetings                     │
  │                                                                │
  │  [PATH B] 2-Step RAG                                           │
  │     retrieve(query, k=4) → format → generate                   │
  │     ─ Simple factual queries with clear retrieval need         │
  │                                                                │
  │  [PATH C] Agentic RAG                                          │
  │     create_agent(tools=[search_kb, fetch_url]) → reason        │
  │     ─ Multi-source, multi-hop, unknown structure               │
  │                                                                │
  │  [PATH D] Hybrid RAG                                           │
  │     enhance → validate → generate → self-correct               │
  │     ─ Ambiguous queries, high-accuracy requirements            │
  │                                                                │
  │  → Return answer + metadata (architecture, sources, quality)  │
  └────────────────────────────────────────────────────────────────┘

Scenarios:
  1. Direct: "What's the capital of France?" → no retrieval
  2. 2-Step: "What is RAG?" → retrieve → answer
  3. Agentic: "Compare transformers vs RNNs AND check current benchmarks"
  4. Hybrid: "Explain the trade-offs between different vector databases"
  5. Multi-turn: Follow-up questions using conversation history
"""

import json
import re
import time
import requests
from dataclasses import dataclass, field
from typing import Literal
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

print("=" * 60)
print("Smart Q&A Assistant — Full Retrieval / RAG Showcase")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE — comprehensive AI/ML technical docs
# ════════════════════════════════════════════════════════════════════

KB_DOCS = [
    Document(page_content="Retrieval-Augmented Generation (RAG) grounds LLM responses in external knowledge retrieved at query time. The pipeline: document loading → splitting → embedding → vector storage → retrieval → generation. RAG reduces hallucination and keeps answers up-to-date.", metadata={"source": "rag_fundamentals", "domain": "rag"}),
    Document(page_content="2-Step RAG: always retrieves before generating. Predictable latency (1 LLM call for retrieval, 1 for generation). Best for FAQs, documentation Q&A, and cases where retrieval is always needed. Weaknesses: retrieves even when unnecessary.", metadata={"source": "two_step_rag", "domain": "rag"}),
    Document(page_content="Agentic RAG: agent (LLM) decides when and how to retrieve. Uses retrieval as tools. Flexible: can skip retrieval for known facts, use multiple sources, chain queries. Higher latency due to multiple LLM calls. Best for complex multi-hop queries.", metadata={"source": "agentic_rag", "domain": "rag"}),
    Document(page_content="Hybrid RAG adds validation steps: query enhancement (rewrite/expand), retrieval validation (relevance check), answer validation (quality check + self-correction). Highest quality but highest latency. Use for high-stakes, accuracy-critical applications.", metadata={"source": "hybrid_rag", "domain": "rag"}),
    Document(page_content="FAISS (Facebook AI Similarity Search): in-memory vector store, no server needed. Supports exact search (IndexFlatL2) and approximate nearest neighbor (IVFFlat, HNSW). Very fast for small-medium datasets. No persistence by default — use save_local()/load_local().", metadata={"source": "faiss_details", "domain": "vectorstore"}),
    Document(page_content="Chroma: local persistent vector store. Supports metadata filtering, persistent collections, and HTTP mode for client-server setup. Good for development and small-medium production workloads. Backed by SQLite for persistence.", metadata={"source": "chroma_details", "domain": "vectorstore"}),
    Document(page_content="Pinecone: managed cloud vector database. Handles billions of vectors, serverless or pod-based deployment. Supports metadata filtering, namespaces for data isolation, and hybrid search (dense + sparse). Best for large-scale production.", metadata={"source": "pinecone_details", "domain": "vectorstore"}),
    Document(page_content="Text embeddings: OpenAI text-embedding-3-small (1536 dims, $0.02/1M tokens), text-embedding-3-large (3072 dims, highest quality). Open-source: all-MiniLM-L6-v2 (384 dims, fast, free), BGE-large (1024 dims, top open-source performance).", metadata={"source": "embedding_models", "domain": "embeddings"}),
    Document(page_content="RecursiveCharacterTextSplitter: splits text using a hierarchy of separators (paragraph → sentence → word → character). Chunk size 300-500 tokens is typical. Overlap of 10-15% of chunk size maintains context across boundaries.", metadata={"source": "text_splitting", "domain": "rag"}),
    Document(page_content="Retriever strategies: similarity search (k-nearest by cosine), MMR (maximal marginal relevance — balances relevance + diversity), score threshold (filter by minimum similarity). MultiQueryRetriever generates query variants for broader recall.", metadata={"source": "retriever_strategies", "domain": "rag"}),
    Document(page_content="RAG evaluation: faithfulness (answer grounded in context?), answer relevance (does it answer the question?), context precision (retrieved docs relevant?), context recall (found all needed docs?). RAGAS library automates evaluation using LLM judges.", metadata={"source": "rag_evaluation", "domain": "rag"}),
    Document(page_content="LangChain create_agent builds ReAct-style agents. Key parameters: model (LLM), tools (list[@tool functions]), system_prompt (instructions), checkpointer (for persistence), store (long-term memory), middleware (intercept model/tool calls).", metadata={"source": "create_agent_ref", "domain": "langchain"}),
]

splitter   = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks     = splitter.split_documents(KB_DOCS)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vs         = FAISS.from_documents(chunks, embeddings)
retriever  = vs.as_retriever(search_kwargs={"k": 4})
llm        = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print(f"\nKB: {len(KB_DOCS)} docs → {len(chunks)} chunks indexed")


# ════════════════════════════════════════════════════════════════════
# ANSWER GENERATION HELPERS
# ════════════════════════════════════════════════════════════════════

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the provided context. Be accurate and cite sources.
If context is insufficient, say so.

Context:
{context}

Question: {question}

Answer:"""
)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('source','?')}]\n{d.page_content}" for d in docs
    )


def generate_answer(question: str, docs: list[Document]) -> str:
    ctx = format_docs(docs)
    return (ANSWER_PROMPT | llm | StrOutputParser()).invoke({
        "context": ctx, "question": question
    })


# ════════════════════════════════════════════════════════════════════
# RETRIEVAL TOOLS (for agentic path)
# ════════════════════════════════════════════════════════════════════

@tool
def search_knowledge_base(query: str, k: int = 4) -> str:
    """Search the technical knowledge base about RAG, LangChain, and vector stores."""
    docs = vs.similarity_search(query, k=k)
    if not docs:
        return "No results found."
    print(f"  [KB] searched: {query[:50]!r} → {len(docs)} results")
    return format_docs(docs)


@tool
def fetch_web_page(url: str) -> str:
    """Fetch current information from a web page URL."""
    print(f"  [Web] fetching: {url}")
    try:
        r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        text = re.sub(r'<[^>]+>', ' ', r.text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:2000]
    except Exception as e:
        return f"Could not fetch URL: {e}"


# ════════════════════════════════════════════════════════════════════
# ROUTER — determines which RAG path to use
# ════════════════════════════════════════════════════════════════════

ROUTE_PROMPT = ChatPromptTemplate.from_template(
    """Classify the user's question into ONE retrieval architecture:

- "direct":   General knowledge, math, greetings — NO retrieval needed
- "two_step": Focused factual question where retrieval is clearly useful
- "agentic":  Multi-source, multi-hop, or needs web search + KB together
- "hybrid":   Ambiguous, complex trade-off analysis, high accuracy critical

Question: {question}

Respond ONLY with valid JSON — no markdown, no backticks:
{{"architecture": "direct|two_step|agentic|hybrid", "reason": "brief explanation"}}"""
)

def route_query(question: str) -> str:
    response = (ROUTE_PROMPT | llm | StrOutputParser()).invoke({"question": question})
    cleaned  = re.sub(r'```json\s*|\s*```', '', response.strip()).strip()
    try:
        decision = json.loads(cleaned)
        arch     = decision.get("architecture", "two_step")
        reason   = decision.get("reason", "")
        return arch, reason
    except Exception:
        return "two_step", "parsing fallback"


# ════════════════════════════════════════════════════════════════════
# QUERY ENHANCEMENT (for hybrid path)
# ════════════════════════════════════════════════════════════════════

ENHANCE_PROMPT = ChatPromptTemplate.from_template(
    "Rewrite this question to be clear and specific for technical document search.\n"
    "Return ONLY the rewritten question.\n\nQuestion: {question}"
)
enhance_chain = ENHANCE_PROMPT | llm | StrOutputParser()


# ════════════════════════════════════════════════════════════════════
# ANSWER VALIDATION (for hybrid path)
# ════════════════════════════════════════════════════════════════════

VALIDATE_PROMPT = ChatPromptTemplate.from_template(
    """Rate this answer's quality (1-10) and groundedness.

Question: {question}
Context provided: {context}
Answer: {answer}

Respond ONLY with valid JSON — no markdown, no backticks:
{{"score": 1-10, "grounded": true/false, "issue": "main issue or 'none'"}}"""
)


def validate_answer(question: str, context: str, answer: str) -> dict:
    response = (VALIDATE_PROMPT | llm | StrOutputParser()).invoke({
        "question": question, "context": context, "answer": answer
    })
    cleaned = re.sub(r'```json\s*|\s*```', '', response.strip()).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return {"score": 8, "grounded": True, "issue": "none"}


# ════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════

@dataclass
class RAGResult:
    question:     str
    architecture: str
    answer:       str
    sources:      list[str] = field(default_factory=list)
    quality:      dict      = field(default_factory=dict)
    latency_ms:   float     = 0.0


agentic_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_knowledge_base, fetch_web_page],
    system_prompt=(
        "You are a technical Q&A assistant. Use search_knowledge_base to find "
        "technical information about RAG, LangChain, and vector stores. "
        "Use fetch_web_page for current/live information. "
        "Cite your sources. Be accurate and concise."
    ),
)


def smart_qa(question: str) -> RAGResult:
    t0 = time.time()
    arch, reason = route_query(question)
    print(f"\n  [Router] {arch!r} — {reason[:70]}")

    answer  = ""
    sources = []
    quality = {}

    if arch == "direct":
        answer = llm.invoke(question).content

    elif arch == "two_step":
        docs    = retriever.invoke(question)
        sources = [d.metadata.get("source", "?") for d in docs]
        answer  = generate_answer(question, docs)

    elif arch == "agentic":
        result  = agentic_agent.invoke({"messages": [{"role": "user", "content": question}]})
        answer  = result["messages"][-1].content
        sources = ["KB + web (agentic)"]

    elif arch == "hybrid":
        enhanced = enhance_chain.invoke({"question": question})
        print(f"  [Hybrid] Enhanced: {enhanced[:60]!r}")
        docs    = retriever.invoke(enhanced)
        sources = [d.metadata.get("source", "?") for d in docs]
        ctx     = format_docs(docs)
        answer  = generate_answer(enhanced, docs)
        quality = validate_answer(enhanced, ctx, answer)
        print(f"  [Hybrid] Quality score: {quality.get('score', '?')}/10")

    latency = (time.time() - t0) * 1000
    return RAGResult(
        question=question, architecture=arch, answer=answer,
        sources=sources, quality=quality, latency_ms=latency,
    )


# ════════════════════════════════════════════════════════════════════
# SCENARIOS
# ════════════════════════════════════════════════════════════════════

scenarios = [
    # (description, question, expected_arch)
    ("Direct — no retrieval",        "What is 2 to the power of 10?",                           "direct"),
    ("2-Step — focused factual",     "What is the difference between FAISS and Chroma?",         "two_step"),
    ("2-Step — RAG concept",         "How does RAG reduce hallucination?",                       "two_step"),
    ("Agentic — multi-source",       "Compare agentic RAG vs 2-step RAG for complex questions.", "agentic"),
    ("Hybrid — trade-off analysis",  "What are the trade-offs when choosing an embedding model?","hybrid"),
]

print("\n" + "─" * 60)
print("Running all scenarios")
print("─" * 60)

results = []
for description, question, expected in scenarios:
    print(f"\n{'─'*40}")
    print(f"Scenario: {description}")
    print(f"Expected: {expected!r}")

    r = smart_qa(question)
    results.append(r)

    match = "✅" if r.architecture == expected else "⚠️ "
    print(f"Routed:   {r.architecture!r} {match}")
    print(f"Latency:  {r.latency_ms:.0f}ms")
    if r.sources:
        print(f"Sources:  {r.sources}")
    if r.quality:
        print(f"Quality:  {r.quality.get('score','?')}/10 | grounded={r.quality.get('grounded','?')}")
    print(f"Answer:   {r.answer[:200]}")


# ════════════════════════════════════════════════════════════════════
# MULTI-TURN CONVERSATION
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("Multi-Turn Conversation Demo")
print("─" * 60)

conv_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_knowledge_base],
    checkpointer=MemorySaver(),
    system_prompt=(
        "You are a RAG expert assistant. Use search_knowledge_base when you need "
        "specific technical information. Remember the conversation context."
    ),
)

cfg     = {"configurable": {"thread_id": "showcase-conv"}}
conv_qs = [
    "What is RAG and why is it useful?",
    "What are the different architectures for it?",    # follow-up
    "Which should I use for a customer support bot?",  # follow-up
]

for q in conv_qs:
    r = conv_agent.invoke({"messages": [{"role": "user", "content": q}]}, config=cfg)
    print(f"\n  User: {q}")
    print(f"  AI:   {r['messages'][-1].content[:150]}")


# ════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("Results Summary")
print("─" * 60)
print(f"{'Scenario':<25} {'Arch':<10} {'Latency':>10} {'Quality':>10}")
print("-" * 60)
for r in results:
    q   = r.question[:22] + "..." if len(r.question) > 25 else r.question
    q_s = r.quality.get("score", "N/A")
    print(f"{q:<25} {r.architecture:<10} {r.latency_ms:>7.0f}ms  {str(q_s):>7}/10")

print("\n" + "═" * 60)
print("Smart Q&A Showcase Summary:")
print("  Router:   LLM structured output → direct|two_step|agentic|hybrid")
print("  Direct:   llm.invoke() → no retrieval, fast, general knowledge")
print("  2-Step:   retriever → format_docs → prompt → answer")
print("  Agentic:  create_agent(tools=[search_kb, fetch_url]) → reason")
print("  Hybrid:   enhance → retrieve → generate → validate → correct")
print("  Conv:     MemorySaver thread → follow-up awareness")
print("═" * 60)
print("\n✅ Full retrieval showcase complete.")
