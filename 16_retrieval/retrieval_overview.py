"""
retrieval_overview.py — LangChain Retrieval / RAG: all key concepts in one file
Covers: Documents, loaders, text splitters, embeddings, FAISS vector store,
        retriever modes (similarity, MMR, threshold), two-step RAG, agentic RAG, hybrid RAG
"""

from dotenv import load_dotenv

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore

load_dotenv()

def section(title): print(f"\n{'─'*55}\n{title}\n{'─'*55}")


# ════════════════════════════════════════════════════════════════════
# 1. DOCUMENTS — the fundamental unit of retrieval
# ════════════════════════════════════════════════════════════════════
section("1. DOCUMENTS")

docs = [
    Document(
        page_content="LangChain is a framework for building LLM-powered applications. It provides abstractions for agents, tools, memory, and retrievers.",
        metadata={"source": "langchain_overview", "category": "framework"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor LLM applications using a graph-based execution model.",
        metadata={"source": "langgraph_overview", "category": "framework"},
    ),
    Document(
        page_content="RAG (Retrieval-Augmented Generation) enhances LLMs with external knowledge by retrieving relevant documents at query time.",
        metadata={"source": "rag_concepts", "category": "technique"},
    ),
    Document(
        page_content="Vector embeddings represent text as high-dimensional vectors. Texts with similar meaning land close together in vector space.",
        metadata={"source": "embeddings_intro", "category": "concept"},
    ),
    Document(
        page_content="FAISS (Facebook AI Similarity Search) is an efficient in-memory library for fast similarity search over dense vectors.",
        metadata={"source": "faiss_overview", "category": "vector_store"},
    ),
    Document(
        page_content="OpenAI's GPT-4 is a multimodal LLM capable of understanding images and text, outperforming GPT-3.5 on most benchmarks.",
        metadata={"source": "openai_gpt4", "category": "model"},
    ),
    Document(
        page_content="Transformer architecture uses self-attention to process all tokens in a sequence simultaneously, capturing long-range dependencies.",
        metadata={"source": "transformers", "category": "concept"},
    ),
    Document(
        page_content="Fine-tuning and RLHF (Reinforcement Learning from Human Feedback) improve LLM performance and align outputs with human preferences.",
        metadata={"source": "training", "category": "technique"},
    ),
]
print(f"Created {len(docs)} documents")


# ════════════════════════════════════════════════════════════════════
# 2. TEXT SPLITTER — chunk large docs into retrievable pieces
# ════════════════════════════════════════════════════════════════════
section("2. TEXT SPLITTER")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,   # max chars per chunk
    chunk_overlap=50, # overlap to keep context across boundaries
    add_start_index=True,
)

# Create a longer doc to demonstrate splitting
long_doc = Document(
    page_content=(
        "Large Language Models (LLMs) have transformed artificial intelligence. "
        "They are trained on vast amounts of text, learning patterns and relationships. "
        "Modern LLMs can write code, answer questions, and summarise documents.\n\n"
        "The key innovation behind LLMs is the transformer architecture, introduced in 2017. "
        "Transformers use self-attention to process all words simultaneously.\n\n"
        "Fine-tuning and instruction tuning further improve LLM performance on specific tasks. "
        "RLHF aligns outputs with human preferences and safety guidelines."
    ),
    metadata={"source": "llm_primer"},
)

chunks = splitter.split_documents([long_doc])
print(f"1 doc → {len(chunks)} chunks")
for i, c in enumerate(chunks):
    print(f"  Chunk {i+1}: {len(c.page_content)} chars, start_index={c.metadata.get('start_index','?')}")

all_chunks = splitter.split_documents(docs)
print(f"\n{len(docs)} docs → {len(all_chunks)} total chunks")


# ════════════════════════════════════════════════════════════════════
# 3. EMBEDDINGS + VECTOR STORE (FAISS)
# ════════════════════════════════════════════════════════════════════
section("3. EMBEDDINGS + FAISS VECTOR STORE")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Show embedding dimension
sample_vec = embeddings.embed_query("What is RAG?")
print(f"Embedding model: text-embedding-3-small  dim={len(sample_vec)}")

# Build FAISS index from all chunks
vs = FAISS.from_documents(all_chunks, embeddings)
print(f"Indexed {len(all_chunks)} chunks into FAISS")

# Persist + reload
vs.save_local("/tmp/faiss_rag_demo")
vs_loaded = FAISS.load_local("/tmp/faiss_rag_demo", embeddings, allow_dangerous_deserialization=True)
print("Saved and reloaded FAISS index successfully")


# ════════════════════════════════════════════════════════════════════
# 4. RETRIEVERS — three search strategies
# ════════════════════════════════════════════════════════════════════
section("4. RETRIEVERS")

# 4a. Similarity search (cosine / L2)
retriever_sim = vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})
q = "What is LangChain?"
results = retriever_sim.invoke(q)
print(f"Similarity (k=3) for {q!r}:")
for d in results:
    print(f"  [{d.metadata['source']}] {d.page_content[:70]}...")

# 4b. MMR — relevance + diversity balance
retriever_mmr = vs.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},  # 0=diversity, 1=relevance
)
results_mmr = retriever_mmr.invoke("LLM frameworks and tools")
print(f"\nMMR (k=3, fetch_k=10):")
for d in results_mmr:
    print(f"  [{d.metadata['source']}] {d.page_content[:70]}...")

# 4c. Score threshold — only above a similarity score
retriever_thresh = vs.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5, "k": 5},
)
results_thresh = retriever_thresh.invoke("What is FAISS?")
print(f"\nScore threshold (>0.5): {len(results_thresh)} docs returned")

# With relevance scores
print("\nSimilarity with scores:")
scored = vs.similarity_search_with_relevance_scores("vector embeddings", k=3)
for d, score in scored:
    print(f"  score={score:.4f}  [{d.metadata['source']}] {d.page_content[:60]}...")


# ════════════════════════════════════════════════════════════════════
# 5. TWO-STEP RAG — retrieve then generate
# ════════════════════════════════════════════════════════════════════
section("5. TWO-STEP RAG")

from langchain.chat_models import init_chat_model

llm = init_chat_model("openai:gpt-4o-mini")

def rag_query(question: str, retriever, top_k: int = 3) -> str:
    """Retrieve relevant docs, inject into prompt, generate answer."""
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in retrieved_docs)
    prompt = [
        {"role": "system", "content": (
            "You are a helpful assistant. Answer questions using ONLY the provided context. "
            "If the answer is not in the context, say 'I don't have that information.'"
        )},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return llm.invoke(prompt).content

for q in ["What is LangChain?", "How does RAG work?"]:
    answer = rag_query(q, retriever_sim)
    print(f"Q: {q}")
    print(f"A: {answer[:150]}")
    print()


# ════════════════════════════════════════════════════════════════════
# 6. AGENTIC RAG — retriever as an agent tool
# ════════════════════════════════════════════════════════════════════
section("6. AGENTIC RAG")

# The agent decides when to search and what to search for
@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information about LangChain, RAG, and LLMs. Args: query."""
    docs_found = retriever_sim.invoke(query)
    if not docs_found:
        return "No relevant information found."
    return "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in docs_found)

rag_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_knowledge_base],
    system_prompt=(
        "You are a knowledgeable assistant. Use search_knowledge_base to answer questions "
        "about LangChain, LangGraph, RAG, and LLM concepts. "
        "Always search before answering factual questions."
    ),
)

r = rag_agent.invoke({"messages": [{"role": "user", "content": "What is LangGraph and how does it differ from LangChain?"}]})
print("Agentic RAG:", r["messages"][-1].content[:200])


# ════════════════════════════════════════════════════════════════════
# 7. HYBRID RAG — vector search + keyword/metadata filtering
# ════════════════════════════════════════════════════════════════════
section("7. HYBRID RAG (semantic + metadata filter)")

def hybrid_search(query: str, category_filter: str | None = None, k: int = 3) -> list[Document]:
    """Combine vector similarity with metadata category filtering."""
    if category_filter:
        # Filter chunks by category first
        filtered = [c for c in all_chunks if c.metadata.get("category") == category_filter]
        if not filtered:
            return retriever_sim.invoke(query)[:k]
        # Build mini index on filtered docs
        mini_vs = FAISS.from_documents(filtered, embeddings)
        return mini_vs.similarity_search(query, k=k)
    return retriever_sim.invoke(query)[:k]

for q, cat in [
    ("How do I build an agent?",    "framework"),
    ("What is self-attention?",      "concept"),
    ("Explain RAG techniques",       "technique"),
]:
    results_h = hybrid_search(q, category_filter=cat)
    print(f"Hybrid [{cat}] for {q!r}:")
    for d in results_h:
        print(f"  [{d.metadata['source']}] {d.page_content[:70]}...")
    print()


# ════════════════════════════════════════════════════════════════════
# 8. RAG WITH LONG-TERM STORE — user-personalised retrieval
# ════════════════════════════════════════════════════════════════════
section("8. RAG + LONG-TERM STORE")

from langgraph.store.base import IndexConfig
from collections.abc import Sequence
from langchain_openai import OpenAIEmbeddings as OAIEmbed

def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    return OAIEmbed(model="text-embedding-3-small").embed_documents(list(texts))

store = InMemoryStore(index=IndexConfig(embed=embed_texts, dims=1536))

# Store user-specific memories
for i, (key, content) in enumerate([
    ("pref_1", "User prefers concise, bullet-point answers."),
    ("pref_2", "User is an expert Python developer with 10 years experience."),
    ("pref_3", "User's current project is a FastAPI microservice."),
]):
    store.put(("user_001", "memories"), key, {"content": content, "type": "preference"})

@tool
def search_kb_personalised(query: str, runtime: ToolRuntime) -> str:
    """Search knowledge base with user-context from long-term memory. Args: query."""
    # Retrieve user preferences from store
    user_prefs = runtime.store.search(("user_001", "memories"), query=query, limit=2)
    pref_text = "\n".join(p.value["content"] for p in user_prefs) if user_prefs else ""

    # Retrieve from knowledge base
    kb_docs = retriever_sim.invoke(query)
    kb_text = "\n\n".join(d.page_content for d in kb_docs)

    return f"User context:\n{pref_text}\n\nKB results:\n{kb_text}"

personalised_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_kb_personalised],
    store=store,
    system_prompt="You are a personalised assistant. Use user preferences when answering.",
)

r = personalised_agent.invoke({"messages": [{"role": "user", "content": "Explain how RAG works."}]})
print("Personalised RAG:", r["messages"][-1].content[:200])

print("""
RAG Pipeline:
  Sources → Loaders → Documents → Splitter → Chunks
  → Embeddings → VectorStore (FAISS) → Retriever → LLM

Retriever modes:
  similarity                   → cosine/L2, returns k docs
  mmr                          → relevance + diversity (lambda_mult controls balance)
  similarity_score_threshold   → only return docs above a score

RAG variants:
  Two-step RAG  → retrieve then generate (simple, predictable)
  Agentic RAG   → agent decides when/what to search (flexible)
  Hybrid RAG    → semantic + metadata filter (precise)
""")
