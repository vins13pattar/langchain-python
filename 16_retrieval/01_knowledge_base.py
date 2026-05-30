"""
01_knowledge_base.py
=====================
Building a searchable knowledge base for RAG — the foundation of
all retrieval-augmented generation systems.

Concepts covered:
  - Document objects and metadata
  - Document loaders (text, web, in-memory)
  - Text splitters — RecursiveCharacterTextSplitter, TokenTextSplitter
  - Chunk size and overlap tradeoffs
  - Embedding models — OpenAI text-embedding-3-small
  - Vector stores — FAISS (in-memory, no server needed)
  - Retriever interface — similarity search, MMR, score_threshold
  - Metadata filtering
  - Persisting and reloading a vector store

Pipeline:
  Sources → Loaders → Documents → Split → Embed → VectorStore → Retriever
"""

import os
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

print("=" * 60)
print("Knowledge Base — Document Loaders, Splitters, Vector Stores")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# PART 1: DOCUMENTS
# The Document is the fundamental unit of retrieval.
# Each Document has page_content (str) and metadata (dict).
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Document Objects ──────────────────────────────────────")

# Manual document creation
docs_manual = [
    Document(
        page_content=(
            "LangChain is a framework for building LLM-powered applications. "
            "It provides abstractions for agents, tools, memory, and retrievers. "
            "LangChain supports Python and JavaScript and integrates with dozens of LLM providers."
        ),
        metadata={"source": "langchain_overview", "category": "framework", "version": "1.x"},
    ),
    Document(
        page_content=(
            "LangGraph is a library for building stateful, multi-actor applications with LLMs. "
            "It uses a graph-based execution model where nodes are computation steps "
            "and edges define the flow. LangGraph is built on top of LangChain."
        ),
        metadata={"source": "langgraph_overview", "category": "framework", "version": "1.x"},
    ),
    Document(
        page_content=(
            "RAG (Retrieval-Augmented Generation) enhances LLMs with external knowledge. "
            "Instead of relying solely on training data, RAG retrieves relevant documents "
            "at query time and injects them into the LLM's context."
        ),
        metadata={"source": "rag_concepts", "category": "technique"},
    ),
    Document(
        page_content=(
            "Vector embeddings represent text as high-dimensional numerical vectors. "
            "Texts with similar meaning land close together in vector space. "
            "This enables semantic search — finding relevant documents by meaning, not keywords."
        ),
        metadata={"source": "embeddings_intro", "category": "concept"},
    ),
    Document(
        page_content=(
            "FAISS (Facebook AI Similarity Search) is an efficient library for "
            "similarity search and clustering of dense vectors. It supports "
            "billion-scale search with low memory footprint and fast query times."
        ),
        metadata={"source": "faiss_overview", "category": "vector_store"},
    ),
]

print(f"  Created {len(docs_manual)} documents manually")
for doc in docs_manual[:2]:
    print(f"    [{doc.metadata['source']}] {doc.page_content[:60]}...")


# ════════════════════════════════════════════════════════════════════
# PART 2: DOCUMENT LOADERS
# Loaders convert various sources into standardized Document objects.
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Document Loaders ──────────────────────────────────────")

# 2a. TextLoader — load from a plain text file
# from langchain_community.document_loaders import TextLoader
# loader = TextLoader("./my_doc.txt", encoding="utf-8")
# docs = loader.load()  # returns list[Document]

# 2b. PyPDFLoader — load from a PDF (requires pypdf)
# from langchain_community.document_loaders import PyPDFLoader
# loader = PyPDFLoader("./report.pdf")
# docs = loader.load_and_split()  # splits by page

# 2c. WebBaseLoader — load from a URL (requires beautifulsoup4)
# from langchain_community.document_loaders import WebBaseLoader
# loader = WebBaseLoader("https://python.langchain.com/docs/")
# docs = loader.load()

# 2d. DirectoryLoader — load all files in a directory
# from langchain_community.document_loaders import DirectoryLoader
# loader = DirectoryLoader("./docs/", glob="**/*.md")
# docs = loader.load()

# For this demo we simulate a "web loader" by creating Documents
web_docs = [
    Document(
        page_content=(
            "OpenAI's GPT-4 is a multimodal large language model capable of "
            "understanding images and text. It significantly outperforms GPT-3.5 "
            "on various benchmarks including coding, reasoning, and language tasks."
        ),
        metadata={"source": "openai.com/gpt4", "type": "web", "loaded_at": "2024-01"},
    ),
    Document(
        page_content=(
            "Anthropic's Claude 3 family includes Haiku, Sonnet, and Opus. "
            "Claude excels at nuanced reasoning, long context (200K tokens), "
            "and following complex instructions with minimal hallucination."
        ),
        metadata={"source": "anthropic.com/claude", "type": "web", "loaded_at": "2024-03"},
    ),
]

all_docs = docs_manual + web_docs
print(f"  Total raw documents: {len(all_docs)}")


# ════════════════════════════════════════════════════════════════════
# PART 3: TEXT SPLITTERS
# Large documents must be split into smaller chunks that:
#   1. Fit within the model's context window
#   2. Are individually retrievable
# The key parameters: chunk_size, chunk_overlap, separators
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Text Splitters ────────────────────────────────────────")

# 3a. RecursiveCharacterTextSplitter — tries paragraph → sentence → char
rcts = RecursiveCharacterTextSplitter(
    chunk_size=300,         # max chars per chunk
    chunk_overlap=50,       # overlap keeps context across chunk boundaries
    length_function=len,    # use character count (use tiktoken for token count)
    add_start_index=True,   # adds start_index to metadata
)

# Create longer test documents to demonstrate splitting
long_doc = Document(
    page_content=(
        "Large Language Models (LLMs) have transformed artificial intelligence. "
        "They are trained on vast amounts of text data, learning patterns and "
        "relationships between words and concepts. Modern LLMs can write code, "
        "answer questions, summarize documents, and engage in natural conversation. "
        "\n\n"
        "The key innovation behind LLMs is the transformer architecture, introduced "
        "in 2017. Transformers use self-attention mechanisms to process all words "
        "in a sequence simultaneously, capturing long-range dependencies efficiently. "
        "\n\n"
        "Fine-tuning and instruction tuning further improve LLM performance on "
        "specific tasks. Reinforcement Learning from Human Feedback (RLHF) aligns "
        "the model's outputs with human preferences and safety guidelines."
    ),
    metadata={"source": "llm_primer", "category": "education"},
)

chunks_rcts = rcts.split_documents([long_doc])
print(f"  RecursiveCharacter: 1 doc → {len(chunks_rcts)} chunks")
for i, chunk in enumerate(chunks_rcts):
    print(f"    Chunk {i+1}: {len(chunk.page_content)} chars | "
          f"start_index={chunk.metadata.get('start_index', '?')}")

# Split all docs
all_chunks = rcts.split_documents(all_docs)
print(f"\n  Split {len(all_docs)} docs → {len(all_chunks)} chunks")

# 3b. TokenTextSplitter — split by token count (more precise for LLM context)
tts = TokenTextSplitter(
    chunk_size=100,         # max tokens per chunk
    chunk_overlap=10,
)
token_chunks = tts.split_documents([long_doc])
print(f"\n  TokenTextSplitter: 1 doc → {len(token_chunks)} chunks (by token count)")

# 3c. Chunk size guidance
print("\n  Chunk size guidance:")
print("    Small  (100-200):  high precision, less context per chunk")
print("    Medium (300-500):  good balance for most RAG applications")
print("    Large  (1000+):    more context but harder to match precisely")
print("    Overlap 10-15% of chunk_size is a good default")


# ════════════════════════════════════════════════════════════════════
# PART 4: EMBEDDINGS
# Embeddings convert text chunks to dense vectors.
# Chunks with similar meaning land close in vector space.
# ════════════════════════════════════════════════════════════════════

print("\n── 4. Embeddings ────────────────────────────────────────────")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Embed a single query to see the dimension
sample_embedding = embeddings.embed_query("What is RAG?")
print(f"  Model: text-embedding-3-small")
print(f"  Vector dimension: {len(sample_embedding)}")
print(f"  First 5 values: {[round(x, 4) for x in sample_embedding[:5]]}")

# Embed multiple texts
sample_texts = ["LangChain is a framework", "FAISS is a vector database"]
multi_embeddings = embeddings.embed_documents(sample_texts)
print(f"  Embedded {len(multi_embeddings)} texts")


# ════════════════════════════════════════════════════════════════════
# PART 5: VECTOR STORE — FAISS
# FAISS stores embeddings for fast similarity search.
# Works entirely in-memory — no server needed.
# ════════════════════════════════════════════════════════════════════

print("\n── 5. Vector Store (FAISS) ──────────────────────────────────")

# Build vector store from all chunks
vs = FAISS.from_documents(all_chunks, embeddings)
print(f"  Indexed {len(all_chunks)} chunks into FAISS")

# Basic similarity search
query   = "What is LangChain?"
results = vs.similarity_search(query, k=3)
print(f"\n  Query: {query!r}")
for i, doc in enumerate(results, 1):
    src = doc.metadata.get("source", "?")
    print(f"  [{i}] ({src}) {doc.page_content[:80]}...")

# Similarity search with relevance scores
print(f"\n  Similarity search with scores:")
scored = vs.similarity_search_with_relevance_scores(query, k=3)
for doc, score in scored:
    src = doc.metadata.get("source", "?")
    print(f"    score={score:.4f} | ({src}) {doc.page_content[:60]}...")

# Persist and reload
vs.save_local("/tmp/faiss_demo_index")
reloaded_vs = FAISS.load_local(
    "/tmp/faiss_demo_index",
    embeddings,
    allow_dangerous_deserialization=True,  # needed for local files
)
reload_results = reloaded_vs.similarity_search("vector embeddings", k=1)
print(f"\n  Reloaded index — query result: {reload_results[0].page_content[:60]}...")


# ════════════════════════════════════════════════════════════════════
# PART 6: RETRIEVERS
# A retriever wraps a vector store to provide a standard interface.
# Three retrieval strategies: similarity, MMR, score_threshold.
# ════════════════════════════════════════════════════════════════════

print("\n── 6. Retrievers ────────────────────────────────────────────")

# 6a. Basic similarity retriever
retriever_sim = vs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
docs_sim = retriever_sim.invoke("How does semantic search work?")
print(f"\n  Similarity retriever (k=3):")
for d in docs_sim:
    print(f"    [{d.metadata.get('source','?')}] {d.page_content[:70]}...")

# 6b. MMR (Maximal Marginal Relevance) — balances relevance + diversity
retriever_mmr = vs.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},
    # lambda_mult: 0=max diversity, 1=max relevance (default 0.5)
)
docs_mmr = retriever_mmr.invoke("LLM frameworks and tools")
print(f"\n  MMR retriever (k=3, fetch_k=10):")
for d in docs_mmr:
    print(f"    [{d.metadata.get('source','?')}] {d.page_content[:70]}...")

# 6c. Score threshold — only return chunks above a similarity score
retriever_threshold = vs.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5, "k": 5},
)
docs_threshold = retriever_threshold.invoke("What is FAISS?")
print(f"\n  Score threshold retriever (>0.5): {len(docs_threshold)} docs returned")

# 6d. Retriever invocation (standard interface)
print("\n  Retriever invoke() interface:")
retrieved = retriever_sim.invoke("transformer attention mechanism")
print(f"    {len(retrieved)} documents retrieved")
for d in retrieved[:2]:
    print(f"    → {d.page_content[:80]}...")


# ════════════════════════════════════════════════════════════════════
# PART 7: METADATA FILTERING
# Filter results by metadata before performing similarity search.
# Useful when documents span multiple sources/categories.
# ════════════════════════════════════════════════════════════════════

print("\n── 7. Metadata Filtering ────────────────────────────────────")

# Build separate indices per category for filtering simulation
concept_docs = [c for c in all_chunks if c.metadata.get("category") == "concept"]
framework_docs = [c for c in all_chunks if c.metadata.get("category") == "framework"]

print(f"  concept docs: {len(concept_docs)}")
print(f"  framework docs: {len(framework_docs)}")

if concept_docs:
    concept_vs    = FAISS.from_documents(concept_docs, embeddings)
    concept_ret   = concept_vs.as_retriever(search_kwargs={"k": 2})
    filtered_docs = concept_ret.invoke("what are embeddings")
    print(f"  Filtered (concept only): {len(filtered_docs)} docs")
    for d in filtered_docs:
        print(f"    → {d.page_content[:70]}...")

print("\n" + "═" * 60)
print("Knowledge Base Summary:")
print("  Document          → page_content + metadata dict")
print("  TextLoader        → local files; WebBaseLoader → URLs")
print("  RecursiveCharacterTextSplitter → chunk_size + chunk_overlap")
print("  OpenAIEmbeddings  → embed_query() + embed_documents()")
print("  FAISS             → from_documents() → in-memory index")
print("  as_retriever()    → similarity | mmr | score_threshold")
print("═" * 60)
print("\n✅ Knowledge base demo complete.")
