# 16_retrieval — Retrieval-Augmented Generation (RAG)

> **RAG enhances LLMs with external knowledge fetched at query time,**
> solving the two key LLM limitations: finite context and static knowledge.

---

## Files in this folder

| File | Architecture | Concepts covered |
|------|-------------|-----------------|
| [`01_knowledge_base.py`](01_knowledge_base.py) | Foundation | Document objects, text loaders, splitters, embeddings, FAISS, retrievers |
| [`02_two_step_rag.py`](02_two_step_rag.py) | 2-Step RAG | LCEL chain, source citations, MultiQueryRetriever, contextual compression, conversational |
| [`03_agentic_rag.py`](03_agentic_rag.py) | Agentic RAG | Retrieval as tool, fetch_url, llms.txt doc assistant, multi-source, multi-hop |
| [`04_hybrid_rag.py`](04_hybrid_rag.py) | Hybrid RAG | Query enhancement, HyDE, retrieval validation, answer validation, self-correction |
| [`05_full_retrieval_showcase.py`](05_full_retrieval_showcase.py) | All | Smart Q&A Assistant — router → direct/2-step/agentic/hybrid |
| [`retrieval_overview.py`](retrieval_overview.py) | Complete retrieval overview in one file |

---

## Quick-start

```bash
pip install faiss-cpu langchain-community

python 16_retrieval/01_knowledge_base.py
python 16_retrieval/02_two_step_rag.py
python 16_retrieval/05_full_retrieval_showcase.py
```

---

## Retrieval Pipeline

```
Sources (web, PDF, database, Slack...)
  → Document Loaders       # standardized Document objects
  → Text Splitters         # chunk_size + chunk_overlap
  → Embedding Model        # text → dense vector
  → Vector Store (FAISS)   # indexed embeddings
  → Retriever              # similarity | MMR | score_threshold
  → LLM Generator          # context + question → answer
```

---

## RAG Architectures

### 📄 2-Step RAG — Fixed Pipeline

Retrieval **always** happens before generation. One retrieval call, one generation call.

```python
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

RAG_PROMPT = ChatPromptTemplate.from_template(
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)

def format_docs(docs):
    return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)

# LCEL chain — composable with | pipe
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is RAG?")
```

**Best for:** FAQs, documentation bots, support chatbots, predictable latency.

---

### 🤖 Agentic RAG — Agent with Retrieval Tools

The agent **decides** when and how to retrieve. Retrieval is just another tool.

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def search_knowledge_base(query: str, k: int = 3) -> str:
    """Search the knowledge base for technical information."""
    docs = vs.similarity_search(query, k=k)
    return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)

@tool
def fetch_url(url: str) -> str:
    """Fetch live content from a web URL."""
    import requests
    response = requests.get(url, timeout=10)
    return response.text[:2000]

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_knowledge_base, fetch_url],
    system_prompt="Use tools when you need external information."
)

result = agent.invoke({"messages": [{"role": "user", "content": "What is FAISS?"}]})
```

**Best for:** Research assistants, multi-source queries, when retrieval is conditional.

---

### 🔀 Hybrid RAG — Validate & Self-Correct

Adds validation steps: query enhancement → retrieval check → answer quality check.

```python
# 1. Enhance query
REWRITE = ChatPromptTemplate.from_template("Rewrite for better search: {question}")
enhanced_q = (REWRITE | llm | StrOutputParser()).invoke({"question": user_query})

# 2. Retrieve with validation loop
docs = retriever.invoke(enhanced_q)
validation = validate_retrieval(user_query, docs)  # LLM checks relevance
if not validation["is_sufficient"]:
    docs = retriever.invoke(validation["refined_query"])

# 3. Generate + validate answer
answer     = generate_answer(enhanced_q, docs)
quality    = validate_answer(enhanced_q, context, answer)
if quality["score"] < 7:
    answer = improve_answer(answer, quality["improvement_instruction"])
```

**Best for:** High-accuracy requirements, ambiguous queries, customer-facing critical flows.

---

## Architecture Comparison

| Architecture | LLM Calls | Latency | Flexibility | Best For |
|---|:---:|:---:|:---:|---|
| **2-Step RAG** | 2 | ⚡ Fast | Low | FAQs, docs bots |
| **Agentic RAG** | 3–7+ | ⏳ Variable | High | Research, multi-source |
| **Hybrid RAG** | 4–8+ | ⏳ High | Medium | High accuracy needs |

---

## Knowledge Base Building Blocks

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load documents
from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader
docs = TextLoader("./my_doc.txt").load()

# 2. Split
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
chunks   = splitter.split_documents(docs)

# 3. Embed + index
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vs         = FAISS.from_documents(chunks, embeddings)
vs.save_local("./my_index")   # persist

# 4. Retrieve
retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10})
results   = retriever.invoke("your query here")
```

---

## Advanced Retrieval Techniques

| Technique | When to Use |
|-----------|------------|
| **MultiQueryRetriever** | Single query misses relevant docs — generate variants |
| **ContextualCompressionRetriever** | Too many irrelevant sentences in retrieved chunks |
| **MMR (Maximal Marginal Relevance)** | Retrieved docs are too similar to each other |
| **Score threshold** | Only want high-confidence results |
| **HyDE** | Query phrasing is very different from document phrasing |
| **Metadata filtering** | Docs span multiple categories/sources |

---

## Dependencies

```bash
# Core (already in requirements.txt)
langchain langchain-openai langchain-core

# Retrieval extras
pip install faiss-cpu                  # vector store
pip install langchain-community        # loaders, FAISS integration
pip install pypdf                      # PDF loading (optional)
pip install beautifulsoup4 lxml        # web loading (optional)
```
