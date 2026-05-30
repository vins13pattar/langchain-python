"""
02_two_step_rag.py
===================
Demonstrates 2-Step RAG — a fixed pipeline where retrieval ALWAYS
happens before generation. Predictable, fast, and simple.

Pipeline:
  User Question → [Retrieve k docs] → [Generate answer with context] → Response

Concepts covered:
  - Basic 2-step RAG using LCEL (| pipe operator)
  - ChatPromptTemplate for RAG prompting
  - Context formatting from retrieved documents
  - Source citation in answers
  - Multi-query retrieval — generate multiple query variants to improve recall
  - Contextual compression — compress retrieved docs to relevant snippets
  - Conversational RAG — multi-turn Q&A with chat history
  - Comparing 2-step vs agentic approach

Best for: FAQs, documentation bots, when retrieval is always needed.
Not for: Queries that don't need external knowledge (wastes a retrieval call).
"""

from dotenv import load_dotenv

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chat_models import init_chat_model
from langchain.checkpoint.memory import MemorySaver
from langgraph.checkpoint.memory import MemorySaver as LGMemorySaver

load_dotenv()

print("=" * 60)
print("2-Step RAG — Fixed Retrieve-Then-Generate Pipeline")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE (shared across all demos)
# ════════════════════════════════════════════════════════════════════

KNOWLEDGE = [
    Document(page_content="Python is a high-level, interpreted programming language created by Guido van Rossum in 1991. It emphasizes code readability with its clean, English-like syntax. Python supports multiple programming paradigms: object-oriented, functional, and procedural.", metadata={"source": "python_basics", "topic": "python"}),
    Document(page_content="Python's key strengths include: an extensive standard library, massive third-party ecosystem (PyPI has over 400,000 packages), cross-platform compatibility, and a large, active community. It's used in web development, data science, AI/ML, automation, and scientific computing.", metadata={"source": "python_strengths", "topic": "python"}),
    Document(page_content="Python is the dominant language for machine learning and AI. Popular frameworks include TensorFlow, PyTorch, scikit-learn, and Keras. The rich data ecosystem (NumPy, Pandas, Matplotlib) makes Python ideal for data analysis and visualization.", metadata={"source": "python_ml", "topic": "python"}),
    Document(page_content="JavaScript (JS) is the language of the web, running natively in all browsers. It was created by Brendan Eich in 1995 in just 10 days. Today it's also used server-side via Node.js. JS is event-driven and asynchronous by nature, making it excellent for interactive UIs.", metadata={"source": "js_basics", "topic": "javascript"}),
    Document(page_content="The JavaScript ecosystem is vast. Popular frameworks include React (UI library by Meta), Vue.js (progressive framework), Angular (Google's full framework), and Svelte. Node.js and Deno enable server-side JS. TypeScript adds static types to JavaScript.", metadata={"source": "js_ecosystem", "topic": "javascript"}),
    Document(page_content="Rust is a systems programming language focused on safety, speed, and concurrency. It achieves memory safety without a garbage collector using its ownership and borrowing system. Rust compiles to native machine code, matching C/C++ performance.", metadata={"source": "rust_basics", "topic": "rust"}),
    Document(page_content="Rust's ownership model eliminates entire classes of bugs at compile time: null pointer dereferences, data races, and use-after-free errors. The borrow checker enforces these rules. Rust also has excellent tooling: Cargo (build system), rustfmt (formatter), and Clippy (linter).", metadata={"source": "rust_safety", "topic": "rust"}),
    Document(page_content="RAG (Retrieval-Augmented Generation) combines retrieval systems with LLMs. The retrieval step fetches relevant documents from a knowledge base. The generation step uses the retrieved context to produce a grounded, factual answer. RAG reduces hallucination compared to pure generation.", metadata={"source": "rag_overview", "topic": "rag"}),
    Document(page_content="Vector stores index document embeddings for fast similarity search. Popular vector stores include FAISS (in-memory), Pinecone (managed cloud), Chroma (local persistent), Weaviate, and Qdrant. Each offers different tradeoffs in terms of scalability and features.", metadata={"source": "vector_stores", "topic": "rag"}),
    Document(page_content="Embeddings are dense vector representations of text. OpenAI's text-embedding-3-small (1536 dimensions) and text-embedding-3-large (3072 dimensions) are popular choices. Cosine similarity is commonly used to measure semantic similarity between embedding vectors.", metadata={"source": "embeddings", "topic": "rag"}),
]

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
chunks   = splitter.split_documents(KNOWLEDGE)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vs         = FAISS.from_documents(chunks, embeddings)
retriever  = vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print(f"\nKnowledge base: {len(KNOWLEDGE)} docs → {len(chunks)} chunks indexed")


# ════════════════════════════════════════════════════════════════════
# PART 1: BASIC 2-STEP RAG (LCEL chain)
# retrieve → format → prompt → llm → parse
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Basic 2-Step RAG (LCEL) ───────────────────────────────")

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer (cite your sources using [source] notation):"""
)


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a single context string."""
    parts = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", f"doc_{i}")
        parts.append(f"[{src}]\n{doc.page_content}")
    return "\n\n".join(parts)


# LCEL chain: the | operator creates a pipeline
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)

# Test queries
queries = [
    "What are Python's main strengths?",
    "How does Rust ensure memory safety?",
    "What is RAG and how does it work?",
]

for q in queries:
    answer = rag_chain.invoke(q)
    print(f"\n  Q: {q}")
    print(f"  A: {answer[:200]}")


# ════════════════════════════════════════════════════════════════════
# PART 2: WITH SOURCE CITATIONS
# Return both the answer AND the source documents.
# ════════════════════════════════════════════════════════════════════

print("\n── 2. 2-Step RAG with Source Citations ──────────────────────")

from langchain.schema.runnable import RunnableParallel

# Parallel chain: retrieve docs AND run the full RAG chain
rag_with_sources = RunnableParallel(
    answer=rag_chain,
    sources=(retriever | (lambda docs: [d.metadata.get("source") for d in docs])),
)

result = rag_with_sources.invoke("What JavaScript frameworks are popular?")
print(f"\n  Answer: {result['answer'][:200]}")
print(f"  Sources: {result['sources']}")


# ════════════════════════════════════════════════════════════════════
# PART 3: MULTI-QUERY RETRIEVAL
# Generate multiple variants of the user's query → retrieve for each
# → deduplicate → broader recall than single-query retrieval.
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Multi-Query Retrieval ─────────────────────────────────")

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vs.as_retriever(search_kwargs={"k": 3}),
    llm=llm,
)

# Enable logging to see generated queries
import logging
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

mq_query = "How does Python compare to JavaScript for web apps?"
mq_docs  = multiquery_retriever.invoke(mq_query)
print(f"\n  Query: {mq_query!r}")
print(f"  Multi-query retrieved: {len(mq_docs)} unique docs (vs 3 with single query)")
for doc in mq_docs[:3]:
    print(f"    [{doc.metadata.get('source')}] {doc.page_content[:60]}...")

# Build RAG chain with multi-query retriever
mq_rag_chain = (
    {"context": multiquery_retriever | format_docs, "question": RunnablePassthrough()}
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)
mq_answer = mq_rag_chain.invoke(mq_query)
print(f"\n  Answer: {mq_answer[:200]}")


# ════════════════════════════════════════════════════════════════════
# PART 4: CONTEXTUAL COMPRESSION
# Extract only the relevant snippets from each retrieved document,
# reducing noise and token usage before sending to the LLM.
# ════════════════════════════════════════════════════════════════════

print("\n── 4. Contextual Compression ────────────────────────────────")

# LLMChainExtractor: LLM extracts only relevant sentences from each doc
compressor          = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vs.as_retriever(search_kwargs={"k": 4}),
)

comp_query = "What is Rust's borrow checker?"
comp_docs  = compression_retriever.invoke(comp_query)
print(f"\n  Query: {comp_query!r}")
print(f"  Compressed: {len(comp_docs)} docs")
for doc in comp_docs:
    print(f"    [{doc.metadata.get('source')}] {doc.page_content[:120]}")


# ════════════════════════════════════════════════════════════════════
# PART 5: CONVERSATIONAL 2-STEP RAG
# Multi-turn Q&A — reformulate question using chat history before
# retrieving, so follow-up questions work correctly.
# ════════════════════════════════════════════════════════════════════

print("\n── 5. Conversational 2-Step RAG (multi-turn) ────────────────")

# Step 1: Condense the user's question + history into a standalone query
CONDENSE_PROMPT = ChatPromptTemplate.from_template(
    """Given the conversation history and the new question, 
rephrase the follow-up question as a standalone question that is self-contained.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""
)

# Step 2: Answer using retrieved context
QA_PROMPT = ChatPromptTemplate.from_template(
    """Answer based on the context. Be concise.

Context:
{context}

Question: {question}
Answer:"""
)


def build_conversational_rag():
    chat_history = []

    def chat(user_msg: str) -> str:
        # Condense if there's history
        if chat_history:
            history_str = "\n".join(
                f"Human: {h}\nAssistant: {a}" for h, a in chat_history
            )
            standalone = (CONDENSE_PROMPT | llm | StrOutputParser()).invoke({
                "chat_history": history_str,
                "question":     user_msg,
            })
        else:
            standalone = user_msg

        # Retrieve + answer
        docs   = retriever.invoke(standalone)
        ctx    = format_docs(docs)
        answer = (QA_PROMPT | llm | StrOutputParser()).invoke({
            "context":  ctx,
            "question": standalone,
        })

        chat_history.append((user_msg, answer))
        return answer

    return chat


conv_rag = build_conversational_rag()

turns = [
    "What is Python used for?",
    "How does it compare to Rust?",   # follow-up — needs condensing
    "Which is better for systems programming?",
]

print()
for msg in turns:
    answer = conv_rag(msg)
    print(f"  Human: {msg}")
    print(f"  AI:    {answer[:150]}\n")

print("═" * 60)
print("2-Step RAG Summary:")
print("  retriever | format_docs → inject context into prompt")
print("  LCEL pipe (|) builds composable RAG chains")
print("  MultiQueryRetriever → multiple query variants → better recall")
print("  ContextualCompressionRetriever → extract relevant snippets")
print("  Conversational RAG → condense + history + retrieve + answer")
print("  Best for: predictable flows, known retrieval need, FAQs")
print("═" * 60)
print("\n✅ 2-Step RAG demo complete.")
