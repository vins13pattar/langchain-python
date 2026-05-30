"""
04_hybrid_rag.py
=================
Demonstrates Hybrid RAG — combines 2-step and agentic RAG with
intermediate validation steps: query enhancement, retrieval validation,
and answer quality checking with self-correction loops.

Concepts covered:
  - Query enhancement — rewrite/expand queries for better retrieval
  - Multi-query expansion — generate query variants
  - HyDE (Hypothetical Document Embeddings) — generate a fake answer,
    embed it, then search for real docs close to that embedding
  - Retrieval validation — check if retrieved docs are relevant
  - Iterative refinement — refine query and re-retrieve if needed
  - Answer validation — check quality, relevance, grounding
  - Self-correction loop — regenerate if answer quality is poor
  - Structured output at each validation step

Architecture:
  User Question
    → Query Enhancement (rewrite / expand / HyDE)
    → Retrieve Documents
    → [Loop] Retrieval Validation → Refine Query → Re-Retrieve
    → Generate Answer
    → Answer Validation → [Loop] Refine or Return

Best for: High-accuracy requirements, ambiguous queries, multi-source
          workflows where quality control is essential.
"""

from dataclasses import dataclass
from typing import Literal, Optional
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

print("=" * 60)
print("Hybrid RAG — Query Enhancement + Validation + Self-Correction")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# SHARED KNOWLEDGE BASE
# ════════════════════════════════════════════════════════════════════

DOCS = [
    Document(page_content="Machine learning (ML) is a subset of artificial intelligence where systems learn from data without being explicitly programmed. Supervised learning uses labeled examples; unsupervised learning finds patterns in unlabeled data; reinforcement learning trains agents via rewards.", metadata={"source": "ml_intro", "domain": "AI"}),
    Document(page_content="Neural networks are computational models inspired by the human brain. They consist of layers of interconnected nodes (neurons). Deep learning uses neural networks with many layers (deep networks) to learn hierarchical representations from data.", metadata={"source": "neural_nets", "domain": "AI"}),
    Document(page_content="Transformers are the dominant architecture for NLP tasks. Introduced in 'Attention Is All You Need' (2017), they use self-attention mechanisms to process sequences in parallel. BERT, GPT, and T5 are all transformer-based models.", metadata={"source": "transformers", "domain": "AI"}),
    Document(page_content="Large Language Models (LLMs) are transformers trained on massive text corpora. GPT-4 (OpenAI), Claude 3 (Anthropic), and Gemini (Google) are leading LLMs. They exhibit emergent capabilities: in-context learning, chain-of-thought reasoning, and few-shot generalization.", metadata={"source": "llms_overview", "domain": "AI"}),
    Document(page_content="Prompt engineering involves crafting effective inputs to guide LLM behavior. Key techniques: zero-shot prompting, few-shot prompting, chain-of-thought (CoT), tree-of-thought (ToT), and ReAct (Reasoning + Acting). Structured output prompts request JSON or typed responses.", metadata={"source": "prompt_eng", "domain": "AI"}),
    Document(page_content="Fine-tuning adapts a pre-trained model to a specific task or domain using task-specific labeled data. LoRA (Low-Rank Adaptation) and QLoRA make fine-tuning efficient on consumer hardware. RLHF (Reinforcement Learning from Human Feedback) aligns models with human preferences.", metadata={"source": "fine_tuning", "domain": "AI"}),
    Document(page_content="RAG (Retrieval-Augmented Generation) reduces hallucination by grounding LLM responses in retrieved external knowledge. Key components: document loader, text splitter, embedding model, vector store, retriever, and LLM generator.", metadata={"source": "rag_intro", "domain": "RAG"}),
    Document(page_content="Vector databases store and search embeddings efficiently. FAISS (in-memory, Meta), Pinecone (managed cloud), Chroma (local persistent), Weaviate, and Qdrant are popular options. They support exact and approximate nearest neighbor (ANN) search algorithms.", metadata={"source": "vector_dbs", "domain": "RAG"}),
    Document(page_content="Evaluation metrics for RAG systems: faithfulness (is the answer grounded in context?), answer relevance (does it answer the question?), context precision (are retrieved docs relevant?), context recall (did retrieval find all needed docs?). RAGAS is a popular evaluation framework.", metadata={"source": "rag_eval", "domain": "RAG"}),
]

splitter   = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks     = splitter.split_documents(DOCS)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vs         = FAISS.from_documents(chunks, embeddings)
retriever  = vs.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print(f"\nKnowledge base: {len(DOCS)} docs → {len(chunks)} chunks")


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('source','?')}]\n{d.page_content}"
        for d in docs
    )


# ════════════════════════════════════════════════════════════════════
# PART 1: QUERY ENHANCEMENT
# Rewrite or expand the user's query before retrieval to improve recall.
# Three strategies:
#   1. Query rewriting — fix grammar, expand abbreviations, clarify
#   2. Multi-query expansion — generate query variants
#   3. HyDE — generate a hypothetical answer, embed it, search for real docs
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Query Enhancement ─────────────────────────────────────")

# 1a. Query Rewriting
REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """Rewrite the user's question to be more specific, clear, and searchable.
Expand abbreviations, fix ambiguity, and add relevant technical context.
Return ONLY the rewritten question, nothing else.

Original question: {question}

Rewritten question:"""
)

rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()

queries_to_enhance = [
    "how does attn wrk in transformers",
    "whats rlhf",
    "difference between ML and DL",
]

print("\n  Query Rewriting:")
for q in queries_to_enhance:
    rewritten = rewrite_chain.invoke({"question": q})
    print(f"    Original:  {q!r}")
    print(f"    Rewritten: {rewritten!r}\n")


# 1b. Multi-Query Expansion
EXPANSION_PROMPT = ChatPromptTemplate.from_template(
    """Generate {n} different phrasings of this question that would help find
relevant documents in a search. Each phrasing should approach the topic from
a different angle. Return ONLY the questions, one per line, no numbering.

Question: {question}"""
)

expansion_chain = EXPANSION_PROMPT | llm | StrOutputParser()

q_to_expand = "How are large language models trained?"
expanded    = expansion_chain.invoke({"question": q_to_expand, "n": 3})
variants    = [v.strip() for v in expanded.strip().split("\n") if v.strip()]

print(f"\n  Query Expansion for: {q_to_expand!r}")
for v in variants:
    print(f"    Variant: {v}")

# Retrieve with all variants, deduplicate
all_docs_sets: list[Document] = []
seen_content: set[str] = set()
for variant in variants:
    for doc in retriever.invoke(variant):
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            all_docs_sets.append(doc)
print(f"  Retrieved {len(all_docs_sets)} unique docs (vs {3} with single query)")


# 1c. HyDE — Hypothetical Document Embeddings
HYDE_PROMPT = ChatPromptTemplate.from_template(
    """Write a concise, factual passage (2-3 sentences) that would appear in a
technical document answering this question. This is used for search purposes.

Question: {question}

Hypothetical passage:"""
)

hyde_chain = HYDE_PROMPT | llm | StrOutputParser()

hyde_query      = "What makes transformers better than RNNs?"
hypothetical    = hyde_chain.invoke({"question": hyde_query})
print(f"\n  HyDE for: {hyde_query!r}")
print(f"  Hypothetical doc: {hypothetical[:100]}...")

# Search using the hypothetical doc's embedding (finds real docs close to it)
hyde_docs = vs.similarity_search(hypothetical, k=3)
print(f"  HyDE retrieved: {len(hyde_docs)} docs")
for d in hyde_docs[:2]:
    print(f"    [{d.metadata.get('source')}] {d.page_content[:70]}...")


# ════════════════════════════════════════════════════════════════════
# PART 2: RETRIEVAL VALIDATION
# Check if retrieved documents are relevant and sufficient.
# If not — refine the query and retrieve again.
# ════════════════════════════════════════════════════════════════════

print("\n── 2. Retrieval Validation ──────────────────────────────────")

@dataclass
class RetrievalCheck:
    is_relevant: bool
    is_sufficient: bool
    missing_aspects: str
    refined_query: Optional[str]


RETRIEVAL_CHECK_PROMPT = ChatPromptTemplate.from_template(
    """Evaluate whether the retrieved documents are relevant and sufficient
to answer the user's question.

Question: {question}

Retrieved documents:
{context}

Evaluate and respond in this exact JSON format:
{{
  "is_relevant": true/false,
  "is_sufficient": true/false,
  "missing_aspects": "what information is still missing (or 'none')",
  "refined_query": "a better search query if needed, or null"
}}"""
)

from langchain_openai import ChatOpenAI
structured_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

import json

def validate_retrieval(question: str, docs: list[Document]) -> dict:
    """Validate whether retrieved docs are relevant and sufficient."""
    context  = format_docs(docs)
    response = (RETRIEVAL_CHECK_PROMPT | structured_llm | StrOutputParser()).invoke({
        "question": question,
        "context":  context,
    })
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        return {
            "is_relevant": True, "is_sufficient": True,
            "missing_aspects": "none", "refined_query": None
        }


def retrieve_with_validation(question: str, max_iterations: int = 2) -> list[Document]:
    """Retrieve documents, validate, and refine if needed."""
    current_query = question
    all_docs: list[Document] = []

    for iteration in range(max_iterations):
        docs = retriever.invoke(current_query)
        all_docs.extend(docs)
        print(f"  [Iter {iteration+1}] Query: {current_query[:60]!r} → {len(docs)} docs")

        check = validate_retrieval(question, all_docs)
        print(f"  [Iter {iteration+1}] relevant={check['is_relevant']}, "
              f"sufficient={check['is_sufficient']}")

        if check["is_sufficient"] or not check.get("refined_query"):
            break

        current_query = check["refined_query"]
        print(f"  [Iter {iteration+1}] Refining → {current_query[:60]!r}")

    # Deduplicate
    seen: set[str] = set()
    unique_docs: list[Document] = []
    for d in all_docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            unique_docs.append(d)
    return unique_docs


test_q = "How does RLHF fine-tune language models to align with human values?"
validated_docs = retrieve_with_validation(test_q)
print(f"  Final: {len(validated_docs)} validated docs")


# ════════════════════════════════════════════════════════════════════
# PART 3: ANSWER VALIDATION + SELF-CORRECTION
# Generate an answer, then validate quality/grounding.
# If poor — self-correct by regenerating.
# ════════════════════════════════════════════════════════════════════

print("\n── 3. Answer Validation + Self-Correction ───────────────────")

GENERATE_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the provided context. Be accurate and complete.

Context:
{context}

Question: {question}

Answer (cite sources):"""
)

VALIDATE_PROMPT = ChatPromptTemplate.from_template(
    """Evaluate this answer for quality.

Question: {question}
Context used:
{context}

Answer: {answer}

Respond ONLY with valid JSON:
{{
  "is_grounded": true/false,
  "is_complete": true/false,
  "quality_score": 1-10,
  "issues": "list of issues found (or 'none')",
  "improvement_instruction": "specific instruction to improve the answer (or null)"
}}"""
)


def validate_answer(question: str, context: str, answer: str) -> dict:
    response = (VALIDATE_PROMPT | structured_llm | StrOutputParser()).invoke({
        "question": question,
        "context":  context,
        "answer":   answer,
    })
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        return {"is_grounded": True, "is_complete": True, "quality_score": 8,
                "issues": "none", "improvement_instruction": None}


IMPROVE_PROMPT = ChatPromptTemplate.from_template(
    """Improve this answer based on the instruction provided.

Context:
{context}

Question: {question}

Original answer: {answer}

Improvement instruction: {instruction}

Improved answer:"""
)


def hybrid_rag_pipeline(question: str, max_corrections: int = 2) -> dict:
    """Full hybrid RAG: enhance → retrieve+validate → generate → validate → correct."""
    print(f"\n  Pipeline for: {question!r}")

    # Step 1: Query enhancement
    enhanced_q = rewrite_chain.invoke({"question": question})
    print(f"  [Enhanced] {enhanced_q!r}")

    # Step 2: Retrieve with validation
    docs = retrieve_with_validation(enhanced_q, max_iterations=2)
    ctx  = format_docs(docs)

    # Step 3: Generate initial answer
    answer = (GENERATE_PROMPT | llm | StrOutputParser()).invoke({
        "context":  ctx,
        "question": enhanced_q,
    })
    print(f"  [Generated] {answer[:80]}...")

    # Step 4: Validate + self-correct loop
    for correction_round in range(max_corrections):
        validation = validate_answer(enhanced_q, ctx, answer)
        score = validation.get("quality_score", 8)
        print(f"  [Validate round {correction_round+1}] score={score}/10, "
              f"grounded={validation['is_grounded']}, complete={validation['is_complete']}")

        if score >= 7 and validation["is_grounded"] and validation["is_complete"]:
            print(f"  [Accept] Quality sufficient at score {score}/10")
            break

        instruction = validation.get("improvement_instruction")
        if not instruction:
            break

        print(f"  [Correct] {instruction[:80]}")
        answer = (IMPROVE_PROMPT | llm | StrOutputParser()).invoke({
            "context":     ctx,
            "question":    enhanced_q,
            "answer":      answer,
            "instruction": instruction,
        })

    return {
        "question": question,
        "enhanced_query": enhanced_q,
        "num_docs": len(docs),
        "answer": answer,
        "quality": validation,
    }


# Run hybrid pipeline
results = hybrid_rag_pipeline("what r the main AI models architectures")
print(f"\n  Final answer: {results['answer'][:300]}")
print(f"  Quality score: {results['quality'].get('quality_score', '?')}/10")


# ════════════════════════════════════════════════════════════════════
# PART 4: ADAPTIVE ROUTING — 2-Step vs Hybrid
# Decide which RAG architecture to use based on query complexity.
# ════════════════════════════════════════════════════════════════════

print("\n── 4. Adaptive Routing (choose architecture) ────────────────")

@dataclass
class ArchitectureDecision:
    architecture: Literal["direct", "two_step", "hybrid"]
    reasoning: str


ROUTE_PROMPT = ChatPromptTemplate.from_template(
    """Choose the best RAG architecture for this query.

- direct:   No retrieval needed (math, trivial facts, conversational)
- two_step: Simple, focused query where retrieval is clearly needed
- hybrid:   Ambiguous, complex, or multi-aspect query needing validation

Query: {query}

Respond ONLY with valid JSON:
{{"architecture": "direct|two_step|hybrid", "reasoning": "brief reason"}}"""
)

import re

def route_architecture(query: str) -> str:
    response = (ROUTE_PROMPT | llm | StrOutputParser()).invoke({"query": query})
    # Clean potential markdown code blocks
    cleaned = re.sub(r'```json\s*|\s*```', '', response.strip()).strip()
    try:
        decision = json.loads(cleaned)
        arch = decision.get("architecture", "two_step")
        reason = decision.get("reasoning", "")
        print(f"  [Router] {arch!r} — {reason[:70]}")
        return arch
    except Exception:
        return "two_step"


ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)

def adaptive_rag(query: str) -> str:
    arch = route_architecture(query)

    if arch == "direct":
        return llm.invoke(query).content

    elif arch == "two_step":
        docs   = retriever.invoke(query)
        ctx    = format_docs(docs)
        return (ANSWER_PROMPT | llm | StrOutputParser()).invoke({
            "context": ctx, "question": query
        })

    else:  # hybrid
        result = hybrid_rag_pipeline(query, max_corrections=1)
        return result["answer"]


routing_tests = [
    ("What is 2 + 2?",                              "direct"),
    ("What is RLHF?",                                "two_step"),
    ("Compare transformers vs RNNS for NLP and their training approaches", "hybrid"),
]

print()
for q, expected in routing_tests:
    result = adaptive_rag(q)
    print(f"\n  Q: {q}")
    print(f"  A: {result[:150]}")

print("\n" + "═" * 60)
print("Hybrid RAG Summary:")
print("  Query Enhancement:   rewrite → expand → HyDE")
print("  Retrieval Validation: check relevance → refine → re-retrieve")
print("  Answer Validation:   score → improvement instruction → self-correct")
print("  Adaptive Routing:    direct | two_step | hybrid by complexity")
print("  Best for: high-accuracy needs, ambiguous queries, quality control")
print("  Trade-off: more LLM calls, higher quality output")
print("═" * 60)
print("\n✅ Hybrid RAG demo complete.")
