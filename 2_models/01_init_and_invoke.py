"""
01_init_and_invoke.py
=====================
Demonstrates the TWO ways to initialize a LangChain chat model and the
THREE invocation methods: invoke, stream, and batch.

Concepts covered:
  - init_chat_model()   — provider-agnostic factory (recommended)
  - ChatOpenAI()        — direct class instantiation
  - model.invoke()      — single request, full response
  - model.stream()      — real-time token-by-token output
  - model.batch()       — parallel requests for efficiency

Models are the REASONING ENGINE of agents. The same interface works
standalone (like this file) and inside create_agent().
"""

import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()


# ════════════════════════════════════════════════════════════════════
# 1. INITIALIZING MODELS
# ════════════════════════════════════════════════════════════════════

# ── Method A: init_chat_model() — recommended ─────────────────────
# Provider-agnostic: change the string to swap providers with zero
# other code changes.  Format: "provider:model" or just "model".

model_a = init_chat_model("openai:gpt-4o-mini")               # OpenAI
# model_a = init_chat_model("anthropic:claude-haiku-3-5")     # Anthropic (swap here)
# model_a = init_chat_model("google_genai:gemini-2.0-flash")  # Google Gemini

# ── Method B: Direct class instantiation ─────────────────────────
# Gives access to provider-specific parameters not exposed by init_chat_model.

model_b = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=500,
    timeout=30,
    max_retries=3,
)

print("=" * 60)
print("Model Initialization & Invocation Demo")
print("=" * 60)
print(f"\nModel A (init_chat_model): {model_a.__class__.__name__}")
print(f"Model B (ChatOpenAI):      {model_b.__class__.__name__}")


# ════════════════════════════════════════════════════════════════════
# 2. INVOKE — single request, waits for full response
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("INVOKE — full response returned after generation completes")
print("─" * 60)

# ── 2a. Simple string input ───────────────────────────────────────
print("\n📌 2a. Simple string input")
response = model_a.invoke("Why do parrots talk?")
print(f"Type:    {type(response).__name__}")    # AIMessage
print(f"Content: {response.content}")
print(f"Text:    {response.text}")              # convenience alias for .content

# ── 2b. Dict format (conversation history) ────────────────────────
print("\n📌 2b. Dictionary message format")
conversation = [
    {"role": "system",    "content": "You are a translator. Be brief."},
    {"role": "user",      "content": "Translate: I love programming."},
    {"role": "assistant", "content": "J'adore la programmation."},
    {"role": "user",      "content": "Translate: The sky is blue."},
]
response = model_a.invoke(conversation)
print(f"Translation: {response.content}")

# ── 2c. Message objects ───────────────────────────────────────────
print("\n📌 2c. Message object format")
messages = [
    SystemMessage("Answer in exactly one sentence."),
    HumanMessage("What is a large language model?"),
]
response = model_a.invoke(messages)
print(f"Answer: {response.content}")


# ════════════════════════════════════════════════════════════════════
# 3. STREAM — real-time token output
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("STREAM — tokens printed as they are generated")
print("─" * 60)

print("\n📌 3a. Basic text streaming (tokens separated by |)")
print("Output: ", end="", flush=True)
for chunk in model_a.stream("Name 5 programming languages in a comma-separated list."):
    print(chunk.text, end="|", flush=True)
print()   # newline

# ── Accumulate chunks into a full AIMessage ───────────────────────
print("\n📌 3b. Accumulating chunks into a complete message")
full = None
for chunk in model_a.stream("What is 7 × 8?"):
    full = chunk if full is None else full + chunk

print(f"Full content: {full.content}")
print(f"Full type:    {type(full).__name__}")   # AIMessageChunk sums to same interface


# ════════════════════════════════════════════════════════════════════
# 4. BATCH — parallel requests
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("BATCH — multiple independent prompts in parallel")
print("─" * 60)

prompts = [
    "What is the capital of France? (1 word)",
    "What is 12 × 12? (number only)",
    "Name the fastest land animal. (1 word)",
]

print("\n📌 4a. batch() — waits for all to finish")
responses = model_a.batch(prompts)
for prompt, resp in zip(prompts, responses):
    print(f"  Q: {prompt}")
    print(f"  A: {resp.content}\n")

print("📌 4b. batch_as_completed() — yields results as each finishes")
print("  (Results may arrive out of order — includes original index)\n")
for idx, resp in model_a.batch_as_completed(prompts):
    print(f"  [{idx}] {resp.content}")

print("\n📌 4c. batch() with max_concurrency")
responses = model_a.batch(
    prompts,
    config={"max_concurrency": 2},   # at most 2 parallel API calls
)
print(f"  Got {len(responses)} responses (max 2 at a time)")
