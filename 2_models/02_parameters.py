"""
02_parameters.py
================
Demonstrates MODEL PARAMETERS — how to control model behaviour.

Concepts covered:
  - temperature     — creativity vs. determinism
  - max_tokens      — limit output length
  - timeout         — request deadline
  - max_retries     — resilience on rate-limits / network errors
  - Comparing outputs at different temperatures
  - Connection resilience and retry behaviour

The same parameters work whether you use init_chat_model() or a
direct class like ChatOpenAI / ChatAnthropic.
"""

import os
import time
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model

load_dotenv()


# ════════════════════════════════════════════════════════════════════
# 1. TEMPERATURE — creativity dial
# ════════════════════════════════════════════════════════════════════
# temperature=0.0  → deterministic, consistent, factual
# temperature=0.7  → balanced
# temperature=1.0  → creative, variable, sometimes surprising

print("=" * 60)
print("Model Parameters Demo")
print("=" * 60)

PROMPT = "Write one short, creative sentence about the ocean."
RUNS   = 2       # how many times to run each temperature level

for temp in [0.0, 0.5, 1.0]:
    model = init_chat_model("openai:gpt-4o-mini", temperature=temp)
    print(f"\n🌡️  temperature={temp}")
    for i in range(RUNS):
        response = model.invoke(PROMPT)
        print(f"  [{i+1}] {response.content}")


# ════════════════════════════════════════════════════════════════════
# 2. MAX_TOKENS — truncate long outputs
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("max_tokens — controlling output length")
print("─" * 60)

LONG_PROMPT = "Explain quantum computing in detail."

for max_tok in [30, 100, 300]:
    model = init_chat_model("openai:gpt-4o-mini", max_tokens=max_tok)
    response = model.invoke(LONG_PROMPT)
    word_count = len(response.content.split())
    print(f"\n  max_tokens={max_tok:3d}  → ~{word_count} words: {response.content[:120]}…")


# ════════════════════════════════════════════════════════════════════
# 3. TIMEOUT — request deadline
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("timeout — maximum wait for a response")
print("─" * 60)

# A very short timeout will fail on slow connections/models.
# We'll demonstrate the configuration pattern safely.
model_with_timeout = init_chat_model(
    "openai:gpt-4o-mini",
    timeout=30,   # 30 seconds — reasonable for production
)

start = time.time()
response = model_with_timeout.invoke("What is 2 + 2?")
elapsed = time.time() - start

print(f"\n  timeout=30s configured")
print(f"  Response received in {elapsed:.2f}s: {response.content}")


# ════════════════════════════════════════════════════════════════════
# 4. MAX_RETRIES — connection resilience
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("max_retries — automatic retry on rate-limits & network errors")
print("─" * 60)

# Default is 6 retries with exponential backoff + jitter.
# Retried automatically:  network errors, 429 rate-limits, 5xx server errors
# NOT retried:            401 (auth), 404 (not found)

model_resilient = init_chat_model(
    "openai:gpt-4o-mini",
    max_retries=3,    # lower for demos; increase to 10-15 for long agents
    timeout=60,
)
print(f"\n  max_retries=3 configured (default=6)")
response = model_resilient.invoke("Name the planets in our solar system.")
print(f"  Response: {response.content}")


# ════════════════════════════════════════════════════════════════════
# 5. COMBINING PARAMETERS
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("All parameters combined — production-ready configuration")
print("─" * 60)

production_model = init_chat_model(
    "openai:gpt-4o-mini",
    temperature=0.3,      # slightly creative but mostly deterministic
    max_tokens=500,       # cap response length
    timeout=30,           # 30 second deadline
    max_retries=6,        # default — handles transient failures
)

response = production_model.invoke(
    "Summarise the benefits of using LangChain for building AI applications."
)
print(f"\n{response.content}")

# ── Token usage metadata ──────────────────────────────────────────
if hasattr(response, "usage_metadata") and response.usage_metadata:
    usage = response.usage_metadata
    print(f"\n  📊 Token usage:")
    print(f"     Input tokens:  {usage.get('input_tokens', 'N/A')}")
    print(f"     Output tokens: {usage.get('output_tokens', 'N/A')}")
    print(f"     Total tokens:  {usage.get('total_tokens', 'N/A')}")
