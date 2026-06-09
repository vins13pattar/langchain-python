"""
models_overview.py — LangChain Models: all key concepts in one file
Covers: init, invoke/stream/batch, parameters, structured output, multimodality, tool calling
"""

import base64
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv()

model = init_chat_model("openai:gpt-4o-mini")

# ── Separators ──────────────────────────────────────────────────────
def section(title): print(f"\n{'─'*50}\n{title}\n{'─'*50}")


# ════════════════════════════════════════════════════════════════════
# 1. INIT & INVOKE
# ════════════════════════════════════════════════════════════════════
section("1. INIT & INVOKE")

# invoke — string input
resp = model.invoke("Why do parrots talk?")
print("String input:", resp.content)

# invoke — message list
resp = model.invoke([
    SystemMessage("Answer in one sentence."),
    HumanMessage("What is a large language model?"),
])
print("Message list:", resp.content)

# stream
print("\nStream tokens: ", end="", flush=True)
for chunk in model.stream("Name 3 programming languages."):
    print(chunk.text, end="|", flush=True)
print()

# batch
prompts = ["Capital of France? (1 word)", "12 × 12? (number only)", "Fastest land animal? (1 word)"]
responses = model.batch(prompts)
for p, r in zip(prompts, responses):
    print(f"  Q: {p}  →  A: {r.content}")


# ════════════════════════════════════════════════════════════════════
# 2. PARAMETERS
# ════════════════════════════════════════════════════════════════════
section("2. PARAMETERS")

# temperature
for temp in [0.0, 0.5, 1.0]:
    m = init_chat_model("openai:gpt-4o-mini", temperature=temp)
    print(f"temp={temp}: {m.invoke('One creative sentence about the ocean.').content}")

# max_tokens
for max_tok in [20, 80]:
    m = init_chat_model("openai:gpt-4o-mini", max_tokens=max_tok)
    resp = m.invoke("Explain quantum computing.")
    print(f"max_tokens={max_tok}: {resp.content[:100]}…")

# production config: temperature + max_tokens + timeout + max_retries
prod = init_chat_model("openai:gpt-4o-mini", temperature=0.3, max_tokens=200, timeout=30, max_retries=6)
resp = prod.invoke("Benefits of LangChain?")
print("\nProduction model response:", resp.content[:200])
if hasattr(resp, "usage_metadata") and resp.usage_metadata:
    u = resp.usage_metadata
    print(f"Tokens — in: {u.get('input_tokens')}  out: {u.get('output_tokens')}  total: {u.get('total_tokens')}")


# ════════════════════════════════════════════════════════════════════
# 3. STRUCTURED OUTPUT
# ════════════════════════════════════════════════════════════════════
section("3. STRUCTURED OUTPUT")

# Pydantic schema — single object
class Movie(BaseModel):
    title: str = Field(description="Movie title")
    director: str = Field(description="Director name")
    release_year: int = Field(description="Release year")
    genres: List[str] = Field(description="List of genres")
    rating: Optional[float] = Field(None, description="Rating out of 10")

structured = model.with_structured_output(Movie)
result = structured.invoke(
    "In 1999, the Wachowskis directed The Matrix — a sci-fi action film rated 8.7."
)
print(f"Movie: {result.title} ({result.release_year}) | Dir: {result.director} | Rating: {result.rating}")

# Pydantic schema — nested list
class MovieList(BaseModel):
    movies: List[Movie] = Field(description="List of movies")

listing = model.with_structured_output(MovieList)
result2 = listing.invoke(
    "Interstellar (2014) by Nolan, sci-fi, 8.6. Inception (2010) by Nolan, action/sci-fi, 8.8."
)
for m in result2.movies:
    print(f"  → {m.title} ({m.release_year}): {m.rating}")

# TypedDict schema — returns raw dict
class BookSchema(TypedDict):
    title: str
    author: str
    publish_year: int
    summary: str

book_model = model.with_structured_output(BookSchema)
book = book_model.invoke("To Kill a Mockingbird by Harper Lee, published 1960. About racial injustice in the South.")
print(f"Book: {book['title']} by {book['author']} ({book['publish_year']})")


# ════════════════════════════════════════════════════════════════════
# 4. MULTIMODALITY
# ════════════════════════════════════════════════════════════════════
section("4. MULTIMODALITY")

# Image via URL
IMAGE_URL = "https://picsum.photos/seed/antigravity/600/400"
msg = HumanMessage(content=[
    {"type": "text", "text": "Describe the main colors and mood of this image."},
    {"type": "image_url", "image_url": {"url": IMAGE_URL}},
])
try:
    resp = model.invoke([msg])
    print("Image URL analysis:", resp.content[:200])
except Exception as e:
    print(f"Vision error: {e}")

# Image via base64 (tiny 1×1 pixel PNG for demo)
b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
b64_msg = HumanMessage(content=[
    {"type": "text", "text": "What color is this image?"},
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
])
try:
    resp2 = model.invoke([b64_msg])
    print("Base64 image analysis:", resp2.content)
except Exception as e:
    print(f"Base64 vision error: {e}")


# ════════════════════════════════════════════════════════════════════
# 5. TOOL CALLING (manual loop)
# ════════════════════════════════════════════════════════════════════
section("5. TOOL CALLING")

@tool
def get_tax_rate(state: str) -> float:
    """Get US sales tax rate for a state code (e.g. 'CA', 'NY', 'TX')."""
    return {"CA": 0.0825, "NY": 0.08875, "TX": 0.0625}.get(state.upper(), 0.0)

@tool
def calculate_subtotal(items: list[dict]) -> float:
    """Calculate subtotal from a list of items with 'price' and 'quantity' keys."""
    return sum(float(i.get("price", 0)) * int(i.get("quantity", 1)) for i in items)

TOOLS = {"get_tax_rate": get_tax_rate, "calculate_subtotal": calculate_subtotal}

model_tools = model.bind_tools(list(TOOLS.values()))

messages = [HumanMessage(
    "2 shirts at $25 each + 1 pair of shoes at $80. "
    "What's my total with California sales tax?"
)]
print(f"User: {messages[0].content}")

# Step 1: model decides which tools to call
resp1 = model_tools.invoke(messages)
messages.append(resp1)
print(f"Tool calls requested: {[tc['name'] for tc in resp1.tool_calls]}")

# Step 2: execute tools and append ToolMessages
for tc in resp1.tool_calls:
    result = TOOLS[tc["name"]].invoke(tc["args"])
    print(f"  {tc['name']}({tc['args']}) → {result}")
    messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

# Step 3: model generates final answer from tool results
resp2 = model_tools.invoke(messages)
print(f"\nFinal Answer: {resp2.content}")
