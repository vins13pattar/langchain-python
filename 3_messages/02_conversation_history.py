"""
02_conversation_history.py
==========================
Demonstrates building and managing MULTI-TURN conversations with messages.

Concepts covered:
  - Growing a message list across turns (stateless pattern)
  - Correctly ordering SystemMessage → HumanMessage → AIMessage → …
  - Injecting fake AI messages into history
  - Passing conversation history to get context-aware replies
  - When to use strings vs message objects vs dicts

Without a checkpointer (like in agents), the model itself is STATELESS.
You own the conversation list and append to it on every turn.
The model only knows what's in the list you pass each time.
"""

import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

load_dotenv()

model = init_chat_model("openai:gpt-4o-mini", temperature=0.7)

print("=" * 60)
print("Conversation History Demo")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# 1. STATELESS TURN-BY-TURN CONVERSATION
#    Manually grow the message list each turn.
# ════════════════════════════════════════════════════════════════════

print("\n── 1. Manual conversation loop ───────────────────────────")

# Start with a system message that persists for the whole conversation
conversation: list = [
    SystemMessage(
        "You are a friendly Python tutor. "
        "Keep answers short (2-3 sentences). "
        "Remember everything the student tells you."
    )
]

def chat(user_text: str) -> str:
    """Send a message, get a reply, and update the conversation history."""
    # Append the new user message
    conversation.append(HumanMessage(user_text))

    # Invoke the model with the FULL history
    response = model.invoke(conversation)

    # Append the model's reply to history (so next turn has context)
    conversation.append(response)   # response is an AIMessage

    return response.content


# Simulate a 4-turn conversation
turns = [
    "Hi! My name is Vinod and I am learning Python.",
    "What is a list comprehension?",
    "Can you show me a simple example?",
    "What was my name again?",   # Tests recall — model should remember "Vinod"
]

for user_input in turns:
    reply = chat(user_input)
    print(f"\n🧑 {user_input}")
    print(f"🤖 {reply}")

print(f"\n  Total messages in history: {len(conversation)}")
print(f"  Breakdown:")
for i, msg in enumerate(conversation):
    role = type(msg).__name__.replace("Message", "")
    preview = (msg.content if isinstance(msg.content, str) else str(msg.content))[:60]
    print(f"    [{i}] {role:8s}: {preview}…")


# ════════════════════════════════════════════════════════════════════
# 2. DICT FORMAT CONVERSATION (OpenAI style)
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("\n── 2. Dict-format conversation ───────────────────────────")

dict_conversation = [
    {"role": "system",    "content": "You are a helpful assistant. Be brief."},
    {"role": "user",      "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a high-level, interpreted programming language known for its simplicity."},
    {"role": "user",      "content": "What makes it popular for AI?"},
]

response = model.invoke(dict_conversation)
print(f"\n🤖 {response.content}")


# ════════════════════════════════════════════════════════════════════
# 3. INJECTING A FAKE AI MESSAGE
#    Useful for few-shot prompting or steering the model's "memory"
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("\n── 3. Injecting a fake AIMessage into history ────────────")
print("  (Few-shot: teach the model a response style by example)")

few_shot_messages = [
    SystemMessage("You are a pirate assistant. Always speak like a pirate."),

    # Injected example exchange — model never actually said this,
    # but including it as "history" steers the model's style
    HumanMessage("What is the weather today?"),
    AIMessage("Arrr, the skies be clear as the Caribbean sea, matey!"),

    HumanMessage("What is Python?"),
    AIMessage("Ahoy! Python be a treasure of a language, beloved by coders far and wide!"),

    # Now the real user question
    HumanMessage("Explain list comprehensions."),
]

response = model.invoke(few_shot_messages)
print(f"\n🤖 {response.content}")


# ════════════════════════════════════════════════════════════════════
# 4. BRANCHING CONVERSATIONS
#    Same base history → different continuations
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("\n── 4. Branching from shared history ──────────────────────")

shared_base = [
    SystemMessage("You are a concise technical assistant."),
    HumanMessage("I want to learn about Python data structures."),
    AIMessage("Great! Python has four main built-in data structures: lists, tuples, sets, and dictionaries. Which would you like to explore?"),
]

# Branch A: dive into lists
branch_a = shared_base + [HumanMessage("Tell me about lists.")]
r_a = model.invoke(branch_a)

# Branch B: dive into dictionaries
branch_b = shared_base + [HumanMessage("Tell me about dictionaries.")]
r_b = model.invoke(branch_b)

print(f"\nBranch A (lists):")
print(f"  {r_a.content[:200]}…")
print(f"\nBranch B (dicts):")
print(f"  {r_b.content[:200]}…")


# ════════════════════════════════════════════════════════════════════
# 5. INSPECTING AIMessage METADATA
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("\n── 5. AIMessage metadata ────────────────────────────────")

response = model.invoke("What is 17 × 34?")

print(f"\nContent:           {response.content}")
print(f"Type:              {type(response).__name__}")
print(f"ID:                {response.id}")

if response.response_metadata:
    meta = response.response_metadata
    print(f"Response metadata: {list(meta.keys())}")

if response.usage_metadata:
    u = response.usage_metadata
    print(f"Token usage:")
    print(f"  input_tokens:  {u.get('input_tokens')}")
    print(f"  output_tokens: {u.get('output_tokens')}")
    print(f"  total_tokens:  {u.get('total_tokens')}")
