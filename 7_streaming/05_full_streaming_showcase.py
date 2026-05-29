"""
05_full_streaming_showcase.py
==============================
A complete, interactive terminal chatbot demonstrating real-time token streaming
and state management using stream_mode="messages".

Features:
  - Persistent conversation state using thread checkpoints (MemorySaver)
  - Interactive user prompt in a continuous loop
  - Live token streaming to stdout using a custom typing/printing wrapper
  - Visual tool execution indicator (displays tool names and args while executing)
  - Clean error boundaries and simple command exits ('exit', 'quit', 'clear')
"""

import os
import sys
import time
import uuid
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ── 1. Tools ──────────────────────────────────────────────────────────

@tool
def get_current_time() -> str:
    """Get the current time and date in a readable format."""
    return time.strftime("🕒 %Y-%m-%d %H:%M:%S Local Time")


@tool
def query_knowledge_base(query: str) -> str:
    """Query the local knowledge base for specific technical information.

    Args:
        query: Technical search term or question
    """
    # Simulated quick KB lookup
    kb_data = {
        "langchain": "LangChain is a framework for developing applications powered by large language models.",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent workflows.",
        "streaming": "LangGraph supports three graph streaming modes: updates (node additions), values (full state), and messages (raw tokens).",
    }
    
    # Simulate searching...
    time.sleep(0.5)
    
    q_lower = query.lower()
    for key, val in kb_data.items():
        if key in q_lower:
            return f"📚 KB Match: {val}"
    return f"📚 KB: No exact matching articles found for '{query}'."


# ── 2. Initialize Agent ───────────────────────────────────────────────

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_current_time, query_knowledge_base],
    checkpointer=MemorySaver(),
    system_prompt=(
        "You are 'Antigravity-Lite', a premium console assistant. "
        "Use the tools at your disposal when asked about time or technical facts. "
        "Keep your tone professional, concise, and helpful."
    ),
)


# ── 3. Streaming Runner ────────────────────────────────────────────────

def run_chat_turn(user_input: str, thread_id: str) -> None:
    """Run a single interaction turn, streaming model chunks to stdout."""
    config = {"configurable": {"thread_id": thread_id}}
    state_input = {"messages": [{"role": "user", "content": user_input}]}
    
    print("\n🤖 ", end="", flush=True)
    
    in_tool_call = False
    
    # Stream using low-level messages mode
    for chunk in agent.stream(state_input, config=config, stream_mode="messages", version="v2"):
        
        # Unpack v2 envelope
        if isinstance(chunk, dict) and chunk.get("type") == "messages":
            payload = chunk.get("data")
            msg_chunk = payload[0] if isinstance(payload, tuple) else payload
        else:
            msg_chunk = chunk[0] if isinstance(chunk, tuple) else chunk
            
        # We process AIMessageChunks for terminal token delivery
        if isinstance(msg_chunk, AIMessageChunk):
            
            # 1. Tool Call detection
            # Check if the model has initiated a tool call but hasn't finalized it
            if msg_chunk.tool_call_chunks:
                if not in_tool_call:
                    print("\n🔧 [System: Model binding tools...]", end="", flush=True)
                    in_tool_call = True
                continue  # skip printing tool-call arguments directly
                
            # 2. Text generation
            # Print standard text tokens
            if msg_chunk.content:
                # If we were previously in a tool-call indicator block, add a new line
                if in_tool_call:
                    print("\n", end="", flush=True)
                    in_tool_call = False
                
                text = msg_chunk.content if isinstance(msg_chunk.content, str) else str(msg_chunk.content)
                sys.stdout.write(text)
                sys.stdout.flush()

    print("\n")


# ── 4. Main Interactive Loop ──────────────────────────────────────────

def main():
    session_id = str(uuid.uuid4())[:8]
    thread_id = f"cli-session-{session_id}"
    
    print("=" * 60)
    print("      🚀 Welcome to Antigravity-Lite Terminal Chatbot 🚀")
    print("=" * 60)
    print(f"Session Thread ID: {thread_id}")
    print("Commands: 'exit' or 'quit' to end | 'clear' to reset memory\n")
    
    while True:
        try:
            user_input = input("🧑 User: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye! Have an excellent day. 👋")
                break
                
            if user_input.lower() == "clear":
                thread_id = f"cli-session-{str(uuid.uuid4())[:8]}"
                print(f"\n🔄 State reset! New thread ID: {thread_id}\n")
                continue
                
            # Run the agent turn with streaming
            run_chat_turn(user_input, thread_id)
            print("─" * 60)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}\n")


if __name__ == "__main__":
    main()
