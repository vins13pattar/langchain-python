"""
01_agent_server.py
===================
Backend: a LangChain agent exposed as a streaming HTTP API.
Frontends (React, Vue, Svelte, plain JS) connect to this server
using the LangGraph streaming protocol.

Concepts covered:
  - create_agent → compiled LangGraph graph
  - Serving via FastAPI + langserve (add_routes)
  - Serving via LangGraph Dev server (langgraph dev)
  - Stream modes: messages, updates, values
  - CORS configuration for frontend access
  - Thread-based conversation management
  - Human-in-the-loop agent (for HITL frontend demo)
  - Structured output agent (for typed UI cards)

Architecture:
  create_agent() ──► LangGraph Graph ──► HTTP streaming API
                                              │
                          ┌───────────────────┘
                          │  SSE stream (text/event-stream)
                          ▼
                    useStream() / fetch()
                      (React, Vue, plain JS)

Running this server:
  Option A — FastAPI (this file):
    pip install fastapi uvicorn langserve
    python 18_frontend/01_agent_server.py

  Option B — LangGraph Dev Server (recommended):
    pip install langgraph-cli
    langgraph dev --host 0.0.0.0 --port 2024
    # reads langgraph.json config

Server endpoints (FastAPI):
  POST /agent/invoke          → single response
  POST /agent/stream          → streaming response
  GET  /agent/playground      → built-in UI
  GET  /health                → health check
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("Agent Server — Backend for Frontend Streaming")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# BUILD AGENTS
# ════════════════════════════════════════════════════════════════════

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "london": "🌧️ Cloudy, 14°C, 85% humidity",
        "new york": "☀️ Sunny, 22°C, 45% humidity",
        "tokyo": "🌤️ Partly cloudy, 19°C, 60% humidity",
        "paris": "⛅ Overcast, 16°C, 70% humidity",
        "sydney": "🌞 Clear, 26°C, 35% humidity",
    }
    result = weather_data.get(city.lower(), f"🌡️ {city}: 20°C, typical conditions")
    print(f"  [Tool] get_weather({city!r}) → {result}")
    return result


@tool
def search_web(query: str) -> str:
    """Search the web for current information."""
    print(f"  [Tool] search_web({query!r})")
    return (
        f"Search results for '{query}': "
        f"Found 3 relevant articles discussing {query}. "
        f"Key findings: [1] Recent developments in this area... "
        f"[2] Expert opinions suggest... [3] Studies show..."
    )


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return "Error: only basic arithmetic allowed"
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        print(f"  [Tool] calculate({expression!r}) → {result}")
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"


# Basic chat agent
chat_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather, search_web, calculate],
    checkpointer=MemorySaver(),
    system_prompt=(
        "You are a helpful AI assistant. You have access to weather lookup, "
        "web search, and calculation tools. Use them when relevant. "
        "Be friendly, concise, and helpful."
    ),
)

# Structured output agent — returns typed JSON
from typing_extensions import TypedDict
from typing import Literal

class WeatherReport(TypedDict):
    city:        str
    temperature: int
    condition:   str
    humidity:    int
    recommendation: str


@tool
def get_detailed_weather(city: str) -> str:
    """Get detailed weather data for structured display."""
    import json
    data = {
        "city": city,
        "temperature": 22,
        "condition": "Partly Cloudy",
        "humidity": 65,
        "recommendation": "Light jacket recommended",
    }
    return json.dumps(data)


structured_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_detailed_weather],
    checkpointer=MemorySaver(),
    system_prompt=(
        "You provide structured weather reports. "
        "Always use get_detailed_weather to get data, "
        "then present it in a friendly way."
    ),
)

# HITL agent — pauses for human approval
from langchain.agents.middleware import HumanInTheLoopMiddleware

hitl_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_web, calculate],
    checkpointer=MemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on=[search_web],   # require approval before web search
        )
    ],
    system_prompt=(
        "You are a careful assistant. For web searches, you'll ask for approval first. "
        "Calculations don't need approval."
    ),
)

print(f"\n✓ Built 3 agents: chat_agent, structured_agent, hitl_agent")


# ════════════════════════════════════════════════════════════════════
# FASTAPI SERVER
# ════════════════════════════════════════════════════════════════════

try:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="LangChain Agent API",
        description="Streaming agent backend for frontend demos",
        version="1.0.0",
    )

    # CORS — allow any origin for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health check ──────────────────────────────────────────────
    @app.get("/health")
    async def health():
        return {"status": "ok", "agents": ["chat", "structured", "hitl"]}

    # ── Streaming endpoint ─────────────────────────────────────────
    from fastapi.responses import StreamingResponse
    import json as json_lib
    import asyncio

    @app.post("/chat/stream")
    async def chat_stream(body: dict):
        """Stream chat agent responses as Server-Sent Events."""
        message     = body.get("message", "")
        thread_id   = body.get("thread_id", "default")

        async def generate():
            config = {"configurable": {"thread_id": thread_id}}
            try:
                async for chunk in chat_agent.astream_events(
                    {"messages": [{"role": "user", "content": message}]},
                    config=config,
                    version="v2",
                ):
                    event_type = chunk.get("event", "")
                    if event_type == "on_chat_model_stream":
                        token = chunk["data"]["chunk"].content
                        if token:
                            data = json_lib.dumps({"type": "token", "content": token})
                            yield f"data: {data}\n\n"
                    elif event_type == "on_tool_start":
                        data = json_lib.dumps({
                            "type": "tool_start",
                            "tool":  chunk["name"],
                            "input": chunk["data"].get("input", {}),
                        })
                        yield f"data: {data}\n\n"
                    elif event_type == "on_tool_end":
                        data = json_lib.dumps({
                            "type":   "tool_end",
                            "tool":   chunk["name"],
                            "output": str(chunk["data"].get("output", ""))[:200],
                        })
                        yield f"data: {data}\n\n"
                yield f"data: {json_lib.dumps({'type': 'done'})}\n\n"
            except Exception as e:
                yield f"data: {json_lib.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.post("/chat/invoke")
    async def chat_invoke(body: dict):
        """Non-streaming chat agent invocation."""
        message   = body.get("message", "")
        thread_id = body.get("thread_id", "default")
        config    = {"configurable": {"thread_id": thread_id}}
        result    = chat_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
        )
        return {
            "response": result["messages"][-1].content,
            "thread_id": thread_id,
        }

    print("\n✓ FastAPI app configured")
    print("  POST /chat/stream  → SSE streaming")
    print("  POST /chat/invoke  → single response")
    print("  GET  /health       → health check")

    # ── Start server ───────────────────────────────────────────────
    if __name__ == "__main__":
        port = int(os.getenv("PORT", "8000"))
        print(f"\n🚀 Starting server at http://localhost:{port}")
        print(f"   Frontend can connect to: http://localhost:{port}/chat/stream")
        print(f"   Open browser: http://localhost:{port}/docs")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

except ImportError as e:
    print(f"\n⚠️  FastAPI/uvicorn not installed: {e}")
    print("   pip install fastapi uvicorn")
    print("\n   Alternative: use LangGraph Dev Server")
    print("   pip install langgraph-cli")
    print("   langgraph dev --port 2024")
    print("\n   This configures the agent as a LangGraph graph")
    print("   that the official useStream() hook connects to.")
    print("\nShowing streaming example without server:")

    # Demo streaming without server
    config = {"configurable": {"thread_id": "demo"}}
    print("\n  Streaming chat agent response:")
    for chunk in chat_agent.stream(
        {"messages": [{"role": "user", "content": "What's 42 * 7?"}]},
        config=config,
        stream_mode="messages",
    ):
        if hasattr(chunk[0], 'content') and chunk[0].content:
            print(f"  Token: {chunk[0].content!r}")

print("\n✅ Agent server setup demo complete.")

# ════════════════════════════════════════════════════════════════════
# LANGGRAPH DEV SERVER CONFIGURATION
# ════════════════════════════════════════════════════════════════════
#
# To use the official LangGraph Dev Server (recommended for frontend):
#
# 1. Create langgraph.json in project root:
#
#    {
#      "dependencies": ["."],
#      "graphs": {
#        "chat_agent":      "./18_frontend/01_agent_server.py:chat_agent",
#        "structured_agent": "./18_frontend/01_agent_server.py:structured_agent",
#        "hitl_agent":      "./18_frontend/01_agent_server.py:hitl_agent"
#      },
#      "env": ".env"
#    }
#
# 2. Start the dev server:
#    langgraph dev --host 0.0.0.0 --port 2024
#
# 3. Frontend useStream() connects to http://localhost:2024
#
# ════════════════════════════════════════════════════════════════════
