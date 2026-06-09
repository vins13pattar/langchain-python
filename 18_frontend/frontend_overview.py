"""
frontend_overview.py — Frontend Integration: all key concepts in one file
Covers: building a FastAPI/LangGraph streaming backend, SSE protocol, token streaming,
        HITL agent API, thread-based memory, structured output agent
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("Frontend Integration — Agent Server (Backend for Frontend)")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# BUILD AGENTS
# ════════════════════════════════════════════════════════════════════

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal
from typing_extensions import TypedDict

@tool
def get_weather(city: str) -> str:
    """Get weather for a city. Args: city: City name."""
    return {
        "london": "Cloudy 14°C, 85% humidity",
        "tokyo":  "Sunny 28°C, 60% humidity",
        "paris":  "Clear 22°C, 70% humidity",
    }.get(city.lower(), f"{city}: 20°C, typical conditions")

@tool
def search_web(query: str) -> str:
    """Search the web. Args: query: Search query."""
    return f"Search results for '{query}': 3 relevant articles found."

@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression. Args: expression: e.g. '2 + 2'."""
    try:
        return str(eval(expression, {"__builtins__": {}}))  # noqa: S307
    except Exception as e:
        return f"Error: {e}"

# 1. Basic streaming chat agent
chat_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather, search_web, calculate],
    checkpointer=MemorySaver(),
    system_prompt="You are a helpful assistant with weather, search, and math tools.",
)

# 2. Structured output agent — returns typed JSON
class WeatherReport(TypedDict):
    city:           str
    temperature:    int
    condition:      str
    humidity:       int
    recommendation: str

@tool
def get_detailed_weather(city: str) -> str:
    """Get detailed weather. Args: city: City name."""
    import json
    return json.dumps({"city": city, "temperature": 22, "condition": "Partly Cloudy", "humidity": 65, "recommendation": "Light jacket recommended"})

structured_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_detailed_weather],
    checkpointer=MemorySaver(),
    response_format=WeatherReport,
    system_prompt="You provide structured weather reports. Always use get_detailed_weather.",
)

# 3. HITL agent — pauses for human approval before web search
hitl_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_web, calculate],
    checkpointer=MemorySaver(),
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"search_web": True})],
    system_prompt="You are a careful assistant. Web searches require approval first.",
)

print(f"\n✓ Built 3 agents: chat_agent, structured_agent, hitl_agent")


# ════════════════════════════════════════════════════════════════════
# FASTAPI SERVER
# ════════════════════════════════════════════════════════════════════

try:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    import json as json_lib
    import asyncio

    app = FastAPI(
        title="LangChain Agent API",
        description="Streaming agent backend for React/Vue/plain JS frontends",
        version="1.0.0",
    )

    # Allow all origins for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok", "agents": ["chat", "structured", "hitl"]}

    # ── Streaming — Server-Sent Events (SSE) ─────────────────────────
    @app.post("/chat/stream")
    async def chat_stream(body: dict):
        """Stream chat agent as SSE. Each event is JSON with {type, content}."""
        message   = body.get("message", "")
        thread_id = body.get("thread_id", "default")

        async def generate():
            config = {"configurable": {"thread_id": thread_id}}
            try:
                async for chunk in chat_agent.astream_events(
                    {"messages": [{"role": "user", "content": message}]},
                    config=config, version="v2",
                ):
                    event_type = chunk.get("event", "")
                    if event_type == "on_chat_model_stream":
                        token = chunk["data"]["chunk"].content
                        if token:
                            yield f"data: {json_lib.dumps({'type': 'token', 'content': token})}\n\n"
                    elif event_type == "on_tool_start":
                        yield f"data: {json_lib.dumps({'type': 'tool_start', 'tool': chunk['name'], 'input': chunk['data'].get('input', {})})}\n\n"
                    elif event_type == "on_tool_end":
                        yield f"data: {json_lib.dumps({'type': 'tool_end', 'tool': chunk['name'], 'output': str(chunk['data'].get('output', ''))[:200]})}\n\n"
                yield f"data: {json_lib.dumps({'type': 'done'})}\n\n"
            except Exception as e:
                yield f"data: {json_lib.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    # ── Non-streaming invoke ──────────────────────────────────────────
    @app.post("/chat/invoke")
    async def chat_invoke(body: dict):
        message   = body.get("message", "")
        thread_id = body.get("thread_id", "default")
        config    = {"configurable": {"thread_id": thread_id}}
        result    = chat_agent.invoke({"messages": [{"role": "user", "content": message}]}, config=config)
        return {"response": result["messages"][-1].content, "thread_id": thread_id}

    # ── Structured output endpoint ────────────────────────────────────
    @app.post("/weather/structured")
    async def weather_structured(body: dict):
        city   = body.get("city", "London")
        result = structured_agent.invoke({"messages": [{"role": "user", "content": f"Weather report for {city}"}]})
        return {
            "response": result["messages"][-1].content,
            "structured": result.get("structured_response"),
        }

    # ── HITL endpoints ────────────────────────────────────────────────
    @app.post("/hitl/invoke")
    async def hitl_invoke(body: dict):
        message   = body.get("message", "")
        thread_id = body.get("thread_id", "hitl-session")
        config    = {"configurable": {"thread_id": thread_id}}
        result    = hitl_agent.invoke({"messages": [{"role": "user", "content": message}]}, config=config, version="v2")
        if result.interrupts:
            action = result.interrupts[0].value["action_requests"][0]
            return {
                "status": "interrupted",
                "action": action["name"],
                "args":   action["arguments"],
                "thread_id": thread_id,
            }
        return {"status": "complete", "response": result.value["messages"][-1].content, "thread_id": thread_id}

    @app.post("/hitl/resume")
    async def hitl_resume(body: dict):
        from langgraph.types import Command
        thread_id = body.get("thread_id", "hitl-session")
        decision  = body.get("decision", "approve")   # "approve" | "reject"
        config    = {"configurable": {"thread_id": thread_id}}
        result    = hitl_agent.invoke(Command(resume={"decisions": [{"type": decision}]}), config=config, version="v2")
        return {"status": "complete", "response": result.value["messages"][-1].content}

    print("\n✓ FastAPI endpoints registered:")
    print("  GET  /health")
    print("  POST /chat/stream     → SSE token streaming")
    print("  POST /chat/invoke     → single response")
    print("  POST /weather/structured → structured TypedDict response")
    print("  POST /hitl/invoke     → HITL agent (pauses on search_web)")
    print("  POST /hitl/resume     → resume with approve/reject decision")

    if __name__ == "__main__":
        port = int(os.getenv("PORT", "8000"))
        print(f"\n🚀 Starting at http://localhost:{port}")
        print(f"   Docs: http://localhost:{port}/docs")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

except ImportError as e:
    print(f"\n⚠ FastAPI/uvicorn not installed: {e}")
    print("  pip install fastapi uvicorn")

    # Demo streaming without server
    config = {"configurable": {"thread_id": "demo"}}
    print("\nDemonstrating streaming without server:")
    for chunk in chat_agent.stream(
        {"messages": [{"role": "user", "content": "What's 99 * 12?"}]},
        config=config, stream_mode="messages",
    ):
        msg = chunk[0] if isinstance(chunk, tuple) else chunk
        if hasattr(msg, "content") and msg.content:
            print(f"  token: {msg.content!r}", end="", flush=True)
    print()


print("""
Frontend Integration Guide:

Option A — FastAPI SSE server (this file):
  pip install fastapi uvicorn
  python 18_frontend/01_agent_server.py
  
  Frontend (JS) connects to:
    POST http://localhost:8000/chat/stream   → text/event-stream
    Parse events: {type: "token", content: "..."} | {type: "tool_start"} | {type: "done"}

Option B — LangGraph Dev Server (recommended for React):
  pip install langgraph-cli
  langgraph dev --host 0.0.0.0 --port 2024   (reads langgraph.json)

  Frontend uses the official useStream() hook:
    import { useStream } from "@langchain/langgraph-sdk/react";
    const { values, submit } = useStream({
      apiUrl: "http://localhost:2024",
      assistantId: "chat_agent",
      threadId: "my-thread",
    });

langgraph.json (put in project root):
  {
    "dependencies": ["."],
    "graphs": {
      "chat_agent":      "./18_frontend/01_agent_server.py:chat_agent",
      "hitl_agent":      "./18_frontend/01_agent_server.py:hitl_agent",
      "structured_agent":"./18_frontend/01_agent_server.py:structured_agent"
    },
    "env": ".env"
  }

SSE Event types (streaming):
  {type: "token", content: "..."}       → append to output buffer
  {type: "tool_start", tool: "...", ...} → show tool call indicator
  {type: "tool_end", tool: "...", ...}   → show result
  {type: "done"}                         → complete
  {type: "error", message: "..."}        → handle error
""")
