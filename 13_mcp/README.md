# 13_mcp — Model Context Protocol (MCP)

> **MCP standardizes how applications provide tools and context to LLMs.**
>
> LangChain agents can consume tools defined on MCP servers using
> [`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters).
> This lets you connect to any MCP-compatible server — local or remote — without
> rewriting your agent code.

---

## Files in this folder

| File | Concepts covered |
|------|-----------------|
| [`servers/math_server.py`](servers/math_server.py) | FastMCP stdio server — add, subtract, multiply, divide |
| [`servers/weather_server.py`](servers/weather_server.py) | FastMCP HTTP server — get_weather, get_forecast, get_air_quality |
| [`servers/rich_server.py`](servers/rich_server.py) | FastMCP HTTP server with structured content, resources, prompts, progress |
| [`01_mcp_basics.py`](01_mcp_basics.py) | `MultiServerMCPClient`, `get_tools()`, stateless vs stateful sessions |
| [`02_mcp_transports.py`](02_mcp_transports.py) | stdio vs HTTP transports, custom headers, multi-server config |
| [`03_mcp_tools_resources_prompts.py`](03_mcp_tools_resources_prompts.py) | Tools, structured content, Resources (Blob), Prompts (messages) |
| [`04_mcp_interceptors.py`](04_mcp_interceptors.py) | Logging, runtime context, store, state auth, `request.override()`, retry, composition |
| [`05_mcp_callbacks.py`](05_mcp_callbacks.py) | `on_progress`, `on_logging_message`, `on_elicitation` (accept/decline/cancel) |
| [`06_full_mcp_showcase.py`](06_full_mcp_showcase.py) | Smart Data Assistant — interceptors + callbacks + middleware + multi-turn memory |

---

## Quick-start

```bash
pip install langchain-mcp-adapters fastmcp
```

**Run a demo (math only — no server needed):**
```bash
python 13_mcp/01_mcp_basics.py
```

**Run demos that need the HTTP servers:**
```bash
# Terminal 1 — weather server
python 13_mcp/servers/weather_server.py

# Terminal 2 — rich server
python 13_mcp/servers/rich_server.py

# Terminal 3 — run the demo
python 13_mcp/03_mcp_tools_resources_prompts.py
```

---

## Core Pattern

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",
                "command":   "python",
                "args":      ["/abs/path/to/math_server.py"],
            },
            "weather": {
                "transport": "http",
                "url":       "http://localhost:8000/mcp",
            },
        }
    )

    tools = await client.get_tools()   # LangChain tools from all servers
    agent = create_agent("openai:gpt-4o-mini", tools)

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What is 3 × 4?"}]
    })
    print(result["messages"][-1].content)

asyncio.run(main())
```

---

## Transports

| Transport | Config key | Best for |
|-----------|-----------|---------|
| `stdio` | `"command"`, `"args"` | Local tools, development |
| `http` | `"url"`, `"headers"` | Remote servers, production |

```python
# stdio — local subprocess
{"transport": "stdio", "command": "python", "args": ["/path/server.py"]}

# http — remote server
{"transport": "http", "url": "http://host:port/mcp", "headers": {"Authorization": "Bearer TOKEN"}}
```

---

## Creating MCP Servers (FastMCP)

```python
from fastmcp import FastMCP

mcp = FastMCP("MyServer")

@mcp.tool()
def search(query: str, limit: int = 10) -> list[str]:
    """Search documents matching a query."""
    return [f"result_{i}" for i in range(limit)]

@mcp.resource("file:///data/config.json")
def get_config() -> str:
    """Application configuration."""
    return '{"version": "1.0"}'

@mcp.prompt()
def summarize(text: str) -> str:
    """Summarize the provided text."""
    return f"Summarize this in 3 bullets:\n\n{text}"

if __name__ == "__main__":
    mcp.run(transport="stdio")        # or "streamable-http"
```

---

## Interceptors

```python
from langchain_mcp_adapters.interceptors import MCPToolCallRequest

async def my_interceptor(request: MCPToolCallRequest, handler):
    # Before: modify request
    user_id  = request.runtime.context.user_id    # runtime context
    is_auth  = request.runtime.state.get("authenticated")  # state
    store    = request.runtime.store               # long-term memory
    call_id  = request.runtime.tool_call_id        # for ToolMessage

    modified = request.override(args={**request.args, "user": user_id})

    # Call the tool
    result = await handler(modified)

    # After: post-process
    return result

client = MultiServerMCPClient(
    {...},
    tool_interceptors=[my_interceptor],   # list = onion composition
)
```

---

## Callbacks

```python
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext

async def on_progress(progress, total, message, context: CallbackContext):
    pct = (progress / total * 100) if total else progress
    print(f"[{context.server_name}] {pct:.0f}% — {message}")

async def on_log(params, context: CallbackContext):
    print(f"[{context.server_name}] {params.level}: {params.data}")

client = MultiServerMCPClient(
    {...},
    callbacks=Callbacks(
        on_progress=on_progress,
        on_logging_message=on_log,
    ),
)
```

---

## Stateful Sessions

```python
from langchain_mcp_adapters.tools import load_mcp_tools

# Stateless (default) — fresh session per tool call
tools = await client.get_tools()

# Stateful — persistent session for stateful servers
async with client.session("server_name") as session:
    tools = await load_mcp_tools(session)
    agent = create_agent("openai:gpt-4o-mini", tools)
```

---

## Key Rules

1. **`MultiServerMCPClient` is stateless by default** — each tool call creates a fresh session.
2. **Use `client.session()` for stateful servers** — servers that maintain state across calls.
3. **Interceptors run in "onion" order** — first in list = outermost layer.
4. **MCP servers can't access LangGraph runtime** — use interceptors to bridge this gap.
5. **stdio requires absolute paths** — always use `Path(__file__).parent / "server.py"`.
6. **HTTP servers must be running before the client connects** — start them first.
