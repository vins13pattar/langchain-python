# 18_frontend — Generative UI & Agent Frontends

> **Build rich, interactive frontends for LangChain agents.**
> Every pattern follows the same architecture: `create_agent` backend streams state to a frontend via `useStream` or plain `fetch()`.

---

## Architecture

```
create_agent()  ──►  LangGraph Graph  ──►  HTTP Streaming API
                                                 │
                               ┌─────────────────┘
                               │  SSE (text/event-stream)
                               ▼
                     useStream()  /  fetch()
                  React · Vue · Svelte · plain JS
```

---

## Files in this folder

| File | Description |
|------|-------------|
| [`01_agent_server.py`](01_agent_server.py) | Python backend — `create_agent` + FastAPI SSE streaming + LangGraph Dev config |
| [`frontend/01_basic_chat.html`](frontend/01_basic_chat.html) | Token streaming chat with real-time tool call cards |
| [`frontend/02_hitl_approval.html`](frontend/02_hitl_approval.html) | Human-in-the-loop — approve / reject / edit tool calls |
| [`frontend/03_full_chat_app.html`](frontend/03_full_chat_app.html) | Full showcase — all patterns in one app (sidebar switcher) |

---

## Quick-start

### 1. Open the HTML files directly (demo mode — no server needed)

```bash
# Open in your browser
open 18_frontend/frontend/03_full_chat_app.html
open 18_frontend/frontend/01_basic_chat.html
open 18_frontend/frontend/02_hitl_approval.html
```

All files run in **demo mode** with simulated responses. No API key or server required.

### 2. Connect to a real backend

```bash
# Option A — FastAPI server (this module)
pip install fastapi uvicorn
python 18_frontend/01_agent_server.py
# → http://localhost:8000

# Option B — LangGraph Dev Server (recommended for useStream)
pip install langgraph-cli
langgraph dev --port 2024
# → http://localhost:2024
```

Then in any HTML file, set `USE_DEMO = false` and update `API_URL`.

---

## Patterns

### ⚡ Streaming (Pattern 1)

Tokens stream in real-time from the agent as it generates them.

**Backend** — SSE streaming with `astream_events`:
```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather, search_web],
    checkpointer=MemorySaver(),
)

# Streaming via astream_events
async for chunk in agent.astream_events(
    {"messages": [{"role": "user", "content": message}]},
    config={"configurable": {"thread_id": thread_id}},
    version="v2",
):
    if chunk["event"] == "on_chat_model_stream":
        token = chunk["data"]["chunk"].content
        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
```

**Frontend** — React with `@langchain/react`:
```tsx
import { useStream } from "@langchain/react";

function Chat() {
  const stream = useStream<{ messages: BaseMessage[] }>({
    apiUrl: "http://localhost:2024",
    assistantId: "agent",
  });

  return (
    <div>
      {stream.messages.map(msg => <Message key={msg.id} message={msg} />)}
      <button onClick={() => stream.submit({ messages: [{ role: "user", content: "Hello" }] })}>
        Send
      </button>
    </div>
  );
}
```

**Frontend** — Vanilla JS (no build step):
```js
const res = await fetch("http://localhost:8000/chat/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message, thread_id }),
});

const reader  = res.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const lines = decoder.decode(value).split("\n");
  for (const line of lines) {
    if (!line.startsWith("data: ")) continue;
    const event = JSON.parse(line.slice(6));
    if (event.type === "token") appendText(event.content);
  }
}
```

---

### ⚙️ Tool Calls (Pattern 2)

Show tool calls as rich UI cards with loading → done states.

```js
// When tool starts: show running card
if (event.type === "tool_start") {
  card = createToolCard(event.tool, event.input);  // "running..." state
  bubble.appendChild(card);
}

// When tool ends: update card with result
if (event.type === "tool_end") {
  finishToolCard(card, event.output);              // "✓ done" + result
}
```

---

### 🛡️ Human-in-the-Loop (Pattern 3)

Agent pauses via `interrupt()`, frontend shows approval UI.

**Backend**:
```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_web],
    checkpointer=MemorySaver(),
    middleware=[HumanInTheLoopMiddleware(interrupt_on=[search_web])],
)
```

**Frontend** — when an interrupt is received:
```js
// useStream detects interrupt in stream.interrupt
if (stream.interrupt) {
  renderApprovalCard({
    tool:  stream.interrupt.value.tool_name,
    args:  stream.interrupt.value.tool_args,
    onApprove: () => stream.submit(null, { command: { resume: "approve" } }),
    onReject:  () => stream.submit(null, { command: { resume: "reject"  } }),
  });
}
```

---

### 🃏 Structured Output (Pattern 4)

Agent returns typed JSON → frontend renders a custom component.

```python
# Backend — structured output
class WeatherReport(TypedDict):
    city: str; temperature: int; condition: str; humidity: int

# Agent uses with_structured_output() or returns JSON
```

```js
// Frontend — render as a card, not plain text
function renderWeatherCard(data) {
  return `<div class="weather-card">
    <div class="city">${data.city}</div>
    <div class="temp">${data.temperature}°C</div>
    <div class="condition">${data.condition}</div>
  </div>`;
}
```

---

### ⏰ Time Travel (Pattern 5)

Navigate conversation history using LangGraph checkpoints.

```python
# List checkpoints for a thread
states = list(agent.get_state_history(
    config={"configurable": {"thread_id": "my-thread"}}
))

# Restore to a specific checkpoint
agent.invoke(
    {"messages": [{"role": "user", "content": "continue from here"}]},
    config={"configurable": {
        "thread_id": "my-thread",
        "checkpoint_id": states[2].config["configurable"]["checkpoint_id"],
    }}
)
```

---

## `useStream` Available Frameworks

```ts
import { useStream } from "@langchain/react";    // React
import { useStream } from "@langchain/vue";       // Vue
import { useStream } from "@langchain/svelte";    // Svelte
import { useStream } from "@langchain/angular";   // Angular
```

## `useStream` Key Properties

| Property | Type | Description |
|----------|------|-------------|
| `stream.messages` | `BaseMessage[]` | All messages in the thread |
| `stream.isLoading` | `boolean` | True while streaming |
| `stream.interrupt` | `Interrupt \| null` | HITL pause state |
| `stream.history` | `Snapshot[]` | Checkpoint history (time travel) |
| `stream.submit()` | `fn` | Send a message or resume after interrupt |
| `stream.stop()` | `fn` | Stop a running stream |

---

## LangGraph Dev Server Config

Create `langgraph.json` in the project root:

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./18_frontend/01_agent_server.py:chat_agent"
  },
  "env": ".env"
}
```

Then run:
```bash
pip install langgraph-cli
langgraph dev --port 2024
```

The frontend connects to `http://localhost:2024` — the same URL used by `useStream()`.

---

## Pattern Comparison

| Pattern | Frontend Complexity | Backend Requirement |
|---------|:---:|---|
| Streaming | ⭐ | `astream_events()` or `langgraph dev` |
| Tool calls | ⭐⭐ | Same, parse `tool_start`/`tool_end` events |
| HITL | ⭐⭐⭐ | `HumanInTheLoopMiddleware` + `MemorySaver` |
| Structured output | ⭐⭐ | `with_structured_output()` |
| Time travel | ⭐⭐⭐ | `MemorySaver` + `get_state_history()` |
