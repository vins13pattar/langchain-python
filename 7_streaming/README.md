# 7_streaming — LangGraph Graph Streaming Examples

> **Lower-Level Graph-Step Streaming API**
>
> In addition to the modern v3 Event Streaming API (`stream_events`), LangGraph agents support step-level streaming using `agent.stream(..., stream_mode=...)`. This API yields state snapshots or token streams at the completion of each individual graph node (e.g. after the model finishes reasoning, or after tools finish executing).

---

## Files in this folder

| File | Concepts covered |
|------|-----------------|
| [`01_updates_mode.py`](01_updates_mode.py) | `stream_mode="updates"`, filtering increment events, extracting node additions, multi-turn `thread_id` checks |
| [`02_values_mode.py`](02_values_mode.py) | `stream_mode="values"`, yielding full state snapshots, tracking message list growth, compared directly with updates mode |
| [`03_messages_mode.py`](03_messages_mode.py) | `stream_mode="messages"`, token-by-token text deltas, type-checking `AIMessageChunk`, chunk addition/accumulation |
| [`04_debug_mode.py`](04_debug_mode.py) | `stream_mode="debug"`, complete verbose tracing, node execution inputs/outputs, thread state checkpointing |
| [`05_full_streaming_showcase.py`](05_full_streaming_showcase.py) | An interactive, production-ready console assistant featuring thread memory checkpointers and live token streaming |
| [`streaming_overview.py`](streaming_overview.py) | Complete streaming overview in one file |

---

## Graph Streaming Modes

The `stream_mode` parameter in `agent.stream()` defines what each yielded chunk represents:

| Mode | Yields | Perfect For |
|------|--------|-------------|
| `"updates"` | Dictionary containing only changed values at the active node | Multi-step tracking, knowing when a specific tool executes or when the model finishes a step |
| `"values"` | Dictionary representing the entire current state database | Simple frontends, direct UI state replacements without state aggregation |
| `"messages"` | Raw message token chunks (`AIMessageChunk`) from the model | Classic chatbot UX with live, fast word-by-word streaming |
| `"debug"` | Detailed system trace dictionaries (checkpoints, tasks, states) | Local troubleshooting, visual graphs, performance inspection |

---

## Understanding the v2 Streaming Envelope

Under protocol `"v2"`, the chunks returned by `agent.stream()` are wrapped in a standard event envelope:

```python
for chunk in agent.stream(input, stream_mode="messages", version="v2"):
    # chunk is: {"type": "<stream_mode>", "ns": (...), "data": <payload>}
```

### Unpacking by Mode

#### 1 — Updates Mode (`stream_mode="updates"`)
```python
if chunk["type"] == "updates":
    for node_name, node_data in chunk["data"].items():
        new_messages = node_data.get("messages", [])
```

#### 2 — Values Mode (`stream_mode="values"`)
```python
if chunk["type"] == "values":
    state = chunk["data"]
    all_messages = state.get("messages", [])
```

#### 3 — Messages Mode (`stream_mode="messages"`)
```python
if chunk["type"] == "messages":
    msg_chunk, metadata = chunk["data"]
    # msg_chunk is typically AIMessageChunk
```

---

## Quick-start

Run the interactive terminal chatbot to see token streaming and memory in action:

```bash
pip install langchain langchain-openai langgraph python-dotenv
python 7_streaming/05_full_streaming_showcase.py
```

---

## Key Rules

1. **Unpack with Version checking** — If you run `agent.stream()` with `version="v2"`, you must access values inside `chunk["data"]` instead of directly index-accessing `chunk` keys.
2. **Combine Chunks with `+`** — In `messages` mode, reconstruct the finalized model message by summing the `AIMessageChunk` objects as they arrive: `full_msg = full_msg + chunk`.
3. **Use MemorySaver for Threading** — Step streaming works standalone, but thread memory (`MemorySaver`) is required to preserve conversation history across turns using a custom `thread_id`.
4. **Prefer `Event Streaming` for Complex Apps** — For web applications that require concurrent streaming of tool calls, text deltas, and state values, use `stream_events()` (see `6_event_streaming`), which provides named projection channels instead of low-level `stream_mode` tuple unpacking.
