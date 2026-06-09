# 3_messages — LangChain Messages Examples

> Messages are the **fundamental unit of context** for models in LangChain.
> They represent the input and output of every model call, carrying both
> content and metadata.

---

## What is a Message?

Every call to a model is a list of messages in → one AIMessage out.

```
[SystemMessage, HumanMessage, AIMessage, HumanMessage, …]
         │
         ▼
    model.invoke()
         │
         ▼
    AIMessage  ←── content + tool_calls + usage_metadata
```

---

## Files in this folder

| File | Concepts covered |
|------|-----------------|
| [`01_message_types.py`](01_message_types.py) | `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage` + 3 input formats |
| [`02_conversation_history.py`](02_conversation_history.py) | Multi-turn loops, injecting AI messages, branching, dict format |
| [`03_message_content_and_streaming.py`](03_message_content_and_streaming.py) | `content_blocks`, `stream()`, `AIMessageChunk`, `astream_events()` |
| [`04_tool_message_loop.py`](04_tool_message_loop.py) | `bind_tools`, `tool_call_id`, parallel calls, `artifact`, forced tool choice |
| [`05_multimodal_messages.py`](05_multimodal_messages.py) | Image from URL / base64 / local file, content block format reference |
| [`06_full_messages_showcase.py`](06_full_messages_showcase.py) | All concepts combined — conversation, tools, streaming, few-shot, multimodal |
| [`messages_overview.py`](messages_overview.py) | Complete messages overview in one file |

---

## Quick-start

```bash
# 1. Install dependencies
pip install langchain langchain-openai python-dotenv httpx

# 2. Set your API key
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Run any example
python 01_message_types.py
```

---

## Message Types at a Glance

### SystemMessage
```python
from langchain_core.messages import SystemMessage

SystemMessage("You are a helpful assistant.")
```
- Sets model persona, tone, rules
- Applied before all user messages
- One per conversation (typically)

### HumanMessage
```python
from langchain_core.messages import HumanMessage

# Text only
HumanMessage("What is Python?")

# With metadata
HumanMessage(content="Hello!", name="vinod", id="msg-001")

# Multimodal (image + text)
HumanMessage(content=[
    {"type": "text", "text": "Describe this image."},
    {"type": "image", "url": "https://example.com/img.jpg"},
])
```

### AIMessage
```python
response = model.invoke(messages)   # → AIMessage

response.content          # str or list — raw content
response.text             # str alias for content
response.content_blocks   # standardised cross-provider blocks
response.tool_calls       # tool call requests (if any)
response.usage_metadata   # {"input_tokens": N, "output_tokens": M, ...}
response.id               # unique message identifier
```

### ToolMessage
```python
from langchain_core.messages import ToolMessage

ToolMessage(
    content="Result of the tool",
    tool_call_id="call_abc123",   # MUST match AIMessage tool call ID
    name="get_weather",
    artifact={"raw": "...", "page": 0},  # NOT sent to model
)
```

---

## Three Equivalent Input Formats

```python
# 1. Plain string (shorthand for single HumanMessage)
model.invoke("What is 2 + 2?")

# 2. Dict — OpenAI chat completions style
model.invoke([
    {"role": "system",    "content": "Be brief."},
    {"role": "user",      "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user",      "content": "Times 10?"},
])

# 3. Message objects — most explicit
model.invoke([
    SystemMessage("Be brief."),
    HumanMessage("What is 2 + 2?"),
    AIMessage("4"),
    HumanMessage("Times 10?"),
])
```

---

## Tool Calling Message Flow

```
HumanMessage("What's the weather in Tokyo?")
    │
    ▼ model.invoke()
AIMessage(tool_calls=[{name:"get_weather", args:{city:"Tokyo"}, id:"call_1"}])
    │
    ▼ execute tool
ToolMessage(content="Sunny 28°C", tool_call_id="call_1")
    │
    ▼ model.invoke()
AIMessage(content="The weather in Tokyo is sunny and 28°C.")
```

---

## Content Block Types (Standard)

| type | required fields | use for |
|------|----------------|---------|
| `"text"` | `text` | text response |
| `"reasoning"` | `reasoning` | chain-of-thought |
| `"image"` | `url` or (`base64` + `mime_type`) | image input/output |
| `"audio"` | `url` or (`base64` + `mime_type`) | audio input/output |
| `"video"` | `url` or (`base64` + `mime_type`) | video input |
| `"file"` | `url` or (`base64` + `mime_type`) | PDF / docs |
| `"tool_call"` | `name`, `args`, `id` | tool request |

---

## Key Rules

1. **Always append both sides** — after each turn, append the `HumanMessage` AND the model's `AIMessage` reply to history.
2. **`tool_call_id` must match** — `ToolMessage.tool_call_id` must equal the `id` in `AIMessage.tool_calls`.
3. **`content_blocks` is read-only** — it's a parsed view of `content`; set `content` to change the message.
4. **Strings are HumanMessages** — `model.invoke("hello")` ≡ `model.invoke([HumanMessage("hello")])`.
5. **Inject AI messages for few-shot** — add `AIMessage("example")` to history to steer model style without retraining.
