# 2_models — LangChain LLM Models Examples

> **Models are the Reasoning Engine of Agents**
>
> In LangChain, chat models are standard components that interface with various AI providers (OpenAI, Anthropic, Google, etc.). They process text and multimodal inputs, call external tools, and output typed data.

---

## Files in this folder

| File | Concepts covered |
|------|-----------------|
| [`01_init_and_invoke.py`](01_init_and_invoke.py) | `init_chat_model`, `ChatOpenAI`, invocation methods (`invoke`, `stream`, `batch`, `batch_as_completed`) |
| [`02_parameters.py`](02_parameters.py) | Model-level parameters (`temperature`, `max_tokens`, `timeout`, `max_retries`), token usage metadata |
| [`03_structured_output.py`](03_structured_output.py) | `model.with_structured_output()`, Pydantic schemas (`BaseModel`), lightweight schemas (`TypedDict`) |
| [`04_multimodality.py`](04_multimodality.py) | Multimodal content blocks, public image URLs, local base64-encoded image payloads |
| [`05_tool_calling.py`](05_tool_calling.py) | Tool binding (`model.bind_tools()`), manual parsing of `AIMessage.tool_calls`, manually running tools, creating `ToolMessage`s |
| [`models_overview.py`](models_overview.py) | Complete models overview in one file |

---

## Quick-start

```bash
pip install langchain langchain-openai pydantic python-dotenv
echo "OPENAI_API_KEY=sk-..." > .env
python 2_models/01_init_and_invoke.py
```

---

## Core Concepts at a Glance

### 1 — Model Initialization

We recommend using the provider-agnostic `init_chat_model()` wrapper. It allows you to swap model providers with a single string change.

```python
from langchain.chat_models import init_chat_model

# Recommended provider-agnostic factory
model = init_chat_model("openai:gpt-4o-mini")

# Switch providers instantly
# model = init_chat_model("anthropic:claude-3-5-sonnet")
# model = init_chat_model("google_genai:gemini-2.0-flash")
```

---

### 2 — Invocation Methods

LangChain chat models support three distinct execution patterns:

| Invocation Method | Output Mode | Use Case |
|-------------------|-------------|----------|
| `model.invoke()` | Single full message | Simple, synchronous, standard responses |
| `model.stream()` | Live token iterator | Low latency, conversational chatbots |
| `model.batch()` | Parallel list of responses | High throughput, multiple queries concurrently |

```python
# Stream tokens live
for chunk in model.stream("Tell me a story."):
    print(chunk.text, end="", flush=True)

# Run in parallel
responses = model.batch(["Q1...", "Q2...", "Q3..."])
```

---

### 3 — Model-level Structured Output

Force the model to return valid structured data (Pydantic objects or Python dictionaries) directly without complex prompting.

```python
from pydantic import BaseModel, Field

class EntityExtractor(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")
    city: str = Field(description="Home city")

structured_model = model.with_structured_output(EntityExtractor)
result = structured_model.invoke("John is 35 years old and lives in Seattle.")
# Result is a typed EntityExtractor(name='John', age=35, city='Seattle')
```

---

### 4 — Multimodal Inputs

Pass images along with text instructions to native vision models using message content blocks.

```python
from langchain_core.messages import HumanMessage

message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe this image:"},
        {"type": "image_url", "image_url": {"url": "https://picsum.photos/seed/picsum/200/300"}}
    ]
)
response = model.invoke([message])
```

---

### 5 — Manual Tool Calling Loop

Under the hood, an agent is just a loop that binds tools, invokes the model, calls the tools when requested, and passes the results back.

```python
from langchain_core.messages import ToolMessage

# 1. Bind tools to the model
model_with_tools = model.bind_tools([my_tool])

# 2. First invoke: Model decides to call the tool
messages = [HumanMessage("Use my tool please")]
response = model_with_tools.invoke(messages)
messages.append(response)

# 3. Manually execute the tool and return the output via ToolMessage
if response.tool_calls:
    for tool_call in response.tool_calls:
        output = my_tool.invoke(tool_call["args"])
        messages.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))

# 4. Final invoke: Model receives tool results and responds to user
final_response = model_with_tools.invoke(messages)
print(final_response.content)
```

---

## Key Rules

1. **Prefer `init_chat_model()`** — It keeps your codebase provider-independent.
2. **Handle Model Parameters Wisely** — Lower `temperature` (e.g. `0.0`) for code or mathematics; higher `temperature` (e.g. `0.7` to `1.0`) for creative writing.
3. **Use Pydantic for Complex Schema Extraction** — The model reads the docstrings and field `description` attributes as instructions to select what to extract.
4. **Ensure Vision Support** — Vision capabilities require models like `gpt-4o-mini`, `gpt-4o`, `claude-3-5-sonnet`, etc. Check provider docs.
5. **Pair `tool_call_id` Exactly** — When building custom tool-calling loops, the `tool_call_id` of the `ToolMessage` MUST match the tool call's `id` from the model response exactly, otherwise providers will reject the request.
