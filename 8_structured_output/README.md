# 8_structured_output — LangChain Structured Output Examples

> **Structured Output Guarantees Typed, Validated Agent Returns**
>
> Structured output allows agents to return data in a specific, predictable format. Instead of parsing natural language text or raw chat blocks, you receive structured data in the form of JSON objects, Pydantic models, or Python dataclasses that your application code can consume directly.

---

## Files in this folder

| File | Concepts covered |
|------|-----------------|
| [`01_auto_strategy.py`](01_auto_strategy.py) | Direct schema passing (`response_format=ContactPydantic`), auto-strategy resolution for Pydantic models, Python dataclasses, and TypedDicts |
| [`02_provider_strategy.py`](02_provider_strategy.py) | `ProviderStrategy`, native provider-enforced schemas, enabling `strict=True` validation (e.g. OpenAI Structured Outputs), raw JSON Schemas |
| [`03_tool_strategy_basics.py`](03_tool_strategy_basics.py) | `ToolStrategy`, forcing tool calling for schema extraction, matching parameters |
| [`04_custom_tool_message.py`](04_custom_tool_message.py) | Customizing conversation history representation using the `tool_message_content` parameter to hide internal JSON blocks |
| [`05_error_handling_and_retries.py`](05_error_handling_and_retries.py) | Intelligent retry loops, catching `StructuredOutputValidationError` and `MultipleStructuredOutputsError`, custom retry prompts (strings, lists, callable handlers) |
| [`06_full_structured_output_showcase.py`](06_full_structured_output_showcase.py) | Complete case study: Support ticket classifier using tools and structured extraction simultaneously with robust error limits |

---

## Quick-start

```bash
pip install -r requirements.txt
python 8_structured_output/01_auto_strategy.py
```

---

## Response Format Strategies

The `response_format` parameter in `create_agent` determines how the agent parses the response:

| Strategy | Code | How it Works | Best For |
|----------|------|--------------|----------|
| **Auto Selection** | `response_format=SchemaClass` | Automatically chooses `ProviderStrategy` if model natively supports it, otherwise `ToolStrategy` | standard schemas |
| **Provider Strategy** | `response_format=ProviderStrategy(Schema)` | Uses native model provider API (e.g. OpenAI Structured Outputs, Gemini Schemas) | maximum reliability and strict parsing |
| **Tool Strategy** | `response_format=ToolStrategy(Schema)` | Creates a virtual tool under the hood and instructs the model to call it | legacy models, custom error loops, custom tool message descriptions |

---

## Core Patterns

### 1 — Auto Strategy (Pydantic, Dataclass, TypedDict)

Pass any Pydantic model, dataclass, or TypedDict class directly to the agent. The result is returned under `result["structured_response"]`.

```python
from pydantic import BaseModel
from langchain.agents import create_agent

class Contact(BaseModel):
    name: str
    email: str

agent = create_agent(model="openai:gpt-4o-mini", response_format=Contact)
result = agent.invoke({"messages": [{"role": "user", "content": "Extract: John john@example.com"}]})
contact: Contact = result["structured_response"]
```

---

### 2 — Strict Provider Strategy

For maximum schema adherence, explicitly configure `ProviderStrategy` with `strict=True`.

> [!NOTE]
> If passing a raw JSON Schema directly, a top-level `"title"` key is required to serve as the schema function name:

```python
from langchain.agents.structured_output import ProviderStrategy

contact_schema = {
    "title": "ContactInfo",
    "type": "object",
    "properties": {
        "name": {"type": "string"}
    },
    "required": ["name"]
}

agent = create_agent(
    model="openai:gpt-4o-mini",
    response_format=ProviderStrategy(schema=contact_schema, strict=True)
)
```

---

### 3 — Custom History Tool Messages

Hide raw JSON blocks inside your conversation history logs using `tool_message_content`.

```python
from langchain.agents.structured_output import ToolStrategy

response_format = ToolStrategy(
    schema=MeetingAction,
    tool_message_content="Action item successfully captured and logged!"
)
```

This changes the generated `ToolMessage` inside `result["messages"]` from:
> *Returning structured response: {'task': 'Update timeline', 'assignee': 'Sarah'}*

to:
> *Action item successfully captured and logged!*

---

### 4 — Intelligent Error Retrying

When a model fails Pydantic schema constraint checks (e.g. `ge=1, le=5` rating out of bounds) or calls multiple structured tools mistakenly, the agent intercepts the error, writes the error back into the history, and prompts the model to correct its mistake.

```python
from langchain.agents.structured_output import ToolStrategy

# Handle all exceptions and prompt model with default errors
strategy_1 = ToolStrategy(schema=Schema, handle_errors=True)

# Custom text prompt on failure
strategy_2 = ToolStrategy(schema=Schema, handle_errors="Validation failed. Please correct your inputs.")

# Custom error handler function
def my_error_handler(error: Exception) -> str:
    return f"Custom error log: {error}"

strategy_3 = ToolStrategy(schema=Schema, handle_errors=my_error_handler)
```

---

## Key Rules

1. **Access Result Correctly** — The structured result is stored in `result["structured_response"]` (it returns an instance of your schema). Standard chat messages remain inside `result["messages"]`.
2. **JSON Schemas require a `title`** — Raw JSON schema structures must have a top-level `"title"` attribute to be parsed as tool/function definitions.
3. **Handle Errors in Tool Strategy** — `handle_errors` and `tool_message_content` are attributes of `ToolStrategy` (they do not apply to `ProviderStrategy`).
4. **Tool Compatibility** — If you bind tools (`tools=[...]`) AND use structured output simultaneously, ensure the underlying model supports parallel tool executions and structured generations concurrently.
