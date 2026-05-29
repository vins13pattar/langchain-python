# 🦜🔗 LangChain Python Examples

A structured collection of LangChain examples covering agents, models, messages, and tools — built with Python and designed for learning the LangChain / LangGraph ecosystem.

---

## 📁 Project Structure

```
langchain-python/
├── 1_agents/              # LangChain agent patterns
├── 2_models/              # LLM model initialization & parameters
├── 3_messages/            # Message types, history & streaming
├── 4_tools/               # Tool definitions, schemas & dynamic selection
├── 5_short_term_memory/   # Memory Saver, trimming & summarization
├── 6_event_streaming/     # Event Streaming v3 protocol & projections
├── 7_streaming/           # Graph Streaming modes (updates, values, messages, debug)
└── 8_structured_output/   # Validated schemas, strategies & retry mechanisms
```

---

## 📂 Modules

### 🤖 1_agents — Agent Patterns
| File | Description |
|------|-------------|
| `01_basic_agent.py` | Basic agent setup with tool use |
| `02_agent_with_memory.py` | Agent with conversation memory |
| `03_structured_output.py` | Structured/typed output from agents |
| `04_streaming.py` | Token streaming with agents |
| `05_middleware.py` | Middleware and error handling |
| `06_context_and_runtime.py` | Context injection at runtime |
| `07_full_agent_showcase.py` | End-to-end agent showcase |

### 🧠 2_models — LLM Models
| File | Description |
|------|-------------|
| `01_init_and_invoke.py` | Model initialization and invocation |
| `02_parameters.py` | Configuring model parameters |
| `03_structured_output.py` | Model structured outputs via Pydantic/TypedDict |
| `04_multimodality.py` | Multimodal inputs & image analysis |
| `05_tool_calling.py` | Tool binding and manual loops |

### 💬 3_messages — Messages & History
| File | Description |
|------|-------------|
| `01_message_types.py` | HumanMessage, AIMessage, SystemMessage |
| `02_conversation_history.py` | Managing conversation history |
| `03_message_content_and_streaming.py` | Content blocks and streaming |
| `04_tool_message_loop.py` | Tool call / ToolMessage cycles |
| `05_multimodal_messages.py` | Images and multimodal inputs |
| `06_full_messages_showcase.py` | Full messages showcase |

### 🔧 4_tools — Tools
| File | Description |
|------|-------------|
| `01_basic_tools.py` | Defining and using basic tools |
| `02_advanced_schemas.py` | Advanced Pydantic schemas for tools |
| `03_tool_runtime_context.py` | Injecting runtime context into tools |
| `04_tool_return_values.py` | Handling tool return values |
| `05_dynamic_tool_selection.py` | Dynamic tool selection |
| `06_full_tools_showcase.py` | Full tools showcase |

### 🧠 5_short_term_memory — Short-Term Memory
| File | Description |
|------|-------------|
| `01_checkpointer_basics.py` | Thread-based conversation state |
| `02_custom_state.py` | Persistent custom variables |
| `03_trim_and_delete_messages.py` | Trimming historical messages |
| `04_summarization_and_dynamic_prompt.py` | Compressing history & dynamic prompts |
| `05_full_memory_showcase.py` | Full memory personal assistant |

### 🌊 6_event_streaming — Event Streaming (v3 Projections)
| File | Description |
|------|-------------|
| `01_stream_events_basics.py` | Modern token projections |
| `02_tool_call_streaming.py` | Argument and execution tracking |
| `03_state_and_values_streaming.py` | Value snapshot tracking |
| `04_subagents_and_multiple_projections.py` | Child agent event routing |
| `05_full_streaming_showcase.py` | Production streaming chatbot |

### 📡 7_streaming — Graph Streaming (Step-Level API)
| File | Description |
|------|-------------|
| `01_updates_mode.py` | Step-level incremental state updates |
| `02_values_mode.py` | Step-level full state snapshots |
| `03_messages_mode.py` | Low-level model token streaming |
| `04_debug_mode.py` | Detailed internal tracing events |
| `05_full_streaming_showcase.py` | Interactive terminal streaming showcase |

### 🧱 8_structured_output — Structured Output
| File | Description |
|------|-------------|
| `01_auto_strategy.py` | Auto strategy selection (Pydantic, Dataclasses, TypedDict) |
| `02_provider_strategy.py` | Native ProviderStrategy with strict validation & JSON Schema |
| `03_tool_strategy_basics.py` | ToolStrategy for tool-calling models |
| `04_custom_tool_message.py` | Custom tool message representations in chat logs |
| `05_error_handling_and_retries.py` | Automatic Pydantic retries and custom handlers |
| `06_full_structured_output_showcase.py` | Support ticket classification and Order lookup showcase |

---

## ⚙️ Setup

### Prerequisites
- Python 3.11+
- An API key for your chosen LLM provider (e.g. OpenAI, Anthropic, Google)

### Installation

```bash
# Clone the repo
git clone https://github.com/vins13pattar/langchain-python.git
cd langchain-python

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

A `.env.example` file is provided with all supported keys. Copy it and fill in your values:

```bash
cp .env.example .env
```

Then edit `.env` with your actual API keys:

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic (Claude)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google Gemini
GOOGLE_API_KEY=your_google_api_key_here

# LangSmith (optional — for tracing & observability)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=langchain-python
```

> **Note**: `.env` is gitignored and will never be committed. Only `.env.example` (with placeholder values) is tracked in version control.

---

## 🚀 Running Examples

```bash
# Run any example directly
python 1_agents/01_basic_agent.py
python 2_models/01_init_and_invoke.py
python 3_messages/01_message_types.py
python 4_tools/01_basic_tools.py
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| [LangChain](https://python.langchain.com/) | LLM application framework |
| [LangGraph](https://langchain-ai.github.io/langgraph/) | Stateful agent orchestration |
| [OpenAI / Anthropic](https://openai.com/) | LLM providers |
| [Pydantic](https://docs.pydantic.dev/) | Data validation & schemas |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Environment variable management |

---

## 📄 License

MIT License — feel free to use and adapt these examples for your own projects.

---

## 🙋 Author

**Vinod** — [github.com/vins13pattar](https://github.com/vins13pattar)
