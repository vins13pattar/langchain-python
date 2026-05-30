# 🦜🔗 LangChain Python Examples

A structured collection of LangChain examples covering agents, models, messages, tools, short-term memory, event streaming, graph streaming, structured outputs, middleware, guardrails, runtime context, context engineering, Model Context Protocol (MCP), and human-in-the-loop — built with Python and designed for learning the LangChain / LangGraph ecosystem.

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
├── 8_structured_output/   # Validated schemas, strategies & retry mechanisms
├── 9_middleware/          # Built-in middleware, HITL, custom hooks & guardrails
├── 10_guardrails/         # PIIMiddleware, deterministic & model-based guardrails
├── 11_runtime/            # Runtime context, ToolRuntime, execution_info & server_info
├── 12_context_engineering/ # Model/Tool/Life-cycle context × State/Store/Runtime
├── 13_mcp/                # Model Context Protocol — servers, tools, resources, interceptors
├── 14_human_in_the_loop/  # HITL middleware — approve, edit, reject, respond decisions
├── 15_multi_agent/        # Subagents, Handoffs, Skills, Router patterns
├── 16_retrieval/          # RAG — knowledge base, 2-step, agentic, hybrid
├── 17_longterm_memory/    # Store — semantic/episodic/procedural memory across sessions
└── 18_frontend/           # Generative UI — streaming, tool cards, HITL, time travel
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

### 🛡️ 9_middleware — Middleware
| File | Description |
|------|-------------|
| `01_built_in_middleware.py` | `SummarizationMiddleware`, `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware`, `ToolRetryMiddleware` |
| `02_human_in_the_loop.py` | HITL approve / edit / reject workflows with per-tool policies |
| `03_custom_middleware.py` | `BaseMiddleware` hooks: logging, timing, cost estimation, tool-specific middleware |
| `04_pii_detection_and_guardrails.py` | `PIIDetectionMiddleware`, content guardrails, input validation, output sanitization |
| `05_agent_loop_middleware.py` | Loop observer, rate limiter, early exit, hook firing order |
| `06_full_middleware_showcase.py` | Customer support triage agent with 6 stacked middleware layers |

### 🛑 10_guardrails — Guardrails
| File | Description |
|------|-------------|
| `01_pii_middleware.py` | `PIIMiddleware` with `redact` / `mask` / `hash` / `block` strategies and custom regex detector |
| `02_deterministic_guardrails.py` | `before_agent` class & decorator hooks, keyword filter, rate limiter, input length validation |
| `03_model_based_guardrails.py` | `after_agent` class & decorator hooks, LLM-as-judge safety, topic relevance, quality gate |
| `04_hitl_as_guardrail.py` | `HumanInTheLoopMiddleware` for financial/database/email ops, full approve/edit/reject lifecycle |
| `05_full_guardrails_showcase.py` | Financial advisory agent with 7-layer guardrail stack across 5 real-world scenarios |

### ⚡ 11_runtime — Runtime Context
| File | Description |
|------|-------------|
| `01_context_schema.py` | `context_schema` dataclass, injecting `context=` at invoke time, `@dynamic_prompt` from context |
| `02_tool_runtime.py` | `ToolRuntime[Context]` in tools, `runtime.context`, `runtime.store`, `runtime.writer` |
| `03_runtime_in_middleware.py` | `@before_model` / `@after_model` with `Runtime[Context]`, `execution_info`, `server_info`, RBAC |
| `04_execution_and_server_info.py` | `thread_id`, `run_id`, `attempt`, retry detection, audit trail, production auth gate |
| `05_full_runtime_showcase.py` | Multi-tenant CRM agent with context injection, RBAC, dynamic prompts, store, and audit logging |

### 🧠 12_context_engineering — Context Engineering
| File | Description |
|------|-------------|
| `01_model_context_system_prompt.py` | `@dynamic_prompt` from State, Store, and Runtime Context; combined multi-source prompt |
| `02_model_context_messages.py` | `@wrap_model_call` transient injection: file context, writing style, compliance rules |
| `03_model_context_tools_and_model.py` | Dynamic tool filtering (RBAC, feature flags, auth) and dynamic model switching (cost tier) |
| `04_model_context_response_format.py` | Dynamic Pydantic schema selection by conversation stage, verbosity pref, and role |
| `05_tool_context_reads_writes.py` | Tool reads from state/store/context; writes via `Command` and `store.put()` |
| `06_lifecycle_context.py` | `SummarizationMiddleware`, persistent `before_model` state updates, audit logging |
| `07_full_context_engineering_showcase.py` | Smart Legal Research Agent — all 3 context types × all 3 data sources × 3 scenarios |

### 🔌 13_mcp — Model Context Protocol
| File | Description |
|------|-------------|
| `servers/math_server.py` | FastMCP stdio server — add, subtract, multiply, divide |
| `servers/weather_server.py` | FastMCP HTTP server — get_weather, get_forecast, get_air_quality |
| `servers/rich_server.py` | FastMCP HTTP server with structured content, resources, prompts, progress |
| `01_mcp_basics.py` | `MultiServerMCPClient`, `get_tools()`, stateless vs stateful sessions |
| `02_mcp_transports.py` | stdio vs HTTP transports, custom headers, multi-server config |
| `03_mcp_tools_resources_prompts.py` | Tools, structured content, Resources (Blob), Prompts (messages) |
| `04_mcp_interceptors.py` | Logging, runtime context, store, state auth, `request.override()`, retry, composition |
| `05_mcp_callbacks.py` | `on_progress`, `on_logging_message`, `on_elicitation` (accept/decline/cancel) |
| `06_full_mcp_showcase.py` | Smart Data Assistant — interceptors + callbacks + middleware + multi-turn memory |

### 🧑‍⚖️ 14_human_in_the_loop — Human-in-the-Loop (HITL)
| File | Description |
|------|-------------|
| `01_hitl_basics.py` | `HumanInTheLoopMiddleware`, `interrupt_on`, `version="v2"`, approve, reject, auto-approve |
| `02_decision_types.py` | All 4 types: approve, edit (args + tool swap), reject, respond |
| `03_multiple_decisions.py` | Multiple simultaneous interrupts, mixed decisions, sequential rounds |
| `04_hitl_streaming.py` | `stream()` with `stream_mode=["updates","messages"]`, interrupt detection in stream |
| `05_full_hitl_showcase.py` | Secure Financial Operations Agent — risk-tiered policies, all 4 decisions, streaming |

### 🤖 15_multi_agent — Multi-Agent Systems
| File | Description |
|------|-------------|
| `01_subagents.py` | Tool-per-agent wrapping, `ToolRuntime`, `Command`+`InjectedToolCallId`, parallel calls |
| `02_subagents_dispatch.py` | Single dispatch `task` tool, enum constraint, tool-based discovery, async background jobs |
| `03_handoffs.py` | `current_step` state, `Command` transitions, `@dynamic_prompt`, `@wrap_model_call` |
| `04_skills.py` | On-demand skill loading, stateful reuse across turns, Store cache |
| `05_router.py` | LLM structured output routing, keyword routing, fan-out+merge, async parallel, nested |
| `06_full_multi_agent_showcase.py` | Enterprise Assistant — Router dispatches to Skills / Subagents / Handoffs |

### 🔍 16_retrieval — Retrieval-Augmented Generation (RAG)
| File | Description |
|------|-------------|
| `01_knowledge_base.py` | Document loaders, text splitters, embeddings, vector stores, retrievers |
| `02_two_step_rag.py` | 2-Step RAG — fixed retrieve-then-generate, LCEL chains, multi-query |
| `03_agentic_rag.py` | Agentic RAG — agent with retrieval tools, fetch_url, multi-source |
| `04_hybrid_rag.py` | Hybrid RAG — query enhancement, retrieval validation, answer quality check |
| `05_full_retrieval_showcase.py` | Smart Q&A Assistant — all three RAG architectures with routing |

### 🧠 17_longterm_memory — Long-Term Memory
| File | Description |
|------|-------------|
| `01_store_basics.py` | `InMemoryStore`, `IndexConfig`, `put/get/delete/search`, namespace patterns, `StoreValue` |
| `02_read_memory_in_tools.py` | `ToolRuntime[Context]`, `context_schema`, `runtime.store.get()`, multi-namespace reads, semantic search |
| `03_write_memory_from_tools.py` | `TypedDict` schemas, `save_user_info`, episodic events, procedural rules, delete |
| `04_memory_types.py` | Semantic, episodic, procedural memory types, extraction LLM, cross-memory synthesis |
| `05_full_longterm_memory_showcase.py` | Personal AI — learn in session 1, recall in session 2, multi-user isolation |

### 🌐 18_frontend — Generative UI & Agent Frontends
| File | Description |
|------|-------------|
| `01_agent_server.py` | Python backend — `create_agent` + FastAPI SSE streaming + LangGraph Dev server config |
| `frontend/01_basic_chat.html` | Token streaming chat with live tool call cards (demo + real server mode) |
| `frontend/02_hitl_approval.html` | HITL — agent pauses, human approves/rejects/edits before tool execution |
| `frontend/03_full_chat_app.html` | Full showcase — streaming, tools, HITL, structured output, markdown, time travel |

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
python 9_middleware/01_built_in_middleware.py
python 10_guardrails/01_pii_middleware.py
python 11_runtime/01_context_schema.py
python 12_context_engineering/01_model_context_system_prompt.py
python 13_mcp/01_mcp_basics.py
python 14_human_in_the_loop/01_hitl_basics.py
python 15_multi_agent/01_subagents.py
python 16_retrieval/01_knowledge_base.py
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
