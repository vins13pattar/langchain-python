"""
03_mcp_tools_resources_prompts.py
====================================
Demonstrates the three core MCP primitives: Tools, Resources, and Prompts.

Concepts covered:
  - Tools: structured content, multimodal content, load_mcp_tools
  - Resources: get_resources(), Blob objects, text/binary content
  - Prompts: get_prompt() with and without arguments, load_mcp_prompt
  - Stateful sessions for resources and prompts
  - Integrating MCP prompts into agent workflows

Note: Rich server examples require the rich server running:
  python 13_mcp/servers/rich_server.py  (port 8001)
  Math server examples use stdio (no separate process needed).
"""

import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.resources import load_mcp_resources
from langchain_mcp_adapters.prompts import load_mcp_prompt
from langchain.agents import create_agent
from langchain.messages import ToolMessage

load_dotenv()

MATH_SERVER = str(Path(__file__).parent / "servers" / "math_server.py")
RICH_URL     = "http://localhost:8001/mcp"

print("=" * 60)
print("MCP Tools, Resources & Prompts")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# PART 1: TOOLS
# ════════════════════════════════════════════════════════════════════

async def demo_basic_tools():
    """Load and use MCP tools — the most common MCP primitive."""
    print("\n── 1a. Basic Tool Loading (get_tools) ───────────────────────")

    client = MultiServerMCPClient({
        "math": {
            "transport": "stdio",
            "command":   "python",
            "args":      [MATH_SERVER],
        }
    })

    tools = await client.get_tools()
    print(f"  {len(tools)} tools: {[t.name for t in tools]}")

    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=tools,
        system_prompt="You are a math assistant.",
    )
    r = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Calculate 55 + 45, then multiply by 2."}]
    })
    print(f"  Result: {r['messages'][-1].content}")


async def demo_structured_content():
    """
    Structured content: MCP tools can return JSON alongside human-readable text.
    The adapter wraps it as MCPToolArtifact on the ToolMessage.artifact field.
    """
    print("\n── 1b. Structured Content (MCPToolArtifact) ─────────────────")

    client = MultiServerMCPClient({
        "rich": {"transport": "http", "url": RICH_URL}
    })

    try:
        tools = await client.get_tools()
        agent = create_agent(
            model="openai:gpt-4o-mini",
            tools=tools,
            system_prompt="You are a financial assistant.",
        )

        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": "What is the current AAPL stock price?"}]
        })

        # Extract structured content from tool messages
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage) and msg.artifact:
                structured = msg.artifact.get("structured_content", {})
                print(f"  Structured content: {structured}")
                print(f"  Text response:      {msg.content[:80]}")

        print(f"  Agent response: {result['messages'][-1].content[:120]}")

    except Exception as e:
        print(f"  ⚠️  Rich server not running — skipping ({type(e).__name__})")


async def demo_load_mcp_tools_session():
    """Use load_mcp_tools() with an explicit stateful session."""
    print("\n── 1c. load_mcp_tools() with Stateful Session ───────────────")

    client = MultiServerMCPClient({
        "math": {
            "transport": "stdio",
            "command":   "python",
            "args":      [MATH_SERVER],
        }
    })

    async with client.session("math") as session:
        tools = await load_mcp_tools(session)
        print(f"  Tools in session: {[t.name for t in tools]}")

        agent = create_agent(
            model="openai:gpt-4o-mini",
            tools=tools,
            system_prompt="You are a math assistant.",
        )
        r = await agent.ainvoke({
            "messages": [{"role": "user", "content": "What is 7 × 9?"}]
        })
        print(f"  Session result: {r['messages'][-1].content}")


# ════════════════════════════════════════════════════════════════════
# PART 2: RESOURCES
# ════════════════════════════════════════════════════════════════════

async def demo_resources():
    """
    Resources expose server-side data (files, DB records, API responses)
    as Blob objects with .as_string() and .mimetype.
    """
    print("\n── 2a. Loading Resources (get_resources) ────────────────────")

    client = MultiServerMCPClient({
        "rich": {"transport": "http", "url": RICH_URL}
    })

    try:
        # Load all resources from the server
        blobs = await client.get_resources("rich")
        print(f"  {len(blobs)} resources found:")
        for blob in blobs:
            uri  = blob.metadata.get("uri", "unknown")
            mime = blob.mimetype or "unknown"
            print(f"    URI: {uri}, MIME: {mime}")
            content = blob.as_string()
            print(f"    Content preview: {content[:60].strip()}")

    except Exception as e:
        print(f"  ⚠️  Rich server not running — skipping ({type(e).__name__})")


async def demo_resources_by_uri():
    """Load specific resources by URI."""
    print("\n── 2b. Resources by URI ─────────────────────────────────────")

    client = MultiServerMCPClient({
        "rich": {"transport": "http", "url": RICH_URL}
    })

    try:
        # Load only specific resources
        blobs = await client.get_resources(
            "rich",
            uris=["file:///data/config.json"]
        )
        for blob in blobs:
            print(f"  URI:  {blob.metadata.get('uri')}")
            print(f"  MIME: {blob.mimetype}")
            data = json.loads(blob.as_string())
            print(f"  Data: {data}")

    except Exception as e:
        print(f"  ⚠️  Rich server not running — skipping ({type(e).__name__})")


async def demo_resources_with_session():
    """Use load_mcp_resources() with an explicit session."""
    print("\n── 2c. Resources with load_mcp_resources() ──────────────────")

    client = MultiServerMCPClient({
        "rich": {"transport": "http", "url": RICH_URL}
    })

    try:
        async with client.session("rich") as session:
            blobs = await load_mcp_resources(session)
            print(f"  Session resources: {len(blobs)} found")
            for blob in blobs:
                print(f"    {blob.metadata.get('uri')}: {blob.as_string()[:40].strip()}...")

    except Exception as e:
        print(f"  ⚠️  Rich server not running — skipping ({type(e).__name__})")


# ════════════════════════════════════════════════════════════════════
# PART 3: PROMPTS
# ════════════════════════════════════════════════════════════════════

async def demo_prompts():
    """
    Prompts are server-side reusable templates returned as LangChain messages.
    Use client.get_prompt() to load them into a chat workflow.
    """
    print("\n── 3a. Loading Prompts (get_prompt) ─────────────────────────")

    client = MultiServerMCPClient({
        "rich": {"transport": "http", "url": RICH_URL}
    })

    try:
        # Load prompt without arguments
        messages = await client.get_prompt("rich", "summarize",
                                           arguments={"text": "LangChain is a framework for building LLM applications."})
        print(f"  Prompt messages ({len(messages)}):")
        for msg in messages:
            print(f"    [{msg.type}]: {msg.content[:80]}")

    except Exception as e:
        print(f"  ⚠️  Rich server not running — skipping ({type(e).__name__})")


async def demo_prompts_with_args():
    """Load prompts with arguments — returns messages ready for a chat model."""
    print("\n── 3b. Prompts with Arguments ───────────────────────────────")

    client = MultiServerMCPClient({
        "rich": {"transport": "http", "url": RICH_URL}
    })

    try:
        messages = await client.get_prompt(
            "rich",
            "code_review",
            arguments={
                "code":     "def add(a, b):\n    return a + b",
                "language": "python",
                "focus":    "security",
            }
        )
        print(f"  Prompt messages ({len(messages)}):")
        for msg in messages:
            print(f"    [{msg.type}]: {msg.content[:100]}")

        # These messages can be passed directly to create_agent
        from langchain.chat_models import init_chat_model
        llm = init_chat_model("openai:gpt-4o-mini")
        response = await llm.ainvoke(messages)
        print(f"\n  LLM review: {response.content[:150]}")

    except Exception as e:
        print(f"  ⚠️  Rich server not running — skipping ({type(e).__name__})")


async def demo_prompts_with_session():
    """Use load_mcp_prompt() with an explicit session."""
    print("\n── 3c. Prompts with load_mcp_prompt() ───────────────────────")

    client = MultiServerMCPClient({
        "rich": {"transport": "http", "url": RICH_URL}
    })

    try:
        async with client.session("rich") as session:
            messages = await load_mcp_prompt(
                session,
                "summarize",
                arguments={"text": "Machine learning is a subset of AI."}
            )
            print(f"  Session prompt messages: {len(messages)}")
            for msg in messages:
                print(f"    [{msg.type}]: {msg.content[:80]}")

    except Exception as e:
        print(f"  ⚠️  Rich server not running — skipping ({type(e).__name__})")


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

async def main():
    # Tools (always works — math server via stdio)
    await demo_basic_tools()
    await demo_load_mcp_tools_session()

    # Tools — structured content (needs rich server)
    await demo_structured_content()

    # Resources (needs rich server)
    await demo_resources()
    await demo_resources_by_uri()
    await demo_resources_with_session()

    # Prompts (needs rich server)
    await demo_prompts()
    await demo_prompts_with_args()
    await demo_prompts_with_session()

    print("\n" + "═" * 60)
    print("MCP Primitives Summary:")
    print("  Tools:     get_tools() / load_mcp_tools()     → LangChain tools")
    print("  Resources: get_resources() / load_mcp_resources() → Blob objects")
    print("  Prompts:   get_prompt() / load_mcp_prompt()   → LangChain messages")
    print("  All three can use stateful sessions via client.session()")
    print("═" * 60)
    print("\n✅ MCP tools, resources & prompts demo complete.")


if __name__ == "__main__":
    asyncio.run(main())
