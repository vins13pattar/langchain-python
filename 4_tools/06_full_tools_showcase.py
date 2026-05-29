"""
06_full_tools_showcase.py
=========================
A COMPLETE showcase combining ALL tool concepts.

Demonstrates:
  ✅ @tool decorator with type hints and docstrings
  ✅ Custom tool name and description
  ✅ Pydantic args_schema for rich validation
  ✅ ToolRuntime — state, context, store, tool_call_id
  ✅ Return string / dict / Command
  ✅ Error handling with wrap_tool_call middleware
  ✅ Role-based dynamic tool filtering with wrap_model_call
  ✅ Long-term memory with InMemoryStore
  ✅ Streaming tool calls
  ✅ Direct tool invocation

This is a simulated e-commerce assistant that supports:
  - Product search and lookup
  - Shopping cart management (state)
  - Personalised recommendations (store)
  - Order placement (Command + ToolMessage)
  - Role-based admin tools
  - Graceful error handling
"""

import os
import uuid
from dataclasses import dataclass
from typing import Callable, Optional, List
from dotenv import load_dotenv

from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import wrap_model_call, wrap_tool_call, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from pydantic import BaseModel, Field

load_dotenv()

print("=" * 60)
print("Full Tools Showcase — E-Commerce Assistant")
print("=" * 60)


# ════════════════════════════════════════════════════════════════════
# CONTEXT & STATE SCHEMA
# ════════════════════════════════════════════════════════════════════

@dataclass
class CustomerCtx:
    customer_id: str
    name:        str
    role:        str = "customer"    # customer | admin


class ShopState(AgentState):
    cart_items:  list  = []          # items in current session cart
    order_count: int   = 0           # orders placed this session


# ════════════════════════════════════════════════════════════════════
# PRODUCT CATALOGUE (simulated database)
# ════════════════════════════════════════════════════════════════════

PRODUCTS = {
    "P001": {"name": "Wireless Headphones",   "price": 79.99,  "stock": 15, "category": "electronics"},
    "P002": {"name": "Python Programming Book", "price": 39.99, "stock": 30, "category": "books"},
    "P003": {"name": "Mechanical Keyboard",    "price": 149.99, "stock": 8,  "category": "electronics"},
    "P004": {"name": "Yoga Mat",               "price": 29.99,  "stock": 50, "category": "fitness"},
    "P005": {"name": "Coffee Grinder",         "price": 59.99,  "stock": 12, "category": "kitchen"},
}


# ════════════════════════════════════════════════════════════════════
# TOOLS
# ════════════════════════════════════════════════════════════════════

class ProductSearchInput(BaseModel):
    """Input for product search."""
    query:    str            = Field(description="Product search keywords")
    category: Optional[str] = Field(default=None, description="Filter by category: electronics, books, fitness, kitchen")
    max_price: Optional[float] = Field(default=None, description="Maximum price in USD")


@tool(args_schema=ProductSearchInput)
def search_products(
    query: str,
    category: Optional[str] = None,
    max_price: Optional[float] = None,
) -> str:
    """Search the product catalogue by keyword, category, and price.

    Use this when the user wants to find, browse, or compare products.
    """
    results = []
    for pid, p in PRODUCTS.items():
        match = query.lower() in p["name"].lower()
        cat_ok = category is None or p["category"] == category
        price_ok = max_price is None or p["price"] <= max_price
        if match and cat_ok and price_ok and p["stock"] > 0:
            results.append(f"  [{pid}] {p['name']} — ${p['price']} ({p['stock']} in stock)")
    if not results:
        return f"No products found for '{query}'"
    return "Found products:\n" + "\n".join(results)


@tool
def get_product_details(product_id: str) -> dict:
    """Get detailed information about a specific product by its ID.

    Returns structured product data including price, stock, and category.

    Args:
        product_id: Product identifier (e.g. 'P001', 'P002')
    """
    product = PRODUCTS.get(product_id.upper())
    if not product:
        return {"error": f"Product '{product_id}' not found"}
    return {"id": product_id.upper(), **product}


@tool
def add_to_cart(product_id: str, quantity: int = 1, runtime: ToolRuntime[CustomerCtx] = None) -> Command:
    """Add a product to the shopping cart.

    Updates the cart in session state.

    Args:
        product_id: Product ID to add (e.g. 'P001')
        quantity:   Number of units to add (default: 1)
    """
    product = PRODUCTS.get(product_id.upper())
    if not product:
        return Command(update={"messages": [
            ToolMessage(content=f"Product '{product_id}' not found.", tool_call_id=runtime.tool_call_id)
        ]})
    if product["stock"] < quantity:
        return Command(update={"messages": [
            ToolMessage(
                content=f"Only {product['stock']} units of {product['name']} available.",
                tool_call_id=runtime.tool_call_id
            )
        ]})

    cart = runtime.state.get("cart_items", [])
    cart.append({"id": product_id.upper(), "name": product["name"], "price": product["price"], "qty": quantity})

    return Command(update={
        "cart_items": cart,
        "messages": [ToolMessage(
            content=f"✅ Added {quantity}x {product['name']} (${product['price']}) to cart.",
            tool_call_id=runtime.tool_call_id,
        )],
    })


@tool
def view_cart(runtime: ToolRuntime[CustomerCtx]) -> str:
    """Show the current shopping cart contents and total.

    No input needed — reads cart from session state.
    """
    cart = runtime.state.get("cart_items", [])
    if not cart:
        return "🛒 Your cart is empty."
    lines = [f"  • {item['qty']}x {item['name']} @ ${item['price']} each" for item in cart]
    total = sum(item["price"] * item["qty"] for item in cart)
    return "🛒 Your cart:\n" + "\n".join(lines) + f"\n  Total: ${total:.2f}"


@tool
def place_order(runtime: ToolRuntime[CustomerCtx]) -> Command:
    """Place an order for all items in the cart.

    Clears the cart and saves order to long-term store.
    """
    cart = runtime.state.get("cart_items", [])
    if not cart:
        return Command(update={"messages": [
            ToolMessage(content="Your cart is empty. Add items before placing an order.", tool_call_id=runtime.tool_call_id)
        ]})

    total     = sum(i["price"] * i["qty"] for i in cart)
    order_id  = f"ORD-{abs(hash(str(cart)))%100000:05d}"
    ctx       = runtime.context

    # Save to long-term store
    existing  = runtime.store.get(("orders",), ctx.customer_id)
    orders    = existing.value if existing else []
    orders.append({"order_id": order_id, "items": cart, "total": total})
    runtime.store.put(("orders",), ctx.customer_id, orders)

    return Command(update={
        "cart_items":  [],           # clear the cart
        "order_count": runtime.state.get("order_count", 0) + 1,
        "messages": [ToolMessage(
            content=(
                f"🎉 Order placed! ID: {order_id}\n"
                f"   Items: {len(cart)} product(s)\n"
                f"   Total: ${total:.2f}\n"
                f"   Estimated delivery: 3-5 business days"
            ),
            tool_call_id=runtime.tool_call_id,
        )],
    })


@tool
def get_order_history(runtime: ToolRuntime[CustomerCtx]) -> str:
    """Retrieve the customer's full order history.

    No input needed — reads from long-term store using customer ID.
    """
    ctx      = runtime.context
    existing = runtime.store.get(("orders",), ctx.customer_id)
    if not existing or not existing.value:
        return "No previous orders found."
    orders = existing.value
    lines  = [f"  [{o['order_id']}] {len(o['items'])} item(s) — ${o['total']:.2f}" for o in orders]
    return f"Order history for {ctx.name}:\n" + "\n".join(lines)


# Admin-only tools
@tool
def admin_list_all_orders(runtime: ToolRuntime[CustomerCtx]) -> str:
    """[Admin only] List all orders in the system.

    No input needed.
    """
    return "📊 Admin view: 1,234 orders totalling $98,765.00 (simulated)"


@tool
def admin_update_stock(product_id: str, new_stock: int) -> str:
    """[Admin only] Update the stock level for a product.

    Args:
        product_id: Product ID to update
        new_stock:  New stock quantity
    """
    product = PRODUCTS.get(product_id.upper())
    if not product:
        return f"Product '{product_id}' not found"
    old_stock = product["stock"]
    product["stock"] = new_stock
    return f"✅ Stock updated: {product['name']} {old_stock} → {new_stock} units"


# ════════════════════════════════════════════════════════════════════
# MIDDLEWARE
# ════════════════════════════════════════════════════════════════════

CUSTOMER_TOOLS = {t.name for t in [search_products, get_product_details, add_to_cart, view_cart, place_order, get_order_history]}
ADMIN_TOOLS    = CUSTOMER_TOOLS | {admin_list_all_orders.name, admin_update_stock.name}


@wrap_model_call
def filter_by_role(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Filter tools based on customer role."""
    ctx  = request.runtime.context if request.runtime else None
    role = getattr(ctx, "role", "customer")
    allowed = ADMIN_TOOLS if role == "admin" else CUSTOMER_TOOLS
    filtered = [t for t in request.tools if t.name in allowed]
    return handler(request.override(tools=filtered))


@wrap_tool_call
def handle_errors(
    request: ToolCallRequest,
    handler: Callable,
) -> ToolMessage:
    """Gracefully handle any tool errors."""
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: {e}. Please adjust your request and try again.",
            tool_call_id=request.tool_call["id"],
        )


# ════════════════════════════════════════════════════════════════════
# CREATE AGENT
# ════════════════════════════════════════════════════════════════════

shop_store = InMemoryStore()

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[
        search_products, get_product_details,
        add_to_cart, view_cart, place_order, get_order_history,
        admin_list_all_orders, admin_update_stock,
    ],
    middleware=[filter_by_role, handle_errors],
    context_schema=CustomerCtx,
    state_schema=ShopState,
    checkpointer=MemorySaver(),
    store=shop_store,
    system_prompt=(
        "You are a friendly e-commerce shopping assistant for TechShop. "
        "Help customers find products, manage their cart, and place orders. "
        "Always confirm before placing an order. Be concise."
    ),
)


# ════════════════════════════════════════════════════════════════════
# RUN THE SHOWCASE
# ════════════════════════════════════════════════════════════════════

customer = CustomerCtx(customer_id="CUST-001", name="Vinod", role="customer")
config   = {"configurable": {"thread_id": str(uuid.uuid4())}}


def shop(message: str) -> str:
    result = agent.invoke(
        {"messages": [HumanMessage(message)]},
        config=config,
        context=customer,
    )
    return result["messages"][-1].content


print("\n── Shopping Session ──────────────────────────────────────")

print(f"\n🧑 Search for headphones:")
print(f"🤖 {shop('Show me wireless headphones under $100.')}")

print(f"\n🧑 Add to cart:")
print(f"🤖 {shop('Add 1 unit of the wireless headphones to my cart.')}")

print(f"\n🧑 Check cart:")
print(f"🤖 {shop('What is in my cart?')}")

print(f"\n🧑 Place order:")
print(f"🤖 {shop('Go ahead and place the order.')}")

print(f"\n🧑 Order history:")
print(f"🤖 {shop('Show me my order history.')}")

# Admin view
print("\n── Admin Session ─────────────────────────────────────────")
admin = CustomerCtx(customer_id="ADMIN-001", name="SuperAdmin", role="admin")
admin_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

result = agent.invoke(
    {"messages": [HumanMessage("List all orders in the system.")]},
    config=admin_config,
    context=admin,
)
print(f"\n👤 Admin: List all orders")
print(f"🤖 {result['messages'][-1].content}")
