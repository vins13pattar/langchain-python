from langchain.agents import create_agent
from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import os
load_dotenv()

gemma_model = init_chat_model(
    model="gemma-4-e2b",
    model_provider="openai",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

CHAT_THREAD_1 = {"configurable": {"thread_id": "thread_test_1"}}

messages = [
    HumanMessage(content="Hi, I need help setting up a dashboard for our user activity metrics. Can you check what tables we have in ClickHouse?"),
    AIMessage(
        content="I'll list the tables in Clickhouse to see what we're working with.",
        tool_calls=[{"name": "list_tables", "args": {}, "id": "call_1"}]
    ),
    ToolMessage(content='["users", "user_events", "purchases", "sessions"]', tool_call_id="call_1"),
    AIMessage(content="We have four tables: `users`, `user_events`, `purchases`, and `sessions`. Which metrics would you like to focus on first?"),
    HumanMessage(content="Let's focus on user signups per day for the last month. Can you describe the `users` table schema?"),
    AIMessage(
        content="Let's query the structure of the `users` table.",
        tool_calls=[{"name": "run_select_query", "args": {"query": "DESCRIBE TABLE users"}, "id": "call_2"}]
    ),
    ToolMessage(
        content='[{"name": "user_id", "type": "UUID"}, {"name": "email", "type": "String"}, {"name": "created_at", "type": "DateTime"}, {"name": "country", "type": "String"}]',
        tool_call_id="call_2"
    ),
    AIMessage(content="The `users` table has columns `user_id` (UUID), `email` (String), `created_at` (DateTime), and `country` (String). I will write a query to get daily signup counts."),
    HumanMessage(content="Yes, please run that query for June 2026."),
    AIMessage(
        content="Running the query to count signups per day in June 2026.",
        tool_calls=[{"name": "run_select_query", "args": {"query": "SELECT toDate(created_at) as date, count() as signups FROM users WHERE created_at >= '2026-06-01' AND created_at < '2026-07-01' GROUP BY date ORDER BY date"}, "id": "call_3"}]
    ),
    ToolMessage(
        content='[{"date": "2026-06-01", "signups": 120}, {"date": "2026-06-02", "signups": 145}, {"date": "2026-06-03", "signups": 130}]',
        tool_call_id="call_3"
    ),
    AIMessage(content="Here are the daily signups for June 2026. On average, we had around 130 signups per day. Would you like to check user retention or purchase patterns next?"),
    HumanMessage(content="Let's look at user purchases. What's the schema of the `purchases` table?"),
    AIMessage(
        content="Describing the `purchases` table structure.",
        tool_calls=[{"name": "run_select_query", "args": {"query": "DESCRIBE TABLE purchases"}, "id": "call_4"}]
    ),
    ToolMessage(
        content='[{"name": "purchase_id", "type": "UUID"}, {"name": "user_id", "type": "UUID"}, {"name": "amount", "type": "Decimal(10, 2)"}, {"name": "purchased_at", "type": "DateTime"}]',
        tool_call_id="call_4"
    ),
    AIMessage(content="The `purchases` table has `purchase_id`, `user_id`, `amount`, and `purchased_at`. We can join it with the `users` table on `user_id`."),
    HumanMessage(content="Awesome. Can you show me the total revenue per day for June 2026?"),
    AIMessage(
        content="Querying total revenue per day in June 2026.",
        tool_calls=[{"name": "run_select_query", "args": {"query": "SELECT toDate(purchased_at) as date, sum(amount) as revenue FROM purchases WHERE purchased_at >= '2026-06-01' AND purchased_at < '2026-07-01' GROUP BY date ORDER BY date"}, "id": "call_5"}]
    ),
    ToolMessage(
        content='[{"date": "2026-06-01", "revenue": 1200.50}, {"date": "2026-06-02", "revenue": 1500.00}]',
        tool_call_id="call_5"
    ),
    AIMessage(content="The total revenue per day for June 2026 ranges from $1,000 to $2,500. Total revenue for the month was approximately $45,000. Would you like to segment this by user country?"),
    HumanMessage(content="Yes, let's see which countries generated the most revenue in June."),
    AIMessage(
        content="Querying revenue by country.",
        tool_calls=[{"name": "run_select_query", "args": {"query": "SELECT u.country, sum(p.amount) as total_revenue FROM purchases p JOIN users u ON p.user_id = u.user_id WHERE p.purchased_at >= '2026-06-01' AND p.purchased_at < '2026-07-01' GROUP BY u.country ORDER BY total_revenue DESC"}, "id": "call_6"}]
    ),
    ToolMessage(
        content='[{"country": "US", "total_revenue": 25000.00}, {"country": "IN", "total_revenue": 10000.00}, {"country": "GB", "total_revenue": 5000.00}]',
        tool_call_id="call_6"
    ),
    AIMessage(content="The US leads with $25,000 in revenue, followed by India (IN) with $10,000, and the UK (GB) with $5,000. Do you want to see the average purchase amount for each country?"),
    HumanMessage(content="No, let's switch to user events. How many events were logged in the `user_events` table in June?"),
    AIMessage(
        content="Counting logged user events in June 2026.",
        tool_calls=[{"name": "run_select_query", "args": {"query": "SELECT count() as event_count FROM user_events WHERE event_time >= '2026-06-01' AND event_time < '2026-07-01'"}, "id": "call_7"}]
    ),
    ToolMessage(
        content='[{"event_count": 1254300}]',
        tool_call_id="call_7"
    ),
    AIMessage(content="There were 1,254,300 events logged in June 2026. This is a high-volume table. Should we inspect the most common event types?"),
    HumanMessage(content="Yes, what are the top 5 event types?"),
    AIMessage(
        content="Querying the top 5 event types in June.",
        tool_calls=[{"name": "run_select_query", "args": {"query": "SELECT event_name, count() as count FROM user_events GROUP BY event_name ORDER BY count DESC LIMIT 5"}, "id": "call_8"}]
    ),
    ToolMessage(
        content='[{"event_name": "page_view", "count": 800000}, {"event_name": "button_click", "count": 250000}, {"event_name": "search", "count": 120000}, {"event_name": "add_to_cart", "count": 60000}, {"event_name": "purchase_complete", "count": 24300}]',
        tool_call_id="call_8"
    ),
    AIMessage(content="The top 5 events are `page_view`, `button_click`, `search`, `add_to_cart`, and `purchase_complete`. What analysis should we do next?"),
    HumanMessage(content="Let's calculate the conversion rate from `add_to_cart` to `purchase_complete`."),
    AIMessage(
        content="Calculating the conversion rate.",
        tool_calls=[{"name": "run_select_query", "args": {"query": "SELECT (SELECT count() FROM user_events WHERE event_name = 'purchase_complete') / (SELECT count() FROM user_events WHERE event_name = 'add_to_cart') * 100 as conversion_rate"}, "id": "call_9"}]
    ),
    ToolMessage(
        content='[{"conversion_rate": 40.5}]',
        tool_call_id="call_9"
    ),
    AIMessage(content="The conversion rate from `add_to_cart` to `purchase_complete` is 40.5%. This is a strong conversion rate. Would you like to check if this varies by country?"),
    HumanMessage(content="Let's check if the conversion rate is different for US users versus IN users."),
    AIMessage(
        content="Querying conversion rates by country (US vs IN).",
        tool_calls=[{"name": "run_select_query", "args": {"query": "SELECT u.country, countIf(e.event_name = 'purchase_complete') / countIf(e.event_name = 'add_to_cart') * 100 as conversion_rate FROM user_events e JOIN users u ON e.user_id = u.user_id WHERE u.country IN ('US', 'IN') GROUP BY u.country"}, "id": "call_10"}]
    ),
    ToolMessage(
        content='[{"country": "US", "conversion_rate": 45.2}, {"country": "IN", "conversion_rate": 35.8}]',
        tool_call_id="call_10"
    ),
    AIMessage(content="US users convert at 45.2% while Indian users convert at 35.8%. We might want to optimize the checkout process or local pricing for India. Do you want to see the average order value (AOV) for both countries?"),
    HumanMessage(content="Yes, let's see the Average Order Value (AOV) for US and IN in June."),
    AIMessage(
        content="Querying average order value by country.",
        tool_calls=[{"name": "run_select_query", "args": {"query": "SELECT u.country, avg(p.amount) as aov FROM purchases p JOIN users u ON p.user_id = u.user_id WHERE u.country IN ('US', 'IN') AND p.purchased_at >= '2026-06-01' AND p.purchased_at < '2026-07-01' GROUP BY u.country"}, "id": "call_11"}]
    ),
    ToolMessage(
        content='[{"country": "US", "aov": 75.50}, {"country": "IN", "aov": 42.00}]',
        tool_call_id="call_11"
    ),
    AIMessage(content="The Average Order Value (AOV) is $75.50 in the US and $42.00 in India. This explains the difference in revenue contribution. Would you like to look at the daily trends of AOV?"),
    HumanMessage(content="Let's check the weekly cohort retention for users who signed up in May 2026."),
    AIMessage(
        content="Querying cohort retention for May 2026 signups.",
        tool_calls=[{"name": "run_select_query", "args": {"query": "SELECT toStartOfWeek(u.created_at) as cohort, toStartOfWeek(e.event_time) as activity_week, count(distinct u.user_id) as active_users FROM users u LEFT JOIN user_events e ON u.user_id = e.user_id WHERE u.created_at >= '2026-05-01' AND u.created_at < '2026-06-01' GROUP BY cohort, activity_week"}, "id": "call_12"}]
    ),
    ToolMessage(
        content='[{"cohort": "2026-05-03", "activity_week": "2026-05-03", "active_users": 1000}, {"cohort": "2026-05-03", "activity_week": "2026-05-10", "active_users": 400}, {"cohort": "2026-05-03", "activity_week": "2026-05-17", "active_users": 300}]',
        tool_call_id="call_12"
    ),
    AIMessage(content="For the May 3rd cohort of 1,000 users: Week 1 retention was 40% (400 users), and Week 2 was 30% (300 users). This indicates a typical drop-off. Would you like to see if they made purchases in Week 1?"),
    HumanMessage(content="Yes, how many users from that cohort purchased in Week 1 (May 10 to May 17)?"),
    AIMessage(
        content="Querying purchasers in Week 1 from the May 3rd cohort.",
        tool_calls=[{"name": "run_select_query", "args": {"query": "SELECT count(distinct u.user_id) as purchasers FROM users u JOIN purchases p ON u.user_id = p.user_id WHERE u.created_at >= '2026-05-03' AND u.created_at < '2026-05-10' AND p.purchased_at >= '2026-05-10' AND p.purchased_at < '2026-05-17'"}, "id": "call_13"}]
    ),
    ToolMessage(
        content='[{"purchasers": 120}]',
        tool_call_id="call_13"
    ),
    AIMessage(content="From the 1,000 users who signed up in the week of May 3rd, 120 of them (12%) made a purchase in Week 1. This is a very decent activation rate."),
    HumanMessage(content="Can we find out what was the most purchased item in that cohort's Week 1 purchases?"),
    AIMessage(
        content="Querying most purchased items for that cohort.",
        tool_calls=[{"name": "run_select_query", "args": {"query": "SELECT p.item_name, count() as sales FROM users u JOIN purchases p ON u.user_id = p.user_id WHERE u.created_at >= '2026-05-03' AND u.created_at < '2026-05-10' AND p.purchased_at >= '2026-05-10' AND p.purchased_at < '2026-05-17' GROUP BY p.item_name ORDER BY sales DESC LIMIT 3"}, "id": "call_14"}]
    ),
    ToolMessage(
        content='[{"item_name": "Premium Subscription", "sales": 80}, {"item_name": "Advanced Pack", "sales": 30}, {"item_name": "Basic Ebook", "sales": 10}]',
        tool_call_id="call_14"
    ),
    AIMessage(content="The top item was the 'Premium Subscription' with 80 sales, followed by the 'Advanced Pack' (30) and the 'Basic Ebook' (10). It seems the subscription is the primary driver."),
    HumanMessage(content="Excellent. Can you export these findings to a summary report file called cohort_summary.txt?"),
    AIMessage(
        content="Writing the cohort summary report to a text file.",
        tool_calls=[{"name": "write_to_file", "args": {"filename": "cohort_summary.txt", "content": "Cohort May 3rd: 1000 users\\nWeek 1 Active: 400 (40%)\\nWeek 1 Purchasers: 120 (12%)\\nTop Item: Premium Subscription (80 sales)\\n"}, "id": "call_15"}]
    ),
    ToolMessage(content="File cohort_summary.txt successfully written.", tool_call_id="call_15"),
    AIMessage(content="I have successfully written the cohort summary report to `cohort_summary.txt`. Let me know if you want me to calculate anything else or analyze other cohorts!")
]

agent = create_agent(
    model=gemma_model,
    tools=[],
    system_prompt="You are a helpful assistant",
    checkpointer=InMemorySaver(),
    middleware=[
        ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(
                    trigger=50,
                    keep=2,
                ),
            ],
        ),
    ],
)

agent.update_state(
    config=CHAT_THREAD_1,
    values={"messages": messages}
)


class ModelInputCallbackHandler(BaseCallbackHandler):
    """Callback handler to print messages actually received by the LLM (model)."""
    def on_chat_model_start(self, serialized, messages, **kwargs):
        print("\n=== Messages Received by LLM during Invoke ===")
        for i, msg in enumerate(messages[0]):
            print(f"[{i}] {msg.type.upper()}: {msg.content}")
        print("=============================================\n")
        return

# Before invocation
state = agent.get_state(config=CHAT_THREAD_1)
print("Before State: ", len(state.values.get('messages')))

before_extract_tool_messages = [msg for msg in state.values.get("messages", []) if msg.type == "tool"]
print("Before: Number of tool messages: ", len(before_extract_tool_messages))

# Fix: Only pass the new human message. Passing `messages + [...]` causes LangGraph's 
# add_messages reducer to append all 60 messages again, duplicating the history.
result = agent.invoke(
    {"messages": [HumanMessage(content="hi")]},
    config={**CHAT_THREAD_1, "callbacks": [ModelInputCallbackHandler()]}
)

# After invocation (Note: ContextEditingMiddleware works transiently on model inputs, so state checkpointer is not mutated)
state = agent.get_state(config=CHAT_THREAD_1)
print("\nAfter State: ", len(state.values.get('messages')))

after_extract_tool_messages = [msg for msg in state.values.get("messages", []) if msg.type == "tool"]
print("After: Number of tool messages: ", len(after_extract_tool_messages))
