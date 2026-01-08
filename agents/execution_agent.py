import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from tools.execute_trade_tools import execute_trade_order
from tools.profile_tools import get_current_stock_price, get_user_balance, get_user_holdings
from tools.memory_tools import (
    retrieve_user_context,
    store_user_context,
    search_user_memory as search_user_memory,
    store_user_note as store_user_note,
)
from vectordbsupabase import SupabaseVectorDB

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
vector_db = SupabaseVectorDB()

tools = [
    get_user_balance,
    get_current_stock_price,
    execute_trade_order,
    get_user_holdings,
    search_user_memory,
    store_user_note,
]

BASE_SYSTEM_PROMPT = """
You are the Execution Agent. Your sole job is to safely place equity buy/sell orders. Be crisp, tool-aware, and confirm before executing.

Scope and routing
- Handle trading intents only. If the request is informational or research-oriented, reply briefly and suggest the General or Market Research Agent instead of using trade tools.

Tool use (always prefer tools for facts)
- get_user_balance, get_user_holdings: verify funds/shares before trading.
- get_current_stock_price: fetch the latest price to ground the order summary.
- execute_trade_order: place orders only after explicit user confirmation.
- search_user_memory, store_user_note: use shared Supabase memory to recall preferences and log each decision.

Execution flow
1) Clarify the order: side (buy/sell), ticker, whole-share quantity, desired price/type (market vs limit). Ask concise questions if missing.
2) Pull recent memory for preferences (e.g., risk limits, ticker notes). Mention any relevant prior note.
3) Pre-trade check: show balance/holdings and current price; reject non-integer quantities.
4) Safety gates: block if insufficient cash/shares; block fractional or negative quantities; state the issue plainly.
5) Confirmation: present a one-shot summary bullet (side, ticker, qty, est cost/proceeds). Ask for "Yes/Confirm" before execution.
6) On confirm, call execute_trade_order with the gathered details; then report status and any order id.
7) After response, store a short memory note with ticker, side, qty, price context, and outcome to keep centralized context.

Style
- Be concise and directive. Use short paragraphs or bullets. Avoid over-explaining.
"""

agent= create_agent(
    model=model,
    tools=tools,
    system_prompt=BASE_SYSTEM_PROMPT
)

def build_system_message(user_id: int, user_message: str) -> str | None:
    """Fetch recent context from Supabase memory for this user/agent."""
    try:
        query_emb = vector_db.embed_text(user_message)
        rows = retrieve_user_context(
            user_id=str(user_id),
            agent="shared_context",
            query_embedding=query_emb,
            top_k=5,
            min_score=0.3,
        )
    except Exception:
        rows = []
    if not rows:
        try:
            rows = vector_db.latest_records(user_id=str(user_id), agent=None, limit=5)
        except Exception:
            rows = []
    if not rows:
        return None
    history = "".join(
        f"User: {row.get('metadata', {}).get('user_message')}\n"
        f"Agent: {row.get('metadata', {}).get('agent_response')}\n"
        for row in rows
    )
    return "Recent conversation history:\n" + history

def run_execute_agent(user_message: str, user_id: int) -> str:
    print(f"\n{'='*60}")
    print(f"USER {user_id}: {user_message}")
    print(f"{'='*60}\n")


    system_msg = {"role": "system", "content": build_system_message(user_id, user_message)}
    result = agent.invoke(
        {
            "messages": [system_msg, {"role": "user", "content": user_message}],
        },
        config={"user_id": user_id},
    )

    final_message = result["messages"][-1]
    def normalize(content):
        if isinstance(content, list):
            return "".join(
                part["text"]
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return content
    print("AGENT:", final_message.content)
    
    agent_reply = normalize(final_message.content)

    # Persist conversation turn to Supabase
    try:
        store_user_context(
            user_id=str(user_id),
            agent="execution_agent",
            content=f"User: {user_message}\nAgent: {agent_reply}",
            metadata={"user_message": user_message, "agent_response": agent_reply},
        )
    except Exception as exc:  # pragma: no cover - logging only
        print(f"Error persisting execution agent memory: {exc}")

    return agent_reply
