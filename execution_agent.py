import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from tools.execute_trade_tools import execute_trade_order
from tools.profile_tools import get_current_stock_price, get_user_balance, get_user_holdings
from tools.memory_tools import (
    retrieve_user_context,
    store_user_context,
    search_user_memory_execution as search_user_memory,
    store_user_note_execution as store_user_note,
)
from vectordbsupabase import SupabaseVectorDB

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise RuntimeError("Missing GOOGLE_API_KEY environment variable")
os.environ["GOOGLE_API_KEY"] = google_api_key

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Shared vector DB for persistent chat memory
vector_db = SupabaseVectorDB()


tools = [
    get_user_balance,
    get_current_stock_price,
    execute_trade_order,
    get_user_holdings,
    search_user_memory,
    store_user_note,
]

BASE_SYSTEM_PROMPT = """You are a financial trading assistant. You can help users check their balance, get current stock prices, and execute buy/sell orders.
Your Role is to execute users trade orders based on their requests.
1) Understand the user's request carefully.
2) If the query is generic or unrelated to trading, responde appropriately without using any tools and generally.
3) If the user query is related to trading and executing orders, understand their query and help them with
    a) checking their balance
    b) getting current stock prices
    c) selling or buying stocks
4) If query is about their balance and current stock prices, use the respective tools to fetch the information and provide it to the user.
5) If the user wants to buy or sell stocks, gather all necessary information such as stock symbol, quantity or price, etc.
6) Use the execute_trade_order tool to place the order on behalf of the user and after successful completion of it give confirmation to user.
7) Before placing a order confirm user's trade by summarizing the order details back to them.
8) Always ensure that you confirm the user's trade before executing any trade orders.
9) When providing responses, be clear and concise and give more detailed explanations when necessary.
10) When doing a buy always ensure that the user has sufficient balance to complete the trade. If not, inform them about insufficient balance, when doing a sell ensure that the user has sufficient shares to sell. If not, inform them about insufficient shares.
11) When doing a buy/sell if quantity of share is not an integer or if the number of shares we get for the user mentioned price is not integer tell the user that we can only buy/sell whole shares and ask them to provide a valid quantity or price and provide the closest integer shares they can buy.

1. get_user_balance: Fetches the user's current balance.
    takes no arguments and returns a float.
2. get_current_stock_price: Gets the current stock price for a given symbol.
    takes a stock symbol (str) as input and returns the current price (float).
3. execute_trade_order: Executes a trade order (buy/sell) through the broker API.
    takes order details (dict) and action type (str: "buy" or "sell") as input and returns the order ID (str). order details should include symbol and quantity.
4. get_user_holdings: Retrieves the user's current stock holdings.
    takes no arguments and returns a dictionary of stock symbols and their quantities.
5. search_user_memory: Search recent Supabase memory for this user.
6. store_user_note: Store a short note to Supabase memory.
    
"""


def build_system_message(user_id: int, user_message: str) -> str | None:
    """Fetch recent context from Supabase memory for this user/agent."""
    try:
        query_emb = vector_db.embed_text(user_message)
        rows = retrieve_user_context(
            user_id=str(user_id),
            agent="execution_agent",
            query_embedding=query_emb,
            top_k=5,
            min_score=0.3,
        )
    except Exception:
        rows = []
    if not rows:
        try:
            rows = vector_db.latest_records(user_id=str(user_id), agent="execution_agent", limit=5)
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

agent= create_agent(
    model=model,
    tools=tools,
    system_prompt=BASE_SYSTEM_PROMPT
)

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
