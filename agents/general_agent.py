import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from tools.execute_trade_tools import execute_trade_order
from tools.profile_tools import (
    get_current_stock_price,
    get_user_balance,
    get_user_holdings,
)
from tools.memory_tools import (
    retrieve_user_context,
    store_user_context,
    search_user_memory as search_user_memory,
    store_user_note as store_user_note,
)
from vectordbsupabase import SupabaseVectorDB

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise RuntimeError("Missing GOOGLE_API_KEY environment variable")
os.environ["GOOGLE_API_KEY"] = google_api_key

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

model_with_search=model.bind_tools([{"google_search": {}}])

# Shared vector DB for persistent chat memory
vector_db = SupabaseVectorDB()

@tool
def google_search(query: str) -> str:
    """Perform a Google search and return summarized and processed results."""
    try:
        results = model_with_search.invoke(query)
        return results
    except Exception as exc:
        return f"Google search unavailable: {exc}"

# Expose all account-related tools.
tools = [
    get_user_balance,
    get_user_holdings,
    get_current_stock_price,
    google_search,
    execute_trade_order,
    search_user_memory,
    store_user_note,
]

BASE_SYSTEM_PROMPT = """
You are the General Financial Agent. Act as a fast, trustworthy concierge for account info, quick lookups, and light guidance. Keep one shared memory with other agents via Supabase.

Tools
- get_user_balance, get_user_holdings, get_current_stock_price: use for account-aware answers.
- google_search: use for freshness (news, announcements, current figures).
- search_user_memory, store_user_note: recall/store brief context to keep conversations coherent across agents.

Operating rules
1) Core scope: balances, holdings, price checks, account status, simple company overviews, headlines, and general finance Q&A.
2) Routing: never place or simulate trades. If the user wants to buy/sell/modify/cancel an order, redirect to the Execution Agent. If they want forecasts/deep research, redirect to the Market Research Agent.
3) Recency: when the question depends on latest info, call google_search, then summarize top takeaways with source mentions.
4) Memory: when context matters, search memory for recent intents or preferences; state only what you find. After useful interactions, store a short note (topic, ticker, preference) to shared memory.
5) Safety & tone: avoid personalized investment advice; mark stale/approximate data; ask one clarifying question if needed.
6) Style: concise first answer with key figures, then one clear next step or option.
"""


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
            rows = vector_db.latest_records(
                user_id=str(user_id), agent=None, limit=5
            )
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


agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=BASE_SYSTEM_PROMPT,
)


def run_general_agent(user_message: str, user_id: int) -> str:
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
                part.get("text", "")
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
            agent="general_agent",
            content=f"User: {user_message}\nAgent: {agent_reply}",
            metadata={"user_message": user_message, "agent_response": agent_reply},
        )
    except Exception as exc:  # pragma: no cover - logging only
        print(f"Error persisting general agent memory: {exc}")

    return agent_reply
