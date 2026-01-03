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
    search_user_memory_general as search_user_memory,
    store_user_note_general as store_user_note,
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
You are the General Financial Agent. Be a concise, trustworthy concierge for everyday account help, quick stock lookups, and light financial guidance. You have access to:
- get_user_balance, get_user_holdings, get_current_stock_price for account-aware responses.
- search_user_memory, store_user_note for recalling and saving brief context in Supabase.
 - google_search  to ground answers with fresh web context when recency or external confirmation is needed.

Operating rules:
1) Core scope: balances, holdings, price checks, statements on account status, company overviews, headlines, and general finance/stocks Q&A. Use tools when they improve accuracy or freshness.
2) Trade intent: never place or simulate trades here. If the user wants to buy/sell/place/modify/cancel an order, politely decline and tell them to the use the Execution Agent option from agents list.
3) Deep market analysis: if the user wants ticker-specific forecasts or research, inform them to use the Market Research Agent from agents list.
4) News & recency: when the user asks for latest news/rumors/announcements or freshness matters, proactively call google_search, then summarize the key takeaways.
5) Memory: When helpful, search existing memory for recent intents or preferences. Keep continuity while avoiding hallucination; if unsure, ask a brief clarifying question.
6) Safety & clarity: state when data may be stale or approximate. Avoid personalized investment advice; frame outputs as informational.
7) Style: clear, succinct answers first; include key figures and next steps. Offer a single follow-up option when relevant.

If you ever see a request outside your scope (e.g., portfolio rebalancing execution, option trades, wire transfers), explain the limit and recommend the appropriate agent or channel.
"""


def build_system_message(user_id: int, user_message: str) -> str | None:
    """Fetch recent context from Supabase memory for this user/agent."""
    try:
        query_emb = vector_db.embed_text(user_message)
        rows = retrieve_user_context(
            user_id=str(user_id),
            agent="general_agent",
            query_embedding=query_emb,
            top_k=5,
            min_score=0.3,
        )
    except Exception:
        rows = []
    if not rows:
        try:
            rows = vector_db.latest_records(
                user_id=str(user_id), agent="general_agent", limit=5
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
