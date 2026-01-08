import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from tools.profile_tools import get_user_holdings, get_current_stock_price
from tools.memory_tools import (
    retrieve_user_context,
    store_user_context,
    search_user_memory as search_user_memory,
    store_user_note as store_user_note,
)
from vectordbsupabase import SupabaseVectorDB

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
vector_db = SupabaseVectorDB()

tools=[
    get_current_stock_price,
    get_user_holdings,
    search_user_memory,
    store_user_note,
]

BASE_SYSTEM_PROMPT=""" 

"""

agent = create_agent(
    tools=tools,
    model=model,
    system_prompt=BASE_SYSTEM_PROMPT,
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

def run_portfolio_manager_agent(user_message: str, user_id: int) -> str:
    """Run the Portfolio Manager Agent with user message and context."""
    print(f"\n{'='*60}")
    print(f"USER {user_id}: {user_message}")
    print(f"{'='*60}\n")
    
    system_msg={"role": "system", "content": build_system_message(user_id, user_message)}
    response=agent.invoke(
        {
            "messages": [
                system_msg,
                {"role": "user", "content": user_message},
            ],
        },
        config={"user_id": user_id},
    )
    final_message = response["messages"][-1]
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
    return agent_reply