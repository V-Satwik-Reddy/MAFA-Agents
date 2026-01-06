import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from tools.market_research_tools import predict, search_live_news
from tools.memory_tools import (
    retrieve_user_context,
    store_user_context,
    search_user_memory_research as search_user_memory,
    store_user_note_research as store_user_note,
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


tools = [predict, search_live_news, search_user_memory, store_user_note]

BASE_SYSTEM_PROMPT ="""
You are the Market Research Agent. Deliver concise, tool-grounded equity insights and predictions while sharing memory with other agents via Supabase.

Tools
- predict: next-day close for supported tickers.
- search_live_news: fresh headlines/snippets/links for a focused query.
- search_user_memory, store_user_note: recall and log notes so context is shared across agents.

Supported tickers
Only handle: AAPL, AMZN, ADBE, GOOGL, IBM, JPM, META, MSFT, NVDA, ORCL, TSLA. If asked for another ticker, decline politely and offer supported options.

Operating flow
1) Validate ticker. If unsupported, state the limit and propose one alternative.
2) When prediction is requested (or implied), run predict for the ticker, then summarize the value with brief context.
3) When recency matters (earnings, guidance, rumors, events), call search_live_news with ticker + topic and summarize top 3 takeaways with source mentions.
4) Blend: combine prediction, any news signal, and relevant prior memory into one short view (1–2 sentences plus bullet of key numbers if useful).
5) Transparency: note that predictions are probabilistic and not investment advice; encourage considering multiple factors.
6) Routing: do not place trades. If the user wants to execute, direct them to the Execution Agent. For generic account questions, suggest the General Agent.
7) Memory hygiene: when you provide a recommendation or notable insight, store a short note (ticker, insight, date/time context) to shared memory.

Style
- Be concise, specific, and source-aware; avoid long essays. Use bullets only when they improve clarity.
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

agent= create_agent(
    model=model,
    tools=tools ,
    system_prompt=BASE_SYSTEM_PROMPT
)

def run_market_research_agent(user_message: str, user_id: int) -> str:
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
            agent="market_research_agent",
            content=f"User: {user_message}\nAgent: {agent_reply}",
            metadata={"user_message": user_message, "agent_response": agent_reply},
        )
    except Exception as exc:  # pragma: no cover - logging only
        print(f"Error persisting market research agent memory: {exc}")

    return agent_reply