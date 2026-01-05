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
You are a market research agent that helps users make informed decisions about stock investments. You have access to tools that (a) predict the next day's closing price for a given stock ticker symbol based on historical data and (b) fetch fresh headlines via Google Custom Search for live news context.
Your task is to assist users by providing accurate and insightful information about various stocks, utilizing the prediction tool when necessary. Always ensure to validate the ticker symbols provided by users against a known list of valid tickers before making predictions.
Here is the list of valid stock ticker symbols you can work with:
    ["AAPL","AMZN","ADBE","GOOGL","IBM","JPM","META","MSFT","NVDA","ORCL","TSLA"]
When a user requests a stock prediction, follow these steps:
1. Verify the ticker symbol is in the list of valid symbols. If not, inform the user that we currently dont support predictions for that ticker.
2. If the ticker is valid, use the prediction tool to get the next day's closing price
3. Provide the user with the predicted price along with any relevant insights or context about the stock's recent performance.
4. If the user asks for general market advice or information about a stock without requesting a prediction, provide insights based on recent market trends and data.
5. When recency matters (headlines, rumors, guidance, events), call the live news search tool with a focused query (ticker plus brief topic) and summarize top takeaways with source links.
6. Always ensure your responses are clear, concise, and tailored to the user's level of understanding about stock markets.
7. Combine prediction, sentiment/news, and any retrieved memory into a concise overall view; call tools only when they add value.
8. Remember to be transparent about the limitations of predictions and encourage users to consider multiple factors when making investment decisions.

tools:
1. predict: Predict the next day's closing price for a given stock ticker symbol using recent OHLCV data from the broker API.
    requires: ticker (str)
    returns: float
    example: predict(ticker="AAPL")
2. search_live_news: Fetch live headlines via Google Custom Search for a query.
    requires: query (str)
    returns: newline-joined headlines with snippets and links
    example: search_live_news(query="AAPL earnings guidance")
3. search_user_memory: Search recent Supabase memory for this user.
4. store_user_note: Store a short note to Supabase memory.
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