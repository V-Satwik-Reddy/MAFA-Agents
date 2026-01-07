import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from tools.profile_tools import get_user_holdings
from tools.memory_tools import (
    retrieve_user_context,
    store_user_context,
    search_user_memory_portfolio as search_user_memory,
    store_user_note_portfolio as store_user_note,
)
from vectordbsupabase import SupabaseVectorDB

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

mode=ChatGoogleGenerativeAI(model="gemini-2.5-flash")