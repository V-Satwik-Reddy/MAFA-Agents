from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from vectordbsupabase import SupabaseVectorDB

vector_db = SupabaseVectorDB()


def store_user_context(
    user_id: str,
    agent: str,
    content: str,
    embedding: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Store user and agent context in the shared Supabase vector table."""
    try:
        vector = embedding or vector_db.embed_text(content)
        return vector_db.upsert_record(
            user_id=user_id,
            agent=agent,
            content=content,
            embedding=vector,
            metadata=metadata,
        )
    except Exception as exc:
        print(f"Error storing context: {exc}")
        return ""


def retrieve_user_context(
    user_id: str,
    agent: str,
    query_embedding: List[float],
    top_k: int = 5,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """Retrieve similar context entries for a user and agent using vector search."""
    try:
        return vector_db.similarity_search(
            user_id=user_id,
            agent=agent,
            query_embedding=query_embedding,
            match_count=top_k,
            match_threshold=min_score,
        )
    except Exception as exc:
        print(f"Error retrieving context: {exc}")
        return []


def supabase_vector_schema_sql() -> str:
    """Return SQL required to provision the Supabase vector table and RPC."""
    try:
        return vector_db.schema_sql()
    except Exception as exc:
        print(f"Error generating schema SQL: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Tool-wrapped helpers (agent-specific) for recalling/storing short notes
# ---------------------------------------------------------------------------


def _render_rows(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No recent memory found."
    return "\n".join(
        f"- {row.get('metadata', {}).get('user_message', '')} -> {row.get('metadata', {}).get('agent_response', '')}"
        for row in rows
    )


@tool
def search_user_memory_general(query: str, user_id: str) -> str:
    """Search recent Supabase memory for this user (general agent)."""
    try:
        emb = vector_db.embed_text(query)
        rows = retrieve_user_context(
            user_id=str(user_id),
            agent="general_agent",
            query_embedding=emb,
            top_k=5,
            min_score=0.25,
        )
    except Exception as exc:
        return f"Memory search unavailable: {exc}"
    return _render_rows(rows)


@tool
def store_user_note_general(note: str, user_id: str) -> str:
    """Store a short note to Supabase memory (general agent)."""
    try:
        store_user_context(
            user_id=str(user_id),
            agent="general_agent",
            content=note,
            metadata={"user_message": note, "agent_response": "stored_note"},
        )
        return "Saved to memory."
    except Exception as exc:
        return f"Could not save memory: {exc}"


@tool
def search_user_memory_execution(query: str, user_id: str) -> str:
    """Search recent Supabase memory for this user (execution agent)."""
    try:
        emb = vector_db.embed_text(query)
        rows = retrieve_user_context(
            user_id=str(user_id),
            agent="execution_agent",
            query_embedding=emb,
            top_k=5,
            min_score=0.25,
        )
    except Exception as exc:
        return f"Memory search unavailable: {exc}"
    return _render_rows(rows)


@tool
def store_user_note_execution(note: str, user_id: str) -> str:
    """Store a short note to Supabase memory (execution agent)."""
    try:
        store_user_context(
            user_id=str(user_id),
            agent="execution_agent",
            content=note,
            metadata={"user_message": note, "agent_response": "stored_note"},
        )
        return "Saved to memory."
    except Exception as exc:
        return f"Could not save memory: {exc}"


@tool
def search_user_memory_research(query: str, user_id: str) -> str:
    """Search recent Supabase memory for this user (market research agent)."""
    try:
        emb = vector_db.embed_text(query)
        rows = retrieve_user_context(
            user_id=str(user_id),
            agent="market_research_agent",
            query_embedding=emb,
            top_k=5,
            min_score=0.25,
        )
    except Exception as exc:
        return f"Memory search unavailable: {exc}"
    return _render_rows(rows)


@tool
def store_user_note_research(note: str, user_id: str) -> str:
    """Store a short note to Supabase memory (market research agent)."""
    try:
        store_user_context(
            user_id=str(user_id),
            agent="market_research_agent",
            content=note,
            metadata={"user_message": note, "agent_response": "stored_note"},
        )
        return "Saved to memory."
    except Exception as exc:
        return f"Could not save memory: {exc}"
