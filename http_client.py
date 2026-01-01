import requests
import contextvars
from typing import Any, Dict, Optional

# Shared session so base headers are reused; Authorization is injected per request from contextvar
_session = requests.Session()
_session.headers.update({"Content-Type": "application/json", "User-Agent": "MCP-FinancialAgent/1.0"})

# Request-scoped token to avoid cross-request leakage in concurrent handlers
_request_token: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_request_token", default=None)


def set_request_token(token: Optional[str]) -> None:
    """Persist the user token for the current request context only."""
    _request_token.set(token)


def get_auth_headers(token: Optional[str] = None) -> Dict[str, str]:
    """Return merged headers, ensuring Authorization is applied when present."""
    use_token = token or _request_token.get()
    headers = dict(_session.headers)
    if use_token:
        headers["Authorization"] = use_token if use_token.lower().startswith("bearer ") else f"Bearer {use_token}"
    return headers


def _merge_headers(headers: Optional[Dict[str, str]]) -> Dict[str, str]:
    merged = get_auth_headers()
    if headers:
        merged.update(headers)
    return merged


def get(url: str, *, headers: Optional[Dict[str, str]] = None, **kwargs: Any):
    return _session.get(url, headers=_merge_headers(headers), **kwargs)


def post(url: str, *, headers: Optional[Dict[str, str]] = None, **kwargs: Any):
    return _session.post(url, headers=_merge_headers(headers), **kwargs)
