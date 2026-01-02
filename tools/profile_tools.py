from typing import Any

from langchain_core.tools import tool
from requests import RequestException

from http_client import get

API_BASE = "http://localhost:8080"


def _fetch_json(url: str, timeout: int = 10) -> Any:
    response = get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


@tool
def get_user_balance() -> float:
    """
    function: Fetch the user's current balance from the profile service.
    Returns the balance as a float. If there's an error, returns 0.0.
    """
    url = f"{API_BASE}/profile/balance"
    try:
        payload = _fetch_json(url)
        balance = payload.get("data", 0.0) if isinstance(payload, dict) else 0.0
        return float(balance)
    except (RequestException, ValueError, TypeError) as exc:
        print(f"Error fetching user balance: {exc}")
        return 0.0


@tool
def get_current_stock_price(symbol: str) -> float:
    """
    Function Get the current stock price for a given symbol.
    Arguments:
        symbol (str): The stock ticker symbol.
    Returns:
        float: The current stock price. If there's an error, returns 0.0.
    """
    url = f"{API_BASE}/stockprice?symbol={symbol}"
    try:
        payload = _fetch_json(url)
        price = payload.get("price") if isinstance(payload, dict) else payload
        return float(price) if price is not None else 0.0
    except (RequestException, ValueError, TypeError) as exc:
        print(f"Error fetching price: {exc}")
        return 0.0

@tool
def get_user_holdings() -> dict:
    """Fetch the user's current stock holdings from the profile service.
    Returns a dictionary mapping stock symbols to quantities. If there's an error, returns an empty dictionary.
    """
    url = f"{API_BASE}/profile/holdings"
    try:
        payload = _fetch_json(url)
        holdings = payload.get("data", {}) if isinstance(payload, dict) else {}
        return holdings if isinstance(holdings, dict) else {}
    except (RequestException, ValueError, TypeError) as exc:
        print(f"Error fetching user holdings: {exc}")
        return {}