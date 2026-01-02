from typing import Any, Dict

from langchain_core.tools import tool
from requests import RequestException

from http_client import post

BROKER_BASE = "http://localhost:8080"


def _post_json(url: str, body: Dict[str, Any], timeout: int = 30) -> Any:
    response = post(url, json=body, timeout=timeout)
    response.raise_for_status()
    return response.json()


@tool
def execute_trade_order(order_data: dict, action_type: str) -> str:
    """Execute a trade order (buy/sell) through the broker API."""
    url = f"{BROKER_BASE}/execute/{action_type}"
    try:
        payload = _post_json(url, order_data)
        order_id = payload.get("id") if isinstance(payload, dict) else ""
        return str(order_id) if order_id else ""
    except RequestException as exc:
        print(f"Error executing trade order: {exc}")
        return ""
