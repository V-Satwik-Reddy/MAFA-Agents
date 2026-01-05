import os
from typing import Iterable, List

import pandas as pd
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from requests import RequestException

from http_client import get
from lstm.predict_next_day import predict_next_day_price

API_BASE = "http://localhost:8080"
REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

load_dotenv()
CUSTOM_SEARCH_API_KEY = os.getenv("CUSTOM_SEARCH_API_KEY") or os.getenv("CUSTOM_SEARCH_API_KEY")
CUSTOM_SEARCH_CX = os.getenv("CUSTOM_SEARCH_CX")


def _fetch_ohlcv(ticker: str) -> List[dict]:
    url = f"{API_BASE}/stockdailyprices?symbol={ticker}"
    response = get(url, timeout=15)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", payload)
    if not isinstance(data, Iterable):
        raise ValueError("Unexpected OHLCV payload format")
    return list(data)


def _to_dataframe(records: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in OHLCV data: {', '.join(missing)}")
    df = df[REQUIRED_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if len(df) < 21:
        raise ValueError("Need at least 21 days of OHLCV data to predict")
    return df


def _fetch_live_news(query: str, num: int = 5) -> List[dict]:
    if not CUSTOM_SEARCH_API_KEY:
        raise RuntimeError("Missing GOOGLE_CUSTOM_SEARCH_API_KEY or CUSTOM_SEARCH_API_KEY")
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": CUSTOM_SEARCH_API_KEY,
        "cx": CUSTOM_SEARCH_CX,
        "q": query,
        "num": num,
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    payload = response.json()
    items = payload.get("items") or []
    if not isinstance(items, list):
        return []
    return items[:num]


@tool
def predict(ticker: str) -> float:
    """Predict the next day's closing price using recent OHLCV data from the broker API."""
    try:
        records = _fetch_ohlcv(ticker)
        prices_df = _to_dataframe(records)
        prediction_df = predict_next_day_price(ticker, prices_df)
        return float(prediction_df)
    except (RequestException, ValueError, TypeError) as exc:
        print(f"Error predicting next-day price: {exc}")
        return 0.0


@tool
def search_live_news(query: str) -> str:
    """Search live news via Google Custom Search and return concise headlines with links."""
    try:
        items = _fetch_live_news(query)
    except (RequestException, ValueError, RuntimeError) as exc:
        return f"News search unavailable: {exc}"
    if not items:
        return "No recent news found."
    lines = []
    for item in items:
        title = item.get("title", "Untitled")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        lines.append(f"- {title} | {snippet} | {link}")
    return "\n".join(lines)
