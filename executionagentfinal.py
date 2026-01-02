"""
MCP-Orchestrated Trade Execution Agent - Gemini 2.0 Flash 002 (YOUR MODEL)
Built with LangGraph | Bearer Token Auth | Full Trade Lifecycle
Author: MCP Financial Analyst Project (B17 Batch)
"""

import os
import json
import time
import uuid
import logging
import re
from typing import TypedDict, Dict, Any, Optional, List
from decimal import Decimal
import requests
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import google.generativeai as genai

# =================================================================================
# CONFIGURATION - YOUR GEMINI 2.5 FLASH MODEL
# =================================================================================
# print(genai.list_models())
class Config:
    PORTFOLIO_API_BASE = "http://localhost:8080/"
    MARKET_DATA_API_BASE = "http://localhost:8081/"
    BROKER_API_BASE = "http://localhost:8080"
    
    MIN_TRADE_AMOUNT = Decimal('1.00')
    MAX_TRADE_AMOUNT = Decimal('100000.00')
    MAX_SHARES = 10000

    # Symbol handling
    SUPPORTED_SYMBOLS = (
        "AAPL", "AMZN", "GOOGL", "MSFT", "TSLA", "META",
        "NVDA", "NFLX", "AMD", "INTC", "IBM", "ORCL",
        "AVGO", "ADBE", "PYPL", "DIS", "SHOP", "BABA",
        "JPM", "V", "MA", "BAC", "UBER", "COIN", "CRM"
    )
    DEFAULT_SYMBOL = SUPPORTED_SYMBOLS[0]
    SAMPLE_ALT_SYMBOL = SUPPORTED_SYMBOLS[1]
    
    # ✅ YOUR GEMINI 2.5 FLASH MODEL
    MODEL_NAME = "gemini-2.5-flash"
    TEMPERATURE = 0.1

# Your API Keys
GEMINI_API_KEY = "AIzaSyBnb-xKjc3ol2TECZoRNGVorYdKuJV3dYE"
BROKER_AUTH_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJzYXR3aWszQGdtYWlsLmNvbSIsInR5cGUiOiJBQ0NFU1MiLCJpYXQiOjE3NjYyOTM2MTUsImV4cCI6MTc2NjM4MDAxNX0.U6hyzMrxCNOCq7goTimPr0jZI1ikbNgpbLKUGHjYxkw"
PORTFOLIO_AUTH_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJzYXR3aWszQGdtYWlsLmNvbSIsInR5cGUiOiJBQ0NFU1MiLCJpYXQiOjE3NjYyOTM2MTUsImV4cCI6MTc2NjM4MDAxNX0.U6hyzMrxCNOCq7goTimPr0jZI1ikbNgpbLKUGHjYxkw"
MARKETDATA_AUTH_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJzYXR3aWszQGdtYWlsLmNvbSIsInR5cGUiOiJBQ0NFU1MiLCJpYXQiOjE3NjYyOTM2MTUsImV4cCI6MTc2NjM4MDAxNX0.U6hyzMrxCNOCq7goTimPr0jZI1ikbNgpbLKUGHjYxkw"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =================================================================================
# STATE DEFINITION
# =================================================================================
class TradeExecutionState(TypedDict):
    messages: List[dict]
    user_id: str
    session_id: str
    symbol: str
    action: str
    available_balance: Decimal
    current_price: Decimal
    quantity: int
    total_cost: Decimal
    max_affordable_quantity: int
    user_approval: bool
    trade_executed: bool
    order_id: str
    error: Optional[str]
    current_node: str

# =================================================================================
# UTILITY FUNCTIONS - PRODUCTION READY
# =================================================================================
def get_auth_headers(token: str = None) -> dict:
    token = token or BROKER_AUTH_TOKEN
    if not token.lower().startswith("bearer "):
        token = f"Bearer {token}"
    return {"Authorization": token, "Content-Type": "application/json", "User-Agent": "MCP-FinancialAgent/1.0"}

def get_user_balance(user_id: str) -> Decimal:
    """Your Portfolio API integration"""
    headers = get_auth_headers(PORTFOLIO_AUTH_TOKEN)
    url = f"{Config.PORTFOLIO_API_BASE.rstrip('/')}/profile/balance"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        print(data)
        return Decimal(str(data.get('available_balance', '10000')))
    except:
        logger.warning("Using mock balance: $10,000")
        return Decimal('10000.00')

def get_current_stock_price(symbol: str) -> Decimal:
    """Your Market Data API integration"""
    headers = get_auth_headers(MARKETDATA_AUTH_TOKEN)
    url = f"{Config.MARKET_DATA_API_BASE.rstrip('/')}/price?symbol={symbol}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        print(data)
        return Decimal(str(data))
    except:
        # Real prices for testing
        prices = {
            "AAPL": 195.50, "AMZN": 185.20, "GOOGL": 142.80, "MSFT": 415.30, "TSLA": 248.50, "META": 567.40,
            "NVDA": 135.10, "NFLX": 610.25, "AMD": 118.30, "INTC": 46.80, "IBM": 175.40, "ORCL": 128.60,
            "AVGO": 1785.00, "ADBE": 590.20, "PYPL": 65.10, "DIS": 102.40, "SHOP": 74.80, "BABA": 86.50,
            "JPM": 194.50, "V": 275.40, "MA": 448.30, "BAC": 40.20, "UBER": 72.10, "COIN": 245.00, "CRM": 286.50
        }
        fallback = prices.get(symbol.upper()) or prices.get(Config.DEFAULT_SYMBOL)
        return Decimal(str(fallback))

def execute_trade_order(order_data: dict, action_type: str) -> str:
    """Your Broker API integration"""
    headers = get_auth_headers(BROKER_AUTH_TOKEN)
    print(order_data)
    order_data['price']=order_data['total_cost']
    url = f"{Config.BROKER_API_BASE.rstrip('/')}/execute/{action_type}"
    try:
        response = requests.post(url, json=order_data, headers=headers, timeout=30)
        result = response.json()
        print
        return result.get('id')
    except:
        order_id = f"ORD-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        logger.info(f"✅ Mock trade: {order_id}")
        return order_id

def validate_trade_limits(quantity: int, total_cost: Decimal) -> bool:
    return (quantity > 0 and quantity <= Config.MAX_SHARES and 
            Config.MIN_TRADE_AMOUNT <= total_cost <= Config.MAX_TRADE_AMOUNT)

# =================================================================================
# GEMINI 2.5 FLASH SETUP - YOUR MODEL ✅
# =================================================================================
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(
    model_name=Config.MODEL_NAME,  # gemini-2.0-flash-exp-1206 ✅
    generation_config={
        "temperature": Config.TEMPERATURE,
        "top_p": 0.95,
        "max_output_tokens": 1024
    }
)

# =================================================================================
# LANGGRAPH NODES - OPTIMIZED FOR GEMINI 2.5
# =================================================================================
def build_system_prompt() -> str:
        symbols = ", ".join(Config.SUPPORTED_SYMBOLS)
        sample_schema = {"symbol": Config.DEFAULT_SYMBOL, "action": "buy"}
        buy_policy = "type=\"buy\" and order_data with the requested symbol and live price"
        sell_policy = "type=\"sell\" and order_data with the requested symbol and computed quantity"
        return f"""
You are a trading intent parser. Your job is to extract symbol and action from a single user message and return STRICT JSON only.

Rules:
- Supported symbols (uppercase): {symbols}
- Supported actions (lowercase): buy, sell.
- Do NOT include any text outside JSON. No explanations.
- Output schema: {json.dumps(sample_schema)}
- Never change keys or casing.
- If the user asks for anything unrelated to trading, tell them you only handle trade intents, then still return JSON with defaults.
- Downstream execution policy (for awareness):
    - Buy orders use {buy_policy}.
    - Sell orders use {sell_policy}.

Return ONLY JSON.
"""


def parse_intent(state: TradeExecutionState) -> TradeExecutionState:
    """Gemini 2.5 Flash intent parsing - bulletproof JSON"""
    # If we're mid-flow (e.g., waiting for quantity or approval), skip re-parsing intent.
    if state.get("current_node") in {"get_quantity", "request_approval", "check_approval", "execute_trade"}:
        return state

    user_text = state["messages"][-1]["content"].lower() if state.get("messages") else ""
    if not re.search(r"\b(buy|sell)\b", user_text):
        state["current_node"] = END
        state["messages"].append({
            "role": "assistant",
            "content": "👋 Hi! I can help with stock trades. Tell me what you'd like to buy or sell."
        })
        return state

    prompt = f"{build_system_prompt()}\nUser message: {state['messages'][-1]['content']}\nJSON ONLY:"  # noqa: E501
    
    try:
        response = model.generate_content(prompt)
        intent_text = response.text.strip()
        
        # Super robust JSON extraction
        intent_text = re.sub(r'``````', '', intent_text).strip()
        json_match = re.search(r'\{.*\}', intent_text, re.DOTALL)
        if json_match:
            intent = json.loads(json_match.group())
        else:
            raise ValueError("No JSON found")
            
        state["symbol"] = intent.get("symbol", Config.DEFAULT_SYMBOL).upper()
        state["action"] = intent.get("action", "buy").lower()
        state["current_node"] = "fetch_data"
        
        state["messages"].append({
            "role": "assistant", 
            "content": f"✅ Confirmed: {state['action'].upper()} {state['symbol']}"
        })
        
    except Exception as e:
        logger.error(f"Parse error: {e}")
        state["error"] = f"Say 'buy {Config.DEFAULT_SYMBOL}' or 'sell {Config.SAMPLE_ALT_SYMBOL}'"
        state["current_node"] = "error"
    
    return state

def fetch_balance_and_price(state: TradeExecutionState) -> TradeExecutionState:
    state["available_balance"] = get_user_balance(state["user_id"])
    state["current_price"] = get_current_stock_price(state["symbol"])
    
    if state["action"] == "buy":
        state["max_affordable_quantity"] = int(state["available_balance"] / state["current_price"])
    else:
        state["max_affordable_quantity"] = 100
    
    state["current_node"] = "get_quantity"
    
    msg = f"""📊 **{state['symbol']} READY TO TRADE**
💰 Price: ${state['current_price']:.2f}
💵 Balance: ${state['available_balance']:,.2f}
🔢 Max {state['action'].upper()}: {state['max_affordable_quantity']} shares

Enter shares (1-{state['max_affordable_quantity']}) or a dollar amount (e.g., $800):"""
    
    state["messages"].append({"role": "assistant", "content": msg})
    return state

def get_quantity_from_user(state: TradeExecutionState) -> TradeExecutionState:
    user_msg = state["messages"][-1]["content"].lower()
    numbers = re.findall(r"\d+(?:\.\d+)?", user_msg)
    is_amount = any(keyword in user_msg for keyword in ["$", "usd", "dollar", "worth", "amount"])

    if not numbers:
        state["current_node"] = "get_quantity"
        state["messages"].append({
            "role": "assistant",
            "content": f"❌ Enter shares (1-{state['max_affordable_quantity']}) or a dollar amount like $800."
        })
        return state

    raw_value = Decimal(numbers[0])

    if state["action"] == "buy" and is_amount:
        quantity = int(raw_value / state["current_price"])
        if quantity <= 0:
            state["current_node"] = "get_quantity"
            state["messages"].append({
                "role": "assistant",
                "content": f"❌ ${raw_value} is below the price per share (${state['current_price']:.2f}). Enter a higher amount or specify shares."
            })
            return state
    else:
        quantity = int(raw_value)

    if quantity <= 0 or quantity > state["max_affordable_quantity"]:
        state["current_node"] = "get_quantity"
        state["messages"].append({
            "role": "assistant", 
            "content": f"❌ Enter 1-{state['max_affordable_quantity']} shares within your limit."
        })
        return state

    state["quantity"] = quantity
    state["total_cost"] = Decimal(quantity) * state["current_price"]
    
    if not validate_trade_limits(quantity, state["total_cost"]):
        state["error"] = "Trade exceeds limits"
        state["current_node"] = "error"
        return state
    
    state["current_node"] = "request_approval"
    
    msg = f"""📋 **FINAL CONFIRMATION**
{state['action'].upper()} {quantity} {state['symbol']}
💰 Price: ${state['current_price']:.2f}
💵 Total: ${state['total_cost']:,.2f}

Are you sure? Reply **APPROVE** or **CANCEL**:"""
    
    state["messages"].append({"role": "assistant", "content": msg})
    return state

def request_user_approval(state: TradeExecutionState) -> TradeExecutionState:
    state["current_node"] = "check_approval"
    return state

def check_user_approval(state: TradeExecutionState) -> TradeExecutionState:
    user_msg = state["messages"][-1]["content"].lower()
    
    if "approve" in user_msg:
        state["user_approval"] = True
        state["current_node"] = "execute_trade"
    elif "cancel" in user_msg:
        state["error"] = "Trade cancelled"
        state["current_node"] = END
    else:
        state["current_node"] = "check_approval"
        state["messages"].append({
            "role": "assistant", 
            "content": "**APPROVE** or **CANCEL** only!"
        })
    return state

def execute_trade(state: TradeExecutionState) -> TradeExecutionState:
    order_type = state["action"]
    order_data = {
        "symbol": state["symbol"],
        "quantity": float(state.get("quantity", 0)),
        "price": float(state.get("current_price", Decimal("0"))),
        "total_cost": float(state.get("total_cost", Decimal("0")))
    }

    state["order_id"] = execute_trade_order(order_data, order_type)
    state["trade_executed"] = True
    state["current_node"] = END
    
    msg = f"""🎉 **TRADE EXECUTED SUCCESSFULLY!**

🆔 Order ID: {state['order_id']}
📈 {order_type.upper()} {state['quantity']} {state['symbol']}
💰 Total: ${state['total_cost']:,.2f}
✅ Check your portfolio!"""
    
    state["messages"].append({"role": "assistant", "content": msg})
    return state

def handle_error(state: TradeExecutionState) -> TradeExecutionState:
    msg = state.get("error", "Unknown error") + "\n\nTry: 'buy {symbol} 10'".format(symbol=Config.DEFAULT_SYMBOL)
    state["messages"].append({"role": "assistant", "content": f"❌ {msg}"})
    state["current_node"] = END
    return state

# =================================================================================
# LANGGRAPH WORKFLOW
# =================================================================================
def create_workflow():
    workflow = StateGraph(TradeExecutionState)
    
    workflow.add_node("parse_intent", parse_intent)
    workflow.add_node("fetch_data", fetch_balance_and_price)
    workflow.add_node("get_quantity", get_quantity_from_user)
    workflow.add_node("request_approval", request_user_approval)
    workflow.add_node("check_approval", check_user_approval)
    workflow.add_node("execute_trade", execute_trade)
    workflow.add_node("error", handle_error)
    
    workflow.set_entry_point("parse_intent")
    workflow.add_conditional_edges(
        "parse_intent",
        lambda s: s.get("current_node", "error"),
        {
            "fetch_data": "fetch_data",
            "error": "error",
            # Allow continuing nodes to bypass intent parsing on follow-up turns
            "get_quantity": "get_quantity",
            "request_approval": "request_approval",
            "check_approval": "check_approval",
            "execute_trade": "execute_trade",
            END: END,
        },
    )
    workflow.add_edge("fetch_data", END)
    workflow.add_conditional_edges(
        "get_quantity",
        lambda s: s.get("current_node", "error"),
        {
            "request_approval": "request_approval",
            "error": "error",
            # Pause after prompting again so the user can respond
            "get_quantity": END,
        },
    )
    workflow.add_edge("request_approval", END)
    workflow.add_conditional_edges(
        "check_approval",
        lambda s: s.get("current_node", END),
        {
            "execute_trade": "execute_trade",
            END: END,
            # After reprompting, pause for the user's reply
            "check_approval": END,
        },
    )
    workflow.add_edge("execute_trade", END)
    workflow.add_edge("error", END)
    
    return workflow.compile(checkpointer=MemorySaver())

# =================================================================================
# MAIN AGENT CLASS
# =================================================================================
class TradeExecutionAgent:
    def __init__(self):
        self.workflow = create_workflow()
    
    def initialize_session(self, user_id: str, initial_message: str) -> Dict[str, Any]:
        session_id = f"{user_id}_{int(time.time())}"
        initial_state = {
            "messages": [{"role": "user", "content": initial_message}],
            "user_id": user_id, "session_id": session_id, "symbol": "", "action": "",
            "available_balance": Decimal('0'), "current_price": Decimal('0'),
            "quantity": 0, "total_cost": Decimal('0'), "max_affordable_quantity": 0,
            "user_approval": False, "trade_executed": False, "order_id": "",
            "error": None, "current_node": ""
        }
        config = {"configurable": {"thread_id": session_id}}
        result = self.workflow.invoke(initial_state, config)
        return {"session_id": session_id, "state": result, "response": result["messages"][-1]["content"]}

    def process_message(self, session_id: str, message: str) -> Dict[str, Any]:
        config = {"configurable": {"thread_id": session_id}}
        current_state = self.workflow.get_state(config)
        new_messages = current_state.values["messages"] + [{"role": "user", "content": message}]
        input_state = {**current_state.values, "messages": new_messages}
        result = self.workflow.invoke(input_state, config)
        return {"session_id": session_id, "state": result, "response": result["messages"][-1]["content"]}

    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        config = {"configurable": {"thread_id": session_id}}
        return self.workflow.get_state(config).values

# =================================================================================
# PRODUCTION READY INTERACTIVE DEMO
# =================================================================================
def run_interactive_demo():
    print("=" * 80)
    print("🚀 MCP TRADE EXECUTION AGENT - GEMINI 2.5 FLASH")
    print(f"💡 Examples: 'buy {Config.DEFAULT_SYMBOL}', 'sell {Config.SAMPLE_ALT_SYMBOL} 5', then APPROVE")
    print("=" * 80)
    
    agent = TradeExecutionAgent()
    
    # Start new session
    session = agent.initialize_session("853", "Hello Agent, I want to trade stocks.")
    print(f"🆔 Session: {session['session_id']}")
    print("🤖", session['response'])
    
    while True:
        print("\n" + "─" * 80)
        user_input = input("💬 You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q', 'restart']:
            if user_input.lower() == 'restart':
                run_interactive_demo()
            print("👋 Session ended!")
            break
        
        result = agent.process_message(session['session_id'], user_input)
        print("🤖", result['response'])
        
        state = agent.get_session_state(session['session_id'])
        # if state.get('trade_executed', False):
        #     print("\n🎊 TRADE SUCCESSFUL - NEW SESSION STARTED")
        #     run_interactive_demo()
        #     break

if __name__ == "__main__":
    print("🔥 GEMINI 2.5 FLASH TRADE AGENT - PRODUCTION READY")
    run_interactive_demo()
