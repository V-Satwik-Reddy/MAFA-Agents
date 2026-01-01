from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from execution_agent import run_execute_agent
from general_agent import run_general_agent
from market_search_agent import run_market_research_agent
from http_client import set_request_token
import random
app = FastAPI(title="Execution Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ExecuteAgentRequest(BaseModel):
	query: str
	userId: int


@app.post("/execute-agent")
def execute_agent_endpoint(payload: ExecuteAgentRequest, authorization: str | None = Header(default=None)):
	"""HTTP endpoint that runs the agent and returns its reply."""
	if not authorization or not authorization.lower().startswith("bearer "):
		raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
	token = authorization.split(" ", 1)[1]
	try:
		set_request_token(token)
		result = run_execute_agent(user_message=payload.query, user_id=payload.userId)
		return {"data": result, "userId": payload.userId}
	except Exception as exc:  # pragma: no cover - simple passthrough
		print("Error executing agent:", exc)
		raise HTTPException(status_code=500, detail="Failed to execute agent") from exc
	finally:
		set_request_token(None)

@app.post("/market-research-agent")
def market_research_agent_endpoint(payload: ExecuteAgentRequest, authorization: str | None = Header(default=None)):
	"""HTTP endpoint that runs the agent and returns its reply."""
	if not authorization or not authorization.lower().startswith("bearer "):
		raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
	token = authorization.split(" ", 1)[1]
	try:
		set_request_token(token)
		result = run_market_research_agent(user_message=payload.query, user_id=payload.userId)
		return {"data": result, "userId": payload.userId}
	except Exception as exc:  # pragma: no cover - simple passthrough
		print("Error executing agent:", exc)
		raise HTTPException(status_code=500, detail="Failed to execute agent") from exc
	finally:
		set_request_token(None)


@app.post("/general-agent")
def general_agent_endpoint(payload: ExecuteAgentRequest, authorization: str | None = Header(default=None)):
	"""HTTP endpoint that runs the general agent and returns its reply."""
	if not authorization or not authorization.lower().startswith("bearer "):
		raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
	token = authorization.split(" ", 1)[1]
	try:
		set_request_token(token)
		result = run_general_agent(user_message=payload.query, user_id=payload.userId)
		return {"data": result, "userId": payload.userId}
	except Exception as exc:  # pragma: no cover - simple passthrough
		print("Error executing agent:", exc)
		raise HTTPException(status_code=500, detail="Failed to execute agent") from exc
	finally:
		set_request_token(None)
  
@app.post("/agenttest")
def agent_test_endpoint():
	"""Simple test endpoint to verify the API is working."""
	res=[{
  "agentResponse": "Ready to execute trades for MSFT.",
  "useTool": "execute",
  "toolData": { "symbol": "MSFT" }},{
  "agentResponse": "Here is the chart for AAPL.",
  "useTool": "graph",
  "toolData": { "symbol": "AAPL" }
},{
  "agentResponse": "Showing your TSLA transaction history.",
  "useTool": "transactions",
  "toolData": { "symbol": "TSLA" }
}]
	i=random.randint(0,len(res)-1)
	return res[i]