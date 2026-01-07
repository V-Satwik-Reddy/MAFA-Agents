from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from agents.execution_agent import run_execute_agent  # type: ignore
from agents.general_agent import run_general_agent  # type: ignore
from agents.market_search_agent import run_market_research_agent  # type: i
from agents.portfolio_manager_agent import run_portfolio_manager_agent  # type: ignore
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
  
@app.post("/portfolio-manager-agent")
def portfolio_manager_agent_endpoint(payload: ExecuteAgentRequest, authorization: str | None = Header(default=None)):
	"""HTTP endpoint that runs the portfolio manager agent and returns its reply."""
	if not authorization or not authorization.lower().startswith("bearer "):
		raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
	token = authorization.split(" ", 1)[1]
	try:
		set_request_token(token)
		result = run_portfolio_manager_agent(user_message=payload.query, user_id=payload.userId)
		return {"data": result, "userId": payload.userId}
	except Exception as exc:  # pragma: no cover - simple passthrough
		print("Error executing agent:", exc)
		raise HTTPException(status_code=500, detail="Failed to execute agent") from exc
	finally:
		set_request_token(None)