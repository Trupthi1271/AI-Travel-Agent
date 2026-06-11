"""
TripWeaver AI — FastAPI Server
--------------------------------
Exposes the LangGraph agent via REST endpoints so any frontend
(React, Next.js, mobile, etc.) can consume it.

Endpoints:
  GET  /health          — liveness check
  POST /chat            — single conversational turn
  POST /chat/history    — send with full conversation history
  GET  /session/{id}    — retrieve search history for a session
  POST /plan-trip       — legacy structured endpoint (kept for compatibility)
"""

import uuid
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

app = FastAPI(
    title="TripWeaver AI",
    version="2.0.0",
    description="AI Travel Concierge API for Indian travelers",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # lock down to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(description="The user's message")
    session_id: Optional[str] = Field(default=None, description="Session ID for history tracking")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Previous messages in the conversation")


class TraceInfo(BaseModel):
    path: str
    latency_ms: int
    query_type: Optional[str] = None
    destination: Optional[str] = None
    tools_called: Optional[List[str]] = []
    iterations: Optional[int] = None
    error: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    trace: TraceInfo


class HealthResponse(BaseModel):
    status: str
    version: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness check — use this for deployment health probes."""
    return HealthResponse(status="ok", version="2.0.0")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a single message and get a response from the TripWeaver AI agent.

    - Uses LangGraph StateGraph as primary execution path
    - Falls back to LangChain AgentExecutor if graph fails
    - Persists search history to SQLite

    Example request:
    {
        "message": "What's the weather in Goa?",
        "session_id": "user_123",
        "history": []
    }
    """
    from langchain_core.messages import HumanMessage, AIMessage
    from agent.graph import run_graph

    session_id = request.session_id or str(uuid.uuid4())[:8]

    # Convert history to LangChain messages (trim AI content to avoid context bloat)
    chat_history = []
    for m in (request.history or [])[-8:]:
        if m.role == "user":
            chat_history.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            trimmed = m.content[:200] + "..." if len(m.content) > 200 else m.content
            chat_history.append(AIMessage(content=trimmed))

    try:
        response, trace = run_graph(
            user_input=request.message,
            chat_history=chat_history,
            session_id=session_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(
        response=response,
        session_id=session_id,
        trace=TraceInfo(**trace),
    )


@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Retrieve recent search history and saved itineraries for a session."""
    from database.db import get_recent_searches, get_itineraries

    searches = get_recent_searches(session_id, limit=10)
    itineraries = get_itineraries(session_id, limit=5)

    return {
        "session_id":   session_id,
        "searches":     searches,
        "itineraries":  itineraries,
    }


@app.get("/popular-destinations")
async def popular_destinations():
    """Return the most searched destinations across all sessions."""
    from database.db import get_popular_destinations
    return {"destinations": get_popular_destinations(limit=10)}


# ── Legacy endpoint (kept for backward compatibility) ─────────────────────────

@app.post("/plan-trip")
async def plan_trip_legacy(request: dict):
    """
    Legacy endpoint — redirects to /chat.
    Use /chat for new integrations.
    """
    destination = request.get("destination", "")
    days = request.get("days", 3)
    budget = request.get("budget", 0)

    message = f"Plan a {days}-day trip to {destination}"
    if budget:
        message += f" with a budget of ₹{budget}"

    chat_req = ChatRequest(message=message, session_id=None, history=[])
    return await chat(chat_req)
