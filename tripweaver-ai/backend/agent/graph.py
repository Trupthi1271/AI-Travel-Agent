"""
LangGraph Agent Workflow — Track B
------------------------------------
Replaces the LangChain AgentExecutor with a LangGraph StateGraph.

Architecture:
  START
    │
    ▼
  [classify]  — determine query type and required tools
    │
    ▼
  [agent]     — LLM decides which tool(s) to call
    │
    ├── tool call? ──▶ [tools] ──▶ back to [agent]
    │
    └── done? ──▶ [format] ──▶ END

State:
  TravelAgentState — typed TypedDict with full conversation context
"""

from __future__ import annotations

import os
import time
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from agent.tools import ALL_TOOLS
from agent.logger import logger, metrics, log_agent_run

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=str(_ENV_PATH))


# ── Typed State ───────────────────────────────────────────────────────────────

class TravelAgentState(TypedDict):
    """
    Full state passed between LangGraph nodes.
    `messages` uses add_messages reducer — appends rather than overwrites.
    """
    messages:       Annotated[List[BaseMessage], add_messages]
    query_type:     str                   # weather / hotel / flight / budget / itinerary / general
    destination:    Optional[str]         # extracted destination city
    session_id:     str                   # user session identifier
    iteration:      int                   # safety counter to prevent infinite loops
    error:          Optional[str]         # last error message if any
    tool_calls_made: List[str]            # names of tools called this run


# ── Query classifier ──────────────────────────────────────────────────────────

def _classify_query(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["weather", "rain", "temperature", "forecast", "climate"]):
        return "weather"
    if any(w in t for w in ["hotel", "stay", "accommodation", "hostel", "resort"]):
        return "hotel"
    if any(w in t for w in ["flight", "fly", "airfare", "airline"]):
        return "flight"
    if any(w in t for w in ["budget", "cost", "expense", "inr", "₹"]):
        return "budget"
    if any(w in t for w in ["plan", "itinerary", "trip", "visit", "days"]):
        return "itinerary"
    if any(w in t for w in ["place", "attraction", "see", "do", "tourist"]):
        return "places"
    return "general"


def _extract_destination(text: str, current_dest: Optional[str] = None) -> Optional[str]:
    """
    Extract destination from query text.
    If multiple cities are found (e.g. 'flights from Mumbai to Delhi'),
    use the last-mentioned city and clear stale context.
    """
    import re
    known = [
        "goa", "jaipur", "manali", "delhi", "mumbai", "kerala", "udaipur",
        "shimla", "bangalore", "chennai", "kolkata", "agra", "varanasi",
        "rishikesh", "darjeeling", "ooty", "ladakh", "kashmir", "coorg",
        "munnar", "leh", "kochi", "pune", "hyderabad", "amritsar", "jodhpur",
        "mysore", "hampi", "kodaikanal", "pondicherry", "andaman",
    ]
    text_lower = text.lower()
    found = [city for city in known if city in text_lower]

    if len(found) > 1:
        # Multi-city query (e.g. "flights from Mumbai to Delhi")
        # Use the destination (last city mentioned) and don't carry stale context
        logger.debug(f"Multi-city detected: {found} — using last mentioned: {found[-1]}")
        return found[-1].title()

    if len(found) == 1:
        return found[0].title()

    # Pattern fallback: "to X", "in X"
    match = re.search(r"\b(?:to|in|at|visit|trip to)\s+([A-Z][a-z]+)", text)
    if match:
        return match.group(1)

    return current_dest  # preserve existing context if no city found
    return None


# ── LLM setup — lazy, built once on first agent_node call ────────────────────

_llm = None
_llm_with_tools = None


def _get_llm_with_tools():
    global _llm, _llm_with_tools
    if _llm_with_tools is None:
        _llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1024,
            timeout=30,
        )
        _llm_with_tools = _llm.bind_tools(ALL_TOOLS)
    return _llm_with_tools


SYSTEM_PROMPT = """You are TripWeaver, an AI Travel Concierge for Indian travelers.

TOOL ROUTING — call the correct tool FIRST:
| Query type | Tool |
|---|---|
| weather / rain / temperature | weather_tool |
| hotels / stay / accommodation | hotel_tool |
| budget — amount AND days given | budget_tool |
| flights / airfare | flight_tool |
| attractions / places / sightseeing | places_tool |
| restaurants / food / where to eat | restaurant_tool |
| festivals / visa / news | web_search_tool |
| "save this trip" | save_itinerary_tool |
| "my history" | search_history_tool |

MULTI-TOOL RULE: When user says "plan a trip" or "plan a X-day trip to [city]":
Call ALL of these tools in sequence: weather_tool, places_tool, hotel_tool, budget_tool.
Merge all results into one complete travel package response.

SINGLE-QUERY RULE: For all other queries, call ONE tool then stop immediately and respond.

NEVER make up hotel names, flights, restaurants, or attractions.
After tool returns, paste output then add your insights and recommendations.
ALWAYS call the correct tool — NEVER answer weather/hotel/flight/places/restaurant questions from memory.

━━━ RESPONSE FORMAT ━━━

Weather query:
## 🌤️ Weather in [City]
🌡 Temperature: X°C (Feels like X°C) · 💧 Humidity: X% · 💨 Wind: X km/h · 🌥 Condition: [condition]
📅 7-Day Forecast:
| Date | Condition | High | Low |
|---|---|---|---|
| [date] | [condition] | X°C | X°C |
🧭 Travel Advice: [advice from tool]
_Source: [source]_
---
### 🗺️ Best Places Given This Weather
| # | Place | Why it suits the weather |
|---|---|---|
| 1 | **[Place]** | [reason] |
| 2 | **[Place]** | [reason] |
| 3 | **[Place]** | [reason] |
---
### 💡 Quick Tips
- **Pack:** [2-3 items] · **Tip:** [one local advice]

Hotels:
## 🏨 Hotels in [City]
[paste hotel_tool output — every hotel name]
---
### 💡 Booking Tips · [peak season advice] · [city tip]

Flights:
## ✈️ Flights: [Origin] → [Destination]
📅 Date: [date]
| # | Airline | Departure | Arrival | Duration | Class | Price |
|---|---|---|---|---|---|---|
| 1 | [airline] | [time] | [time] | [duration] | ECONOMY | ₹X |
_Source: [source]_ · Book on MakeMyTrip, Cleartrip, or airline website.
---
### 💡 Tips · Compare fares · Early morning = cheapest

Budget:
## 💰 Budget Breakdown
[paste budget_tool output — every line]

Restaurants:
## 🍽️ Where to Eat in [City]
[paste restaurant_tool output — every restaurant]
---
### 💡 Food Tips · [local specialty] · [best area for food]

Places:
## 🗺️ Top Places in [City]
[paste places_tool output — every attraction with description]

Full Trip Plan (multi-tool):
## 🗺️ [X]-Day Trip to [City]
### 🌤️ Weather & Best Time · [weather summary]
### 🗺️ Top Attractions · [places output]
### 🏨 Where to Stay · [top 3-5 hotels]
### 🍽️ Where to Eat · [top restaurants]
### Day 1 — [Theme] | Time | Activity | Cost |
### Day 2 — [Theme] (repeat)
### 💰 Budget Summary | Category | Cost |
### ✈️ Travel Tips · Best time · Getting there · Don't miss

Use ₹ · --- dividers · markdown tables · keep concise"""


# ── Graph nodes ───────────────────────────────────────────────────────────────

def classify_node(state: TravelAgentState) -> TravelAgentState:
    """Classify the latest user message and extract destination."""
    messages = state["messages"]
    last_human = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )
    query_type = _classify_query(last_human)
    destination = _extract_destination(last_human, state.get("destination"))

    logger.debug(f"Classify: type={query_type}, dest={destination}")
    metrics.query_types[query_type] += 1

    return {
        **state,
        "query_type":  query_type,
        "destination": destination,
        "iteration":   0,
        "error":       None,
        "tool_calls_made": [],
    }


def agent_node(state: TravelAgentState) -> TravelAgentState:
    """LLM decides what to do — call a tool or respond directly."""
    llm_with_tools = _get_llm_with_tools()  # lazy, cached after first call
    system = SystemMessage(content=SYSTEM_PROMPT)
    recent_messages = state["messages"][-6:]
    messages = [system] + recent_messages

    start = time.perf_counter()
    try:
        response = llm_with_tools.invoke(messages)
        latency_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"Agent LLM call: {latency_ms:.0f}ms")
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.error(f"Agent LLM error: {exc}")
        error_msg = AIMessage(content=f"I encountered an error: {exc}. Please try again.")
        return {**state, "messages": [error_msg], "error": str(exc)}

    return {
        **state,
        "messages":  [response],
        "iteration": state.get("iteration", 0) + 1,
    }


def format_node(state: TravelAgentState) -> TravelAgentState:
    """
    Final structural audit pass.
    - Validates required markdown headers are present
    - Detects formatting drift and logs warnings
    - Trims tool output from message history to prevent context bloat
    """
    messages = state["messages"]
    if not messages:
        return state

    last = messages[-1]

    if isinstance(last, AIMessage) and last.tool_calls:
        logger.warning("format_node received message with pending tool calls")
        return state

    if isinstance(last, AIMessage):
        content = last.content
        q_type = state.get("query_type", "general")

        # Validate required headers per query type
        required_headers = {
            "weather":   "## 🌤️",
            "hotel":     "## 🏨",
            "flight":    "## ✈️",
            "budget":    "## 💰",
            "itinerary": "## 🗺️",
            "places":    "## 🗺️",
        }
        expected = required_headers.get(q_type)
        if expected and expected not in content:
            logger.warning(f"Formatting drift for {q_type} — missing header '{expected}'")
            metrics.record_tool_call("format_drift", 0, error=True)

    return state


# ── Routing logic ─────────────────────────────────────────────────────────────

def should_continue(state: TravelAgentState) -> Literal["tools", "format", "end"]:
    """Decide next node after agent_node."""
    messages = state["messages"]
    last = messages[-1] if messages else None

    # For trip planning, allow up to 5 iterations (4 tools + final response)
    # For all other queries, cap at 2 iterations (1 tool + response)
    q_type = state.get("query_type", "general")
    max_iter = 5 if q_type == "itinerary" else 2

    if state.get("iteration", 0) >= max_iter:
        logger.warning(f"Max iterations ({max_iter}) reached for {q_type} — forcing end")
        return "end"

    if state.get("error"):
        return "end"

    if isinstance(last, AIMessage) and last.tool_calls:
        tool_names = [tc["name"] for tc in last.tool_calls]
        logger.debug(f"Routing to tools: {tool_names}")
        return "tools"

    return "format"


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph() -> Any:
    """Build and compile the LangGraph StateGraph with optional SQLite checkpointer."""

    # Custom tools node that tracks calls and compresses large outputs
    def tracked_tool_node(state: TravelAgentState):
        last = state["messages"][-1]
        called = []
        if isinstance(last, AIMessage) and last.tool_calls:
            called = [tc["name"] for tc in last.tool_calls]

        result = ToolNode(ALL_TOOLS).invoke(state)

        # Compress tool messages that are very large to prevent context bloat
        # Keep full output for LLM to use, but trim repeated whitespace/emojis
        compressed_msgs = []
        for msg in result.get("messages", []):
            if isinstance(msg, ToolMessage) and len(msg.content) > 2000:
                # Trim to 2000 chars — enough for the LLM to format the response
                trimmed = msg.content[:2000] + "\n[...output trimmed for context efficiency]"
                compressed_msgs.append(ToolMessage(
                    content=trimmed,
                    tool_call_id=msg.tool_call_id,
                    name=getattr(msg, "name", None),
                ))
                logger.debug(f"Compressed tool output: {len(msg.content)} → 2000 chars")
            else:
                compressed_msgs.append(msg)

        existing = state.get("tool_calls_made") or []
        return {
            "messages":        compressed_msgs,
            "tool_calls_made": existing + called,
        }

    builder = StateGraph(TravelAgentState)

    # Add nodes
    builder.add_node("classify", classify_node)
    builder.add_node("agent",    agent_node)
    builder.add_node("tools",    tracked_tool_node)
    builder.add_node("format",   format_node)

    # Add edges
    builder.add_edge(START,      "classify")
    builder.add_edge("classify", "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools":  "tools",
            "format": "format",
            "end":    END,
        },
    )
    builder.add_edge("tools",  "agent")   # after tool call, back to agent
    builder.add_edge("format", END)

    graph = builder.compile()
    logger.info("LangGraph StateGraph compiled successfully")
    return graph


# ── Public runner ─────────────────────────────────────────────────────────────

# Lazy compilation — built on first use, not at import time
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_graph(
    user_input: str,
    chat_history: List[BaseMessage],
    session_id: str = "default",
    destination: Optional[str] = None,
) -> tuple:
    """
    Run the LangGraph workflow for a single user query.
    Returns (response_text, trace_dict).
    """
    start = time.perf_counter()
    error_occurred = False
    final_state = {}

    initial_state: TravelAgentState = {
        "messages":        chat_history[-6:] + [HumanMessage(content=user_input)],
        "query_type":      "general",
        "destination":     destination,
        "session_id":      session_id,
        "iteration":       0,
        "error":           None,
        "tool_calls_made": [],
    }

    try:
        final_state = _get_graph().invoke(initial_state)
        messages = final_state.get("messages", [])

        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                response = msg.content
                break
        else:
            for msg in reversed(messages):
                if isinstance(msg, ToolMessage):
                    response = msg.content
                    break
            else:
                response = "I couldn't generate a response. Please try again."

    except Exception as exc:
        error_occurred = True
        logger.error(f"Graph execution error: {exc}")
        response = f"I encountered an error: {exc}. Please try again."

    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        metrics.record_agent_run(latency_ms, error=error_occurred)
        logger.info(f"Graph run complete: {latency_ms:.0f}ms")

    trace = {
        "path":         "LangGraph StateGraph",
        "latency_ms":   round(latency_ms),
        "session_id":   session_id,
        "query_type":   final_state.get("query_type", "general"),
        "destination":  final_state.get("destination"),
        "tools_called": final_state.get("tool_calls_made", []),
        "iterations":   final_state.get("iteration", 0),
        "error":        final_state.get("error"),
    }

    return response, trace
