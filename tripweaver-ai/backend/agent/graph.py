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
    # Itinerary check FIRST — "plan a budget trip" should be itinerary, not budget
    if any(w in t for w in ["plan", "itinerary", "trip to", "visit for"]):
        return "itinerary"
    if any(w in t for w in ["weather", "rain", "temperature", "forecast", "climate"]):
        return "weather"
    if any(w in t for w in ["hotel", "stay", "accommodation", "hostel", "resort"]):
        return "hotel"
    if any(w in t for w in ["flight", "fly", "airfare", "airline"]):
        return "flight"
    if any(w in t for w in ["budget", "cost", "expense", "inr", "₹"]):
        return "budget"
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


def _apply_itinerary_budget_defaults(text: str) -> str:
    """
    Add an explicit budget-tool instruction for itinerary requests where the user
    gives a travel tier but no amount. This keeps "budget trip" plans grounded.
    """
    import re

    t = text.lower()
    is_trip_plan = any(w in t for w in ["plan", "itinerary", "trip"])
    if not is_trip_plan:
        return text

    has_amount = bool(re.search(r"(?:₹|rs\.?|inr)\s*\d|(?:\d[\d,]*)\s*(?:₹|rs\.?|inr)", t))
    if has_amount:
        return text

    days_match = re.search(r"(\d+)\s*(?:-| )?\s*day", t)
    if not days_match:
        return text

    days = int(days_match.group(1))
    per_day = None
    if "budget" in t or "cheap" in t or "low cost" in t:
        per_day = 5000
    elif "comfort" in t or "mid-range" in t or "mid range" in t:
        per_day = 8000
    elif "luxury" in t or "premium" in t:
        per_day = 15000

    if per_day is None:
        return text

    total = per_day * days
    return (
        f"{text}\n\n"
        f"Budget assumption for this itinerary: use budget_tool(\"{total},{days}\"). "
        f"Keep the total trip budget at Rs. {total:,} unless the user gives a different amount."
    )


# ── LLM setup — lazy, built once on first agent_node call ────────────────────

_llm = None
_llm_with_tools = None
_llm_provider = None


def _build_gemini_with_tools():
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set")

    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        google_api_key=gemini_key,
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=0.3,
        max_tokens=1024,
    )
    return llm, llm.bind_tools(ALL_TOOLS)


def _switch_to_gemini_fallback() -> bool:
    """Switch the cached LLM to Gemini after a Groq runtime failure."""
    global _llm, _llm_with_tools, _llm_provider
    try:
        _llm, _llm_with_tools = _build_gemini_with_tools()
        _llm_provider = "gemini"
        logger.warning("LLM: Switched to Gemini fallback after Groq runtime error")
        return True
    except Exception as exc:
        logger.error(f"Gemini runtime fallback failed: {exc}")
        return False


def _clear_cached_llm() -> None:
    """Clear cached LLM after quota/rate-limit failures."""
    global _llm, _llm_with_tools, _llm_provider
    _llm = None
    _llm_with_tools = None
    _llm_provider = None


def _is_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(token in text for token in ["resource_exhausted", "quota", "rate limit", "429"])


def _get_llm_with_tools():
    global _llm, _llm_with_tools, _llm_provider
    if _llm_with_tools is None:
        groq_key = os.getenv("GROQ_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if groq_key:
            try:
                _llm = ChatGroq(
                    groq_api_key=groq_key,
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.3,
                    max_tokens=1024,
                    timeout=30,
                )
                _llm_with_tools = _llm.bind_tools(ALL_TOOLS)
                _llm_provider = "groq"
                logger.info("LLM: Using Groq llama-3.3-70b-versatile")
            except Exception as e:
                logger.warning(f"Groq failed: {e} — trying Gemini fallback")
                _llm = None
                _llm_with_tools = None
                _llm_provider = None

        if _llm_with_tools is None and gemini_key:
            try:
                _llm, _llm_with_tools = _build_gemini_with_tools()
                _llm_provider = "gemini"
                logger.info("LLM: Using Gemini fallback")
            except Exception as e:
                logger.error(f"Gemini fallback also failed: {e}")
                raise RuntimeError("No LLM available. Set GROQ_API_KEY or GEMINI_API_KEY in .env")

    return _llm_with_tools


SYSTEM_PROMPT = """You are TripWeaver, an AI Travel Concierge for Indian travelers.

TOOL ROUTING:
- weather/rain/temperature/forecast -> weather_tool
- hotels/stay/accommodation -> hotel_tool
- budget with amount AND days -> budget_tool
- flights/airfare -> flight_tool
- top places/attractions/sightseeing -> places_tool
- restaurants/food/where to eat -> restaurant_tool
- festivals/visa/news -> web_search_tool
- save this trip -> save_itinerary_tool
- my history/saved trips -> search_history_tool

TRIP PLANNING: When user says plan a trip or plan X-day trip to City:
Call these tools in order: weather_tool, places_tool, hotel_tool, budget_tool
Then write ONE complete itinerary using the actual data from all tools.

BUDGET RULE:
If the user says "budget trip" but does not give an amount, assume Rs. 5,000 per day.
If the user says "comfort trip" but does not give an amount, assume Rs. 8,000 per day.
If the user says "luxury trip" but does not give an amount, assume Rs. 15,000 per day.
Total budget = days x per-day amount.
Never exceed the assumed total unless the user gives a higher budget.
For "Plan a 3-day budget trip", call budget_tool with "15000,3".

SINGLE QUERY: Call ONE tool then respond immediately.

STRICT OUTPUT RULES:
1. Use clean Markdown only.
2. Always use pipe Markdown tables.
3. Never output tab-separated tables.
4. Never output plain aligned columns.
5. Every table must have: header row, separator row, then one data row per line.
6. Every table must look exactly like:
| Column A | Column B |
|---|---|
| Value | Value |
7. Never put two table rows on the same line.
8. Use Rs. only when calling budget_tool, but use the rupee symbol in the final response.
9. The response itself should include only one main heading.
10. Do not repeat the same title twice.
11. For trip planning responses, always start with: ## Trip to City (X Days)
12. Weather data inside a trip plan goes under a ### Weather Snapshot section, not as the main heading.
13. Never call a tool that does not exist in the tool list above.
14. NEVER answer weather/hotel/flight/places/restaurants from memory - always call the tool.
15. Use actual data from tool results, not placeholders.
16. Weather responses must include a "Best Places Given This Weather" table with exactly 3 places.
17. For weather responses, paste the weather_tool output as-is and do not remove the current weather table or 3-day forecast.

RESPONSE FORMAT:

Weather:
## Weather in City

| Parameter | Value |
|---|---|
| Temperature | actual value |
| Condition | actual value |
| Humidity | actual value |
| Wind Speed | actual value |

Travel Advice: actual advice from tool

### Best Places Given This Weather

| Place | Reason |
|---|---|
| actual place | actual reason |
| actual place | actual reason |
| actual place | actual reason |

Pack: actual items based on current conditions

Hotels:
## Hotels in City

| # | Hotel Name | Category |
|---|---|---|
| 1 | actual hotel | actual rating |

Flights: paste exact flight table from tool as-is

Budget: paste exact budget table from tool as-is

Restaurants:
## Where to Eat in City
| Restaurant | About |
|---|---|
| actual name | actual description |

Trip Plan:
## 3-Day Budget Trip to City

### Weather Snapshot
- Condition: actual condition
- Temperature: actual temperature
- Travel advice: actual advice

### Top Attractions

| Place | Description |
|---|---|
| actual place | actual description |

### Where to Stay

| # | Hotel | Category |
|---:|---|---|
| 1 | actual hotel | actual rating |

### Day 1 - Theme

| Time | Activity | Cost |
|---|---|---:|
| Morning | actual activity | actual cost |
| Afternoon | actual activity | actual cost |
| Evening | actual activity | actual cost |

### Day 2 - Theme

| Time | Activity | Cost |
|---|---|---:|
| Morning | actual activity | actual cost |
| Afternoon | actual activity | actual cost |
| Evening | actual activity | actual cost |

### Budget Summary

| Category | Total |
|---|---:|
| Accommodation | actual total |
| Food | actual total |
| Transport | actual total |
| Activities | actual total |
| Total | actual total |

### Travel Tips
- Best time: actual
- Getting there: actual
- Local tip: actual

Top places:

| Place | Description |
|---|---|
| actual place | actual description |
"""


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


def _direct_weather_response(destination: str) -> Optional[str]:
    """Call weather_tool and places_tool directly — no LLM needed."""
    try:
        from services.weather import get_weather
        from services.places import get_places
        weather = get_weather(destination)
        places_output = get_places(destination)
        # Extract top 3 places for the "Best Places" section
        place_lines = [l for l in places_output.split("\n") if l.startswith("| **")][:3]
        places_table = ""
        if place_lines:
            places_table = (
                "\n\n### Best Places Given This Weather\n\n"
                "| Place | Why visit |\n|---|---|\n"
            )
            for line in place_lines:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                name = parts[0].replace("**", "") if parts else "—"
                desc = parts[1][:80] if len(parts) > 1 else "Great spot to visit"
                places_table += f"| **{name}** | {desc} |\n"
        return weather + places_table
    except Exception as e:
        logger.error(f"Direct weather response failed: {e}")
        return None


def _direct_places_response(destination: str) -> Optional[str]:
    """Call places_tool directly — no LLM needed."""
    try:
        from services.places import get_places
        return get_places(destination)
    except Exception as e:
        logger.error(f"Direct places response failed: {e}")
        return None


def agent_node(state: TravelAgentState) -> TravelAgentState:
    """LLM decides what to do — call a tool or respond directly."""
    global _llm_provider
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
        if _llm_provider == "groq" and _switch_to_gemini_fallback():
            try:
                retry_start = time.perf_counter()
                response = _get_llm_with_tools().invoke(messages)
                retry_ms = (time.perf_counter() - retry_start) * 1000
                logger.info(f"Agent LLM retry via Gemini: {retry_ms:.0f}ms")
            except Exception as retry_exc:
                logger.error(f"Agent LLM Gemini retry error: {retry_exc}")
                if _is_quota_error(retry_exc):
                    _clear_cached_llm()
                error_msg = AIMessage(content=f"I encountered an error: {retry_exc}. Please try again.")
                return {**state, "messages": [error_msg], "error": str(retry_exc)}
        else:
            logger.error(f"Agent LLM error: {exc}")
            if _is_quota_error(exc):
                _clear_cached_llm()
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
            "weather":   "## Weather",
            "hotel":     "## 🏨",
            "flight":    "## ✈️",
            "budget":    "## 💰",
            "itinerary": "## Trip",
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


def _direct_tool_response(user_input: str) -> Optional[tuple[str, Dict[str, Any]]]:
    """Bypass the LLM for simple single-tool queries."""
    import re

    text = user_input.lower()
    destination = _extract_destination(user_input)

    if destination and any(w in text for w in ["plan", "itinerary", "trip"]):
        days_match = re.search(r"(\d+)\s*(?:-| )?\s*day", text)
        if days_match:
            from agent.budget import calculate_budget
            from services.hotels import get_hotels
            from services.places import get_places
            from services.weather import get_weather

            days = int(days_match.group(1))
            amount_match = re.search(r"(?:₹|rs\.?|inr)\s*(\d[\d,]*)|(\d[\d,]*)\s*(?:₹|rs\.?|inr)", text)
            if amount_match:
                amount = (amount_match.group(1) or amount_match.group(2)).replace(",", "")
            elif any(w in text for w in ["luxury", "premium"]):
                amount = str(days * 15000)
            elif any(w in text for w in ["comfort", "mid-range", "mid range"]):
                amount = str(days * 8000)
            else:
                amount = str(days * 5000)

            weather = get_weather(destination)
            places = get_places(destination)
            hotels = get_hotels(destination)
            budget = calculate_budget(f"{amount},{days}")

            itinerary_lines = [
                f"## Trip to {destination} ({days} Days)",
                "",
                "### Weather Snapshot",
                weather,
                "",
                "### Top Attractions",
                places,
                "",
                "### Where to Stay",
                hotels,
                "",
                "### Day-by-Day Plan",
            ]
            themes = ["Arrival & Local Exploration", "Sightseeing & Experiences", "Culture & Relaxed Departure"]
            for day in range(1, days + 1):
                theme = themes[min(day - 1, len(themes) - 1)]
                itinerary_lines.extend([
                    "",
                    f"#### Day {day} - {theme}",
                    "",
                    "| Time | Activity | Cost |",
                    "|---|---|---:|",
                    f"| Morning | Visit a top attraction in {destination} | ₹500 |",
                    f"| Afternoon | Local sightseeing and food break | ₹700 |",
                    f"| Evening | Market/cafe walk and rest | ₹300 |",
                ])
            itinerary_lines.extend([
                "",
                "### Budget Summary",
                budget,
                "",
                "### Travel Tips",
                "- Keep outdoor activities flexible around the weather.",
                "- Confirm hotel and transport prices before booking.",
                "- Carry cash for local markets and small vendors.",
            ])

            return "\n".join(itinerary_lines), {
                "path": "Direct itinerary composer",
                "query_type": "itinerary",
                "destination": destination,
                "tools_called": ["weather_tool", "places_tool", "hotel_tool", "budget_tool"],
                "iterations": 0,
                "error": None,
            }

    if destination and any(w in text for w in ["weather", "rain", "temperature", "forecast", "climate"]):
        from services.weather import get_weather
        return get_weather(destination), {
            "path": "Direct weather tool",
            "query_type": "weather",
            "destination": destination,
            "tools_called": ["weather_tool"],
            "iterations": 0,
            "error": None,
        }

    if any(w in text for w in ["flight", "flights", "fly", "airfare"]):
        match = re.search(r"\bfrom\s+([a-zA-Z ]+?)\s+to\s+([a-zA-Z ]+?)(?:\s+on\s+(\d{4}-\d{2}-\d{2}))?$", user_input, re.I)
        if match:
            from services.flights import get_flights
            origin = match.group(1).strip()
            dest = match.group(2).strip()
            travel_date = match.group(3)
            return get_flights(origin, dest, travel_date), {
                "path": "Direct flight tool",
                "query_type": "flight",
                "destination": dest.title(),
                "tools_called": ["flight_tool"],
                "iterations": 0,
                "error": None,
            }

    if destination and any(w in text for w in ["hotel", "hotels", "stay", "accommodation", "hostel", "resort"]):
        from services.hotels import get_hotels
        return get_hotels(destination), {
            "path": "Direct hotel tool",
            "query_type": "hotel",
            "destination": destination,
            "tools_called": ["hotel_tool"],
            "iterations": 0,
            "error": None,
        }

    if destination and any(w in text for w in ["top places", "places to visit", "visit in", "attractions", "sightseeing"]):
        from services.places import get_places
        return get_places(destination), {
            "path": "Direct places tool",
            "query_type": "places",
            "destination": destination,
            "tools_called": ["places_tool"],
            "iterations": 0,
            "error": None,
        }

    if destination and any(w in text for w in ["restaurant", "restaurants", "where to eat", "food", "cafe", "dining"]):
        from services.restaurants import get_restaurants
        return get_restaurants(destination), {
            "path": "Direct restaurant tool",
            "query_type": "restaurant",
            "destination": destination,
            "tools_called": ["restaurant_tool"],
            "iterations": 0,
            "error": None,
        }

    budget_match = re.search(r"(\d[\d,]*)\s*(?:inr|rs\.?|₹).*?(\d+)\s*(?:-| )?\s*day", text)
    if not budget_match:
        budget_match = re.search(r"(?:budget|₹|rs\.?|inr).*?(\d[\d,]*).*?(\d+)\s*(?:-| )?\s*day", text)
    if budget_match and any(w in text for w in ["budget", "cost", "expense", "inr", "rs", "₹"]):
        from agent.budget import calculate_budget
        amount = budget_match.group(1).replace(",", "")
        days = budget_match.group(2)
        return calculate_budget(f"{amount},{days}"), {
            "path": "Direct budget tool",
            "query_type": "budget",
            "destination": destination,
            "tools_called": ["budget_tool"],
            "iterations": 0,
            "error": None,
        }

    return None


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
    resolved_input = _apply_itinerary_budget_defaults(user_input)

    direct = _direct_tool_response(user_input)
    if direct:
        response, trace = direct
        trace = {
            **trace,
            "latency_ms": round((time.perf_counter() - start) * 1000),
            "session_id": session_id,
        }
        metrics.record_agent_run(trace["latency_ms"], error=False)
        logger.info(f"Direct tool response complete: {trace['latency_ms']}ms")
        return response, trace

    # Only pass user messages as context — AI responses cause contamination
    # because the LLM confuses previous tool outputs with current query context
    user_history = [m for m in chat_history[-4:] if isinstance(m, HumanMessage)]

    initial_state: TravelAgentState = {
        "messages":        user_history + [HumanMessage(content=resolved_input)],
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
