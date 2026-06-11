"""
Agent Tools — Track B Level
-----------------------------
All 8 LangChain tools with:
  - Pydantic argument schemas (strict validation, no LLM guessing)
  - Circuit breaker pattern (primary API → fallback → static)
  - Structured return strings for reliable LLM parsing
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import tool

from agent.budget import calculate_budget
from agent.error_handler import safe_tool_call, validate_city, validate_budget_input, ToolError
from agent.logger import logger, metrics, log_tool_call
from services.weather import get_weather
from services.hotels import get_hotels
from services.web_search import get_web_search
from services.places import get_places
from services.flights import get_flights
from services.restaurants import get_restaurants
from database.db import (
    save_search, save_itinerary, get_recent_searches,
    get_itineraries, get_popular_destinations,
)


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic Input Schemas
# ══════════════════════════════════════════════════════════════════════════════

class CityInput(BaseModel):
    city: str = Field(
        description="City name in English, e.g. 'Mumbai', 'Goa', 'Jaipur'. No country suffix."
    )

    @field_validator("city")
    @classmethod
    def city_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("City name cannot be empty.")
        if len(v) > 100:
            raise ValueError("City name too long.")
        return v.title()


class BudgetInput(BaseModel):
    input_text: str = Field(
        description="Budget and days in format 'AMOUNT,DAYS'. Digits only. E.g. '15000,3' for ₹15,000 over 3 days."
    )

    @field_validator("input_text")
    @classmethod
    def validate_format(cls, v: str) -> str:
        v = v.strip()
        parts = v.split(",")
        if len(parts) != 2:
            raise ValueError("Format must be 'AMOUNT,DAYS' e.g. '15000,3'")
        try:
            amount = float(parts[0])
            days = int(parts[1])
        except ValueError:
            raise ValueError("Both AMOUNT and DAYS must be numbers.")
        if amount <= 0:
            raise ValueError("Budget must be greater than ₹0.")
        if days <= 0:
            raise ValueError("Days must be at least 1.")
        return v


class FlightInput(BaseModel):
    origin: str = Field(description="Departure city name in English, e.g. 'Delhi'")
    destination: str = Field(description="Arrival city name in English, e.g. 'Goa'")
    travel_date: Optional[str] = Field(
        default=None,
        description="Travel date in YYYY-MM-DD format. Optional — defaults to tomorrow."
    )

    @field_validator("origin", "destination")
    @classmethod
    def city_not_empty(cls, v: str) -> str:
        v = v.strip().title()
        if not v:
            raise ValueError("City name cannot be empty.")
        return v


class WebSearchInput(BaseModel):
    query: str = Field(
        description="Specific search query for travel information, e.g. 'Goa carnival 2025 dates'"
    )

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Search query cannot be empty.")
        if len(v) > 200:
            raise ValueError("Query too long — keep it under 200 characters.")
        return v


class SaveItineraryInput(BaseModel):
    session_id: str = Field(description="Current session ID. Use 'default' if unknown.")
    destination: str = Field(description="Destination city name, e.g. 'Goa'")
    content: str = Field(description="Full itinerary text to save.")

    @field_validator("destination")
    @classmethod
    def dest_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Destination cannot be empty.")
        return v.title()

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Itinerary content cannot be empty.")
        return v


class HistoryInput(BaseModel):
    session_id: str = Field(
        default="default",
        description="Current session ID to retrieve history for. Use 'default' if unknown."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Circuit Breaker Helper
# ══════════════════════════════════════════════════════════════════════════════

def _circuit_breaker(primary_func, fallback_func, tool_name: str, *args, **kwargs) -> str:
    """
    Try primary function first. On any exception, log it and run fallback.
    This implements the circuit breaker pattern for API resilience.
    """
    import time
    start = time.perf_counter()
    try:
        result = primary_func(*args, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        metrics.record_tool_call(tool_name, latency_ms, error=False)
        return result
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        metrics.record_tool_call(tool_name, latency_ms, error=True)
        logger.warning(f"[{tool_name}] Primary failed ({exc}), switching to fallback")
        try:
            return fallback_func(*args, **kwargs)
        except Exception as fallback_exc:
            logger.error(f"[{tool_name}] Fallback also failed: {fallback_exc}")
            return f"⚠️ {tool_name} temporarily unavailable. Please try again shortly."


# ══════════════════════════════════════════════════════════════════════════════
# Tools
# ══════════════════════════════════════════════════════════════════════════════

@tool("budget_tool", args_schema=BudgetInput)
def budget_tool(input_text: str) -> str:
    """Calculate a daily travel budget breakdown.

    Use when user provides BOTH a total rupee amount AND number of days.
    Examples: 'my budget is ₹15000 for 3 days', 'I have 50000 INR for 7 days'.
    Do NOT use for vague cost questions — only when explicit numbers are given.
    """
    return safe_tool_call(calculate_budget, input_text, tool_name="BudgetTool")


@tool("weather_tool", args_schema=CityInput)
def weather_tool(city: str) -> str:
    """Get real-time weather and 7-day forecast for an Indian travel destination.

    Use for: weather, rain, temperature, climate, forecast questions.
    Always call this — never answer weather questions from memory.
    """
    return safe_tool_call(get_weather, city, tool_name="WeatherTool")


@tool("hotel_tool", args_schema=CityInput)
def hotel_tool(city: str) -> str:
    """Find hotels, hostels, and guest houses in an Indian city.

    Use for: hotels, accommodation, where to stay, lodging questions.
    Never make up hotel names — always call this tool.
    """
    return safe_tool_call(get_hotels, city, tool_name="HotelTool")


@tool("web_search_tool", args_schema=WebSearchInput)
def web_search_tool(query: str) -> str:
    """Search the web for current travel information.

    Use for: festivals, visa requirements, travel advisories, safety, current events.
    Do NOT use for weather, hotels, budget, or flights — use dedicated tools for those.
    """
    return safe_tool_call(get_web_search, query, tool_name="WebSearchTool")


@tool("places_tool", args_schema=CityInput)
def places_tool(city: str) -> str:
    """Get top tourist attractions and points of interest for an Indian city.

    Use for: attractions, things to do, must-visit spots, sightseeing questions.
    Never make up place names — always call this tool.
    """
    return safe_tool_call(get_places, city, tool_name="PlacesTool")


@tool("flight_tool", args_schema=FlightInput)
def flight_tool(origin: str, destination: str, travel_date: Optional[str] = None) -> str:
    """Search for flights between two Indian cities.

    Use for: flights, airfare, flying between cities.
    Never make up flight data — always call this tool.
    travel_date is optional (YYYY-MM-DD format), defaults to tomorrow.
    """
    try:
        origin = validate_city(origin)
        destination = validate_city(destination)
    except ToolError as exc:
        return f"❌ {exc}"
    return safe_tool_call(get_flights, origin, destination, travel_date, tool_name="FlightTool")


@tool("save_itinerary_tool", args_schema=SaveItineraryInput)
def save_itinerary_tool(session_id: str, destination: str, content: str) -> str:
    """Save a generated trip itinerary to the database.

    Use when user says: 'save this trip', 'remember this plan', 'bookmark this itinerary'.
    """
    try:
        destination = validate_city(destination)
    except ToolError as exc:
        return f"❌ {exc}"
    try:
        row_id = save_itinerary(
            session_id=session_id or "default",
            destination=destination,
            content=content,
        )
        return (
            f"✅ **Itinerary saved!**\n\n"
            f"📍 Destination: {destination}\n"
            f"🔖 ID: #{row_id}\n\n"
            f"Retrieve anytime by asking 'show my saved trips'."
        )
    except Exception as exc:
        return f"❌ Could not save itinerary: {exc}"


@tool("search_history_tool", args_schema=HistoryInput)
def search_history_tool(session_id: str = "default") -> str:
    """Retrieve the user's recent searches and saved itineraries.

    Use when user asks: 'show my history', 'what have I searched', 'my saved trips'.
    """
    session_id = (session_id or "default").strip()
    try:
        searches    = get_recent_searches(session_id, limit=8)
        itineraries = get_itineraries(session_id, limit=5)
        popular     = get_popular_destinations(limit=5)

        lines = ["📋 **Your Travel History**\n"]

        if searches:
            lines.append("**🔍 Recent Searches:**")
            for s in searches:
                dest = f" → {s['destination']}" if s.get("destination") else ""
                lines.append(f"  • {s['query'][:60]}{dest}  _{s['created_at'][:10]}_")
        else:
            lines.append("**🔍 Recent Searches:** None yet")

        lines.append("")

        if itineraries:
            lines.append("**💾 Saved Itineraries:**")
            for it in itineraries:
                days   = f" · {it['days']} days" if it.get("days") else ""
                budget = f" · ₹{it['budget']:,.0f}" if it.get("budget") else ""
                lines.append(
                    f"  • **{it['destination']}**{days}{budget}  "
                    f"_(#{it['id']}, saved {it['created_at'][:10]})_"
                )
        else:
            lines.append("**💾 Saved Itineraries:** None yet — ask me to save a trip plan!")

        if popular:
            lines.append("\n**🌟 Popular Destinations (all users):**")
            for p in popular:
                lines.append(f"  • {p['destination']} ({p['count']} searches)")

        return "\n".join(lines)

    except Exception as exc:
        return f"❌ Could not retrieve history: {exc}"


# ── 9. Restaurant Tool ────────────────────────────────────────────────────────

@tool("restaurant_tool", args_schema=CityInput)
def restaurant_tool(city: str) -> str:
    """Find top restaurants, cafes, and food spots in an Indian city.

    Use for: restaurants, where to eat, food, cafes, street food, dining questions.
    Examples: 'best restaurants in Goa', 'where to eat in Jaipur', 'street food in Mumbai'.
    Never make up restaurant names — always call this tool.
    """
    return safe_tool_call(get_restaurants, city, tool_name="RestaurantTool")


# ══════════════════════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════════════════════

ALL_TOOLS = [
    budget_tool,
    weather_tool,
    hotel_tool,
    web_search_tool,
    places_tool,
    flight_tool,
    save_itinerary_tool,
    search_history_tool,
    restaurant_tool,
]

TOOL_METADATA = {
    "budget_tool":         {"icon": "💰", "label": "Budget Calculator",   "api": "Built-in"},
    "weather_tool":        {"icon": "🌤", "label": "Live Weather",         "api": "Weatherstack / Open-Meteo"},
    "hotel_tool":          {"icon": "🏨", "label": "Hotel Finder",         "api": "Amadeus / OpenStreetMap"},
    "web_search_tool":     {"icon": "🔍", "label": "Web Search",           "api": "DuckDuckGo (free)"},
    "places_tool":         {"icon": "🗺️", "label": "Places & Attractions", "api": "OpenTripMap / static"},
    "flight_tool":         {"icon": "✈️", "label": "Flight Search",        "api": "Amadeus / static"},
    "save_itinerary_tool": {"icon": "💾", "label": "Save Itinerary",       "api": "SQLite (local)"},
    "search_history_tool": {"icon": "📋", "label": "Search History",       "api": "SQLite (local)"},
    "restaurant_tool":     {"icon": "🍽️", "label": "Restaurants",          "api": "OpenTripMap / curated"},
}
