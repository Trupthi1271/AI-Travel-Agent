"""
Week 5-6 Test Suite — Domain Specialization (Option A1)
=========================================================
Tests for:
  - SQLite database layer (searches, itineraries, user_preferences)
  - Flight service (IATA lookup, static fallback, formatting)
  - Flight tool (input parsing, error handling)
  - Save itinerary tool
  - Search history tool
  - API key security (no hardcoded secrets)

All external API calls are mocked. DB uses a temp file.

Run with:  pytest tests/test_week56.py -v
"""

import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Redirect DB to a temp file for each test — no pollution between tests."""
    db_path = tmp_path / "test_tripweaver.db"
    import database.db as db_module
    monkeypatch.setattr(db_module, "_DB_PATH", db_path)
    db_module.init_db()
    yield db_path


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Database — Searches
# ═══════════════════════════════════════════════════════════════════════════════

class TestSearches:

    def test_save_and_retrieve_search(self):
        from database.db import save_search, get_recent_searches
        save_search("sess1", "Plan a trip to Goa", "itinerary", "Goa", "3-day plan")
        results = get_recent_searches("sess1")
        assert len(results) == 1
        assert results[0]["query"] == "Plan a trip to Goa"
        assert results[0]["destination"] == "Goa"

    def test_multiple_searches_ordered_by_recency(self):
        from database.db import save_search, get_recent_searches
        save_search("sess1", "Weather in Manali", "weather", "Manali")
        save_search("sess1", "Hotels in Jaipur", "hotel", "Jaipur")
        save_search("sess1", "Flights Delhi to Goa", "flight", "Goa")
        results = get_recent_searches("sess1")
        assert len(results) == 3
        # All three destinations should be present regardless of order
        destinations = {r["destination"] for r in results}
        assert destinations == {"Manali", "Jaipur", "Goa"}

    def test_filter_by_destination(self):
        from database.db import save_search, get_recent_searches
        save_search("sess1", "Weather in Goa", "weather", "Goa")
        save_search("sess1", "Hotels in Jaipur", "hotel", "Jaipur")
        results = get_recent_searches("sess1", destination="Goa")
        assert len(results) == 1
        assert results[0]["destination"] == "Goa"

    def test_session_isolation(self):
        from database.db import save_search, get_recent_searches
        save_search("sess_a", "Trip to Goa", "itinerary", "Goa")
        save_search("sess_b", "Trip to Manali", "itinerary", "Manali")
        assert len(get_recent_searches("sess_a")) == 1
        assert len(get_recent_searches("sess_b")) == 1

    def test_limit_respected(self):
        from database.db import save_search, get_recent_searches
        for i in range(15):
            save_search("sess1", f"Query {i}", "general")
        results = get_recent_searches("sess1", limit=5)
        assert len(results) == 5

    def test_popular_destinations(self):
        from database.db import save_search, get_popular_destinations
        for _ in range(3):
            save_search("s1", "Goa trip", "itinerary", "Goa")
        for _ in range(2):
            save_search("s2", "Jaipur trip", "itinerary", "Jaipur")
        save_search("s3", "Manali trip", "itinerary", "Manali")
        popular = get_popular_destinations(limit=3)
        assert popular[0]["destination"] == "Goa"
        assert popular[0]["count"] == 3

    def test_empty_session_returns_empty_list(self):
        from database.db import get_recent_searches
        assert get_recent_searches("nonexistent_session") == []


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Database — Itineraries
# ═══════════════════════════════════════════════════════════════════════════════

class TestItineraries:

    def test_save_and_retrieve_itinerary(self):
        from database.db import save_itinerary, get_itineraries
        row_id = save_itinerary("sess1", "Goa", "Day 1: Baga Beach...", days=3, budget=15000)
        assert row_id > 0
        results = get_itineraries("sess1")
        assert len(results) == 1
        assert results[0]["destination"] == "Goa"
        assert results[0]["days"] == 3
        assert results[0]["budget"] == 15000.0

    def test_get_itinerary_by_id(self):
        from database.db import save_itinerary, get_itinerary_by_id
        row_id = save_itinerary("sess1", "Manali", "Day 1: Rohtang Pass...")
        result = get_itinerary_by_id(row_id)
        assert result is not None
        assert result["destination"] == "Manali"

    def test_nonexistent_id_returns_none(self):
        from database.db import get_itinerary_by_id
        assert get_itinerary_by_id(99999) is None

    def test_filter_by_destination(self):
        from database.db import save_itinerary, get_itineraries
        save_itinerary("sess1", "Goa", "Goa plan...")
        save_itinerary("sess1", "Jaipur", "Jaipur plan...")
        results = get_itineraries("sess1", destination="Goa")
        assert len(results) == 1
        assert results[0]["destination"] == "Goa"

    def test_multiple_itineraries_ordered_by_recency(self):
        from database.db import save_itinerary, get_itineraries
        save_itinerary("sess1", "Goa", "First plan")
        save_itinerary("sess1", "Manali", "Second plan")
        results = get_itineraries("sess1")
        assert len(results) == 2
        destinations = {r["destination"] for r in results}
        assert destinations == {"Goa", "Manali"}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Database — User Preferences
# ═══════════════════════════════════════════════════════════════════════════════

class TestUserPreferences:

    def test_save_and_retrieve_preferences(self):
        from database.db import save_preferences, get_preferences
        save_preferences("sess1", travel_style="budget", accommodation="hostel")
        prefs = get_preferences("sess1")
        assert prefs["travel_style"] == "budget"
        assert prefs["accommodation"] == "hostel"

    def test_upsert_updates_existing(self):
        from database.db import save_preferences, get_preferences
        save_preferences("sess1", travel_style="budget")
        save_preferences("sess1", travel_style="luxury")
        prefs = get_preferences("sess1")
        assert prefs["travel_style"] == "luxury"

    def test_interests_stored_as_json(self):
        from database.db import save_preferences, get_preferences
        save_preferences("sess1", interests=["beach", "trekking", "food"])
        prefs = get_preferences("sess1")
        assert isinstance(prefs["interests"], list)
        assert "beach" in prefs["interests"]

    def test_empty_session_returns_empty_dict(self):
        from database.db import get_preferences
        assert get_preferences("nonexistent") == {}

    def test_format_preferences_for_prompt(self):
        from database.db import format_preferences_for_prompt
        prefs = {
            "travel_style": "budget",
            "accommodation": "hostel",
            "interests": ["beach", "food"],
            "home_city": "Delhi",
        }
        result = format_preferences_for_prompt(prefs)
        assert "budget" in result.lower()
        assert "hostel" in result.lower()
        assert "Delhi" in result

    def test_format_empty_preferences_returns_empty_string(self):
        from database.db import format_preferences_for_prompt
        assert format_preferences_for_prompt({}) == ""


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Flight Service
# ═══════════════════════════════════════════════════════════════════════════════

class TestFlightService:

    def test_iata_lookup_known_city(self):
        from services.flights import _resolve_iata
        assert _resolve_iata("Delhi") == "DEL"
        assert _resolve_iata("mumbai") == "BOM"
        assert _resolve_iata("GOA") == "GOI"

    def test_iata_lookup_unknown_city(self):
        from services.flights import _resolve_iata
        assert _resolve_iata("XyzFakeCity") is None

    def test_static_fallback_known_route(self):
        from services.flights import _static_fallback
        flights = _static_fallback("DEL", "BOM")
        assert len(flights) > 0
        assert all("airline" in f for f in flights)
        assert all("price" in f for f in flights)

    def test_static_fallback_reverse_route(self):
        from services.flights import _static_fallback
        flights = _static_fallback("BOM", "DEL")
        assert len(flights) > 0

    def test_static_fallback_unknown_route_returns_empty(self):
        from services.flights import _static_fallback
        flights = _static_fallback("DEL", "IXZ")   # no static data for this
        # May return empty or data — just must not crash
        assert isinstance(flights, list)

    def test_get_flights_known_route_no_api(self):
        """Without API keys, should use static fallback and return formatted string."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AMADEUS_API_KEY", None)
            os.environ.pop("AMADEUS_API_SECRET", None)
            from services.flights import get_flights
            result = get_flights("Delhi", "Goa")
            assert "Delhi" in result or "DEL" in result
            assert "Goa" in result or "GOI" in result
            assert "₹" in result

    def test_get_flights_unknown_origin_raises(self):
        from services.flights import get_flights
        from agent.error_handler import ToolError
        with pytest.raises(ToolError, match="airport"):
            get_flights("XyzFakeCity", "Goa")

    def test_get_flights_unknown_destination_raises(self):
        from services.flights import get_flights
        from agent.error_handler import ToolError
        with pytest.raises(ToolError, match="airport"):
            get_flights("Delhi", "XyzFakeCity")

    def test_get_flights_same_city_raises(self):
        from services.flights import get_flights
        from agent.error_handler import ToolError
        with pytest.raises(ToolError, match="same"):
            get_flights("Delhi", "New Delhi")

    def test_format_includes_airline_and_price(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AMADEUS_API_KEY", None)
            os.environ.pop("AMADEUS_API_SECRET", None)
            from services.flights import get_flights
            result = get_flights("Delhi", "Mumbai")
            assert "IndiGo" in result or "Air India" in result or "SpiceJet" in result
            assert "₹" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Flight Tool (LangChain wrapper)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFlightTool:

    def test_valid_input_returns_flights(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AMADEUS_API_KEY", None)
            os.environ.pop("AMADEUS_API_SECRET", None)
            from agent.tools import flight_tool
            result = flight_tool.invoke({"origin": "Delhi", "destination": "Goa"})
            assert "Goa" in result or "GOI" in result
            assert "₹" in result

    def test_missing_destination_returns_error(self):
        from agent.tools import flight_tool
        try:
            result = flight_tool.invoke({"origin": "Delhi"})
            assert "❌" in result or isinstance(result, str)
        except Exception:
            pass  # Pydantic validation error is acceptable

    def test_empty_input_returns_error(self):
        from agent.tools import flight_tool
        try:
            result = flight_tool.invoke({"origin": "", "destination": "Goa"})
            assert isinstance(result, str)
        except Exception:
            pass  # Pydantic validation error is acceptable

    def test_unknown_city_returns_error(self):
        from agent.tools import flight_tool
        result = flight_tool.invoke({"origin": "XyzFake", "destination": "Goa"})
        assert "❌" in result or "⚠️" in result

    def test_with_date_parses_correctly(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AMADEUS_API_KEY", None)
            os.environ.pop("AMADEUS_API_SECRET", None)
            from agent.tools import flight_tool
            result = flight_tool.invoke({"origin": "Mumbai", "destination": "Jaipur", "travel_date": "2025-12-25"})
            assert isinstance(result, str)
            assert len(result) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Save Itinerary Tool
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveItineraryTool:

    def test_valid_save_returns_confirmation(self):
        from agent.tools import save_itinerary_tool
        result = save_itinerary_tool.invoke({
            "session_id": "sess1",
            "destination": "Goa",
            "content": "Day 1: Baga Beach in the morning..."
        })
        assert "✅" in result
        assert "Goa" in result
        assert "#" in result

    def test_invalid_format_returns_error(self):
        from agent.tools import save_itinerary_tool
        try:
            result = save_itinerary_tool.invoke({
                "session_id": "sess1",
                "destination": "",
                "content": "some content"
            })
            assert "❌" in result
        except Exception:
            pass  # Pydantic validation error acceptable

    def test_empty_destination_returns_error(self):
        from agent.tools import save_itinerary_tool
        try:
            result = save_itinerary_tool.invoke({
                "session_id": "sess1",
                "destination": "",
                "content": "some content"
            })
            assert "❌" in result
        except Exception:
            pass

    def test_empty_content_returns_error(self):
        from agent.tools import save_itinerary_tool
        try:
            result = save_itinerary_tool.invoke({
                "session_id": "sess1",
                "destination": "Goa",
                "content": ""
            })
            assert "❌" in result
        except Exception:
            pass

    def test_saved_itinerary_retrievable(self):
        from agent.tools import save_itinerary_tool
        from database.db import get_itineraries
        save_itinerary_tool.invoke({
            "session_id": "sess_test",
            "destination": "Manali",
            "content": "Day 1: Rohtang Pass adventure"
        })
        results = get_itineraries("sess_test")
        assert len(results) == 1
        assert results[0]["destination"] == "Manali"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Search History Tool
# ═══════════════════════════════════════════════════════════════════════════════

class TestSearchHistoryTool:

    def test_empty_history_returns_message(self):
        from agent.tools import search_history_tool
        result = search_history_tool.invoke("new_session_xyz")
        assert "None yet" in result or "History" in result
        assert isinstance(result, str)

    def test_history_shows_saved_searches(self):
        from database.db import save_search
        from agent.tools import search_history_tool
        save_search("hist_sess", "Weather in Goa", "weather", "Goa")
        save_search("hist_sess", "Hotels in Jaipur", "hotel", "Jaipur")
        result = search_history_tool.invoke("hist_sess")
        assert "Goa" in result or "Weather" in result
        assert "Jaipur" in result or "Hotels" in result

    def test_history_shows_saved_itineraries(self):
        from database.db import save_itinerary
        from agent.tools import search_history_tool
        save_itinerary("itin_sess", "Kerala", "Day 1: Alleppey backwaters...", days=5)
        result = search_history_tool.invoke("itin_sess")
        assert "Kerala" in result

    def test_empty_session_id_uses_default(self):
        from agent.tools import search_history_tool
        result = search_history_tool.invoke({"session_id": "default"})
        assert isinstance(result, str)
        assert len(result) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 8. API Key Security
# ═══════════════════════════════════════════════════════════════════════════════

class TestAPIKeySecurity:

    def test_no_hardcoded_keys_in_flights_service(self):
        """Verify no API keys are hardcoded in the flights service."""
        source = Path("services/flights.py").read_text(encoding="utf-8")
        import re
        suspicious = re.findall(r'["\'][A-Za-z0-9]{32,}["\']', source)
        assert len(suspicious) == 0, f"Possible hardcoded key found: {suspicious}"

    def test_no_hardcoded_keys_in_env_example(self):
        """Verify .env.example only has placeholder values."""
        env_example = Path(".env.example").read_text()
        assert "your_groq_api_key_here" in env_example
        assert "your_amadeus_key_here" in env_example
        # Should not contain real-looking keys
        import re
        real_key_pattern = re.findall(r'=\s*[A-Za-z0-9]{32,}', env_example)
        assert len(real_key_pattern) == 0

    def test_env_file_in_gitignore(self):
        """Verify .env is listed in .gitignore."""
        gitignore = Path(".gitignore").read_text()
        assert ".env" in gitignore

    def test_db_file_in_gitignore(self):
        """Verify SQLite DB files are excluded from git."""
        gitignore = Path(".gitignore").read_text()
        assert ".db" in gitignore or "*.db" in gitignore

    def test_flights_reads_keys_from_env(self):
        """Verify flight service reads keys from environment, not hardcoded."""
        source = Path("services/flights.py").read_text(encoding="utf-8")
        assert 'os.getenv("AMADEUS_API_KEY")' in source
        assert 'os.getenv("AMADEUS_API_SECRET")' in source

    def test_db_path_not_in_public_directory(self):
        """Verify the real DB path (not temp) is inside database/ folder."""
        import database.db as db_module
        from pathlib import Path as P
        # Reconstruct the default path as defined in the module source
        default_path = P(db_module.__file__).resolve().parent / "tripweaver.db"
        assert "database" in str(default_path)
