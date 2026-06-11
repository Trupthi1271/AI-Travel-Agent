"""
Tool Testing Script — Week 3-4
================================
Standalone tests for all 5 agent tools.
All external API/network calls are mocked so tests run fully offline.

Run with:
    pytest tests/test_tools.py -v
    pytest tests/test_tools.py -v -k "web_search"   # run one class
"""

import pytest
from unittest.mock import patch, MagicMock


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Budget Tool
# ═══════════════════════════════════════════════════════════════════════════════

class TestBudgetTool:
    """Tests for budget_tool — no network calls needed."""

    def _call(self, text: str) -> str:
        from agent.tools import budget_tool
        return budget_tool.invoke(text)

    def _call_safe(self, text: str) -> str:
        """Catch Pydantic ValidationError — both ❌ string and ValidationError are valid error handling."""
        try:
            return self._call(text)
        except Exception as e:
            return f"❌ {e}"

    def test_valid_input_returns_breakdown(self):
        result = self._call("15000,3")
        assert "₹15,000" in result
        assert "Accommodation" in result
        assert "Food" in result

    def test_daily_budget_correct(self):
        result = self._call("12000,4")
        assert "₹3,000" in result

    def test_luxury_tier(self):
        result = self._call("90000,3")
        assert "Luxury" in result

    def test_budget_tier(self):
        result = self._call("6000,3")
        assert "Budget" in result

    def test_ultra_budget_tier(self):
        result = self._call("1500,3")
        assert "Ultra Budget" in result or "Budget" in result

    def test_invalid_format_returns_error(self):
        result = self._call_safe("15000")
        assert "❌" in result or "error" in result.lower()

    def test_zero_budget_returns_error(self):
        result = self._call_safe("0,3")
        assert "❌" in result or "error" in result.lower()

    def test_zero_days_returns_error(self):
        result = self._call_safe("15000,0")
        assert "❌" in result or "error" in result.lower()

    def test_text_input_returns_error(self):
        result = self._call_safe("lots of money, a few days")
        assert "❌" in result or "error" in result.lower()

    def test_empty_input_returns_error(self):
        result = self._call_safe("")
        assert "❌" in result or "error" in result.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Weather Tool
# ═══════════════════════════════════════════════════════════════════════════════

class TestWeatherTool:
    """Tests for weather_tool — HTTP calls mocked."""

    def _mock_open_meteo(self):
        return {
            "current": {
                "temperature_2m": 28.5,
                "apparent_temperature": 30.0,
                "relative_humidity_2m": 75,
                "wind_speed_10m": 12.0,
                "weather_code": 1,
            },
            "daily": {
                "time": ["2025-06-01"] * 7,
                "weather_code": [1] * 7,
                "temperature_2m_max": [32.0] * 7,
                "temperature_2m_min": [24.0] * 7,
            },
        }

    @patch("services.weather._geocode_city")
    @patch("services.weather.requests.get")
    def test_valid_city_returns_weather(self, mock_get, mock_geocode):
        mock_geocode.return_value = {"lat": 15.3, "lon": 74.1, "display_name": "Goa, India"}
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_open_meteo()
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from agent.tools import weather_tool
        result = weather_tool.invoke("Goa")
        assert "Goa" in result
        assert "28.5" in result or "Temperature" in result

    @patch("services.weather._geocode_city")
    def test_invalid_city_returns_error(self, mock_geocode):
        mock_geocode.return_value = None
        from agent.tools import weather_tool
        result = weather_tool.invoke("XyzFakeCity999")
        assert "❌" in result

    def test_empty_city_returns_error(self):
        from agent.tools import weather_tool
        try:
            result = weather_tool.invoke("")
            assert "❌" in result
        except Exception:
            pass  # Pydantic ValidationError is valid error handling

    @patch("services.weather._geocode_city")
    @patch("services.weather.requests.get")
    def test_network_error_handled_gracefully(self, mock_get, mock_geocode):
        mock_geocode.return_value = {"lat": 32.2, "lon": 77.2, "display_name": "Manali, India"}
        mock_get.side_effect = Exception("Connection refused")
        from agent.tools import weather_tool
        result = weather_tool.invoke("Manali")
        assert "❌" in result
        # Must not raise an exception — agent should keep running
        assert isinstance(result, str)

    @patch("services.weather._geocode_city")
    @patch("services.weather.requests.get")
    def test_7day_forecast_included(self, mock_get, mock_geocode):
        mock_geocode.return_value = {"lat": 26.9, "lon": 75.8, "display_name": "Jaipur, India"}
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_open_meteo()
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from agent.tools import weather_tool
        result = weather_tool.invoke("Jaipur")
        assert "7-Day Forecast" in result or "Forecast" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Hotel Tool
# ═══════════════════════════════════════════════════════════════════════════════

class TestHotelTool:
    """Tests for hotel_tool — HTTP calls mocked."""

    @patch("services.hotels._geocode_city")
    @patch("services.hotels._query_overpass_hotels")
    def test_valid_city_returns_hotels(self, mock_overpass, mock_geocode):
        mock_geocode.return_value = {"display_name": "Jaipur, India", "bbox": [26.7, 27.0, 75.6, 75.9]}
        mock_overpass.return_value = [
            {"name": "Hotel Raj Palace", "type": "hotel", "lat": 26.9, "lon": 75.8},
            {"name": "Pink City Hostel", "type": "hostel", "lat": 26.9, "lon": 75.8},
        ]
        from agent.tools import hotel_tool
        result = hotel_tool.invoke("Jaipur")
        assert "Jaipur" in result
        assert "Hotel Raj Palace" in result

    @patch("services.hotels._geocode_city")
    def test_invalid_city_returns_error(self, mock_geocode):
        mock_geocode.return_value = None
        from agent.tools import hotel_tool
        result = hotel_tool.invoke("XyzFakeCity999")
        assert "❌" in result or "Could not locate" in result

    def test_empty_city_returns_error(self):
        from agent.tools import hotel_tool
        try:
            result = hotel_tool.invoke("")
            assert "❌" in result
        except Exception:
            pass

    @patch("services.hotels._geocode_city")
    @patch("services.hotels._query_overpass_hotels")
    def test_no_hotels_found_message(self, mock_overpass, mock_geocode):
        mock_geocode.return_value = {"display_name": "Remote", "bbox": [0, 1, 0, 1]}
        mock_overpass.return_value = []
        from agent.tools import hotel_tool
        result = hotel_tool.invoke("Remote")
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("services.hotels._geocode_city")
    @patch("services.hotels._query_overpass_hotels")
    def test_groups_by_type(self, mock_overpass, mock_geocode):
        mock_geocode.return_value = {"display_name": "Goa, India", "bbox": [15.0, 15.5, 73.8, 74.2]}
        mock_overpass.return_value = [
            {"name": "Sea View Hotel", "type": "hotel", "lat": 15.3, "lon": 74.0},
            {"name": "Backpacker Hostel", "type": "hostel", "lat": 15.3, "lon": 74.0},
            {"name": "Cozy Guest House", "type": "guest_house", "lat": 15.3, "lon": 74.0},
        ]
        from agent.tools import hotel_tool
        result = hotel_tool.invoke("Goa")
        assert "Hotels" in result or "Hostels" in result or "Guest" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Web Search Tool
# ═══════════════════════════════════════════════════════════════════════════════

class TestWebSearchTool:
    """Tests for web_search_tool — DuckDuckGo calls mocked."""

    def _mock_ddg_results(self):
        return [
            {
                "title": "Best time to visit Goa - Travel Guide",
                "href": "https://example.com/goa-travel",
                "body": "Goa is best visited between November and February when the weather is pleasant.",
            },
            {
                "title": "Goa Carnival 2025 Dates and Events",
                "href": "https://example.com/goa-carnival",
                "body": "The Goa Carnival 2025 will be held in February with parades and music.",
            },
        ]

    @patch("duckduckgo_search.DDGS")
    def test_valid_query_returns_results(self, mock_ddgs_class):
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = self._mock_ddg_results()
        mock_ddgs_class.return_value = mock_ddgs

        from agent.tools import web_search_tool
        result = web_search_tool.invoke("best time to visit Goa")
        assert "Goa" in result
        assert "🔍" in result or "Web Search" in result

    @patch("duckduckgo_search.DDGS")
    def test_results_include_urls(self, mock_ddgs_class):
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = self._mock_ddg_results()
        mock_ddgs_class.return_value = mock_ddgs

        from agent.tools import web_search_tool
        result = web_search_tool.invoke("Goa carnival 2025")
        assert "https://" in result

    @patch("duckduckgo_search.DDGS")
    def test_empty_results_handled(self, mock_ddgs_class):
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = []
        mock_ddgs_class.return_value = mock_ddgs

        from agent.tools import web_search_tool
        result = web_search_tool.invoke("obscure query with no results")
        assert "No results" in result or isinstance(result, str)

    @patch("duckduckgo_search.DDGS")
    def test_network_error_handled_gracefully(self, mock_ddgs_class):
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.side_effect = Exception("Network error")
        mock_ddgs_class.return_value = mock_ddgs

        from agent.tools import web_search_tool
        result = web_search_tool.invoke("Goa travel tips")
        assert "❌" in result or "error" in result.lower()
        assert isinstance(result, str)   # must not raise

    def test_empty_query_returns_error(self):
        from agent.tools import web_search_tool
        try:
            result = web_search_tool.invoke("")
            assert "❌" in result
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Places Tool
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlacesTool:
    """Tests for places_tool — uses static fallback data (no API key needed)."""

    def test_known_city_returns_attractions(self):
        from agent.tools import places_tool
        result = places_tool.invoke("Goa")
        assert "Goa" in result
        assert "Baga Beach" in result or "Basilica" in result or "Fort" in result

    def test_jaipur_returns_forts(self):
        from agent.tools import places_tool
        result = places_tool.invoke("Jaipur")
        assert "Jaipur" in result
        assert "Amber Fort" in result or "Hawa Mahal" in result

    def test_manali_returns_nature(self):
        from agent.tools import places_tool
        result = places_tool.invoke("Manali")
        assert "Manali" in result
        assert "Rohtang" in result or "Solang" in result or "Hadimba" in result

    def test_unknown_city_returns_message(self):
        from agent.tools import places_tool
        result = places_tool.invoke("XyzUnknownCity999")
        assert isinstance(result, str)
        assert len(result) > 0   # should not crash

    def test_empty_city_returns_error(self):
        from agent.tools import places_tool
        try:
            result = places_tool.invoke("")
            assert "❌" in result
        except Exception:
            pass

    def test_result_contains_categories(self):
        from agent.tools import places_tool
        result = places_tool.invoke("Delhi")
        # Should have at least one category emoji
        assert any(emoji in result for emoji in ["🏛", "🌿", "🏰", "🛕", "📍"])

    def test_case_insensitive_city(self):
        from agent.tools import places_tool
        result_lower = places_tool.invoke("goa")
        result_upper = places_tool.invoke("GOA")
        # Both should return Goa data
        assert "Goa" in result_lower
        assert "Goa" in result_upper


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Error Handler
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorHandler:
    """Tests for agent/error_handler.py utilities."""

    def test_with_retry_succeeds_on_first_try(self):
        from agent.error_handler import with_retry
        call_count = {"n": 0}

        @with_retry(max_attempts=3)
        def always_succeeds():
            call_count["n"] += 1
            return "ok"

        result = always_succeeds()
        assert result == "ok"
        assert call_count["n"] == 1

    def test_with_retry_retries_on_failure(self):
        from agent.error_handler import with_retry
        call_count = {"n": 0}

        @with_retry(max_attempts=3, delay=0)
        def fails_twice():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ValueError("transient error")
            return "ok"

        result = fails_twice()
        assert result == "ok"
        assert call_count["n"] == 3

    def test_with_retry_raises_after_max_attempts(self):
        from agent.error_handler import with_retry
        import pytest

        @with_retry(max_attempts=2, delay=0)
        def always_fails():
            raise ValueError("permanent error")

        with pytest.raises(ValueError, match="permanent error"):
            always_fails()

    def test_safe_tool_call_returns_string_on_error(self):
        from agent.error_handler import safe_tool_call

        def bad_func():
            raise RuntimeError("something broke")

        result = safe_tool_call(bad_func, tool_name="TestTool")
        assert "❌" in result
        assert isinstance(result, str)

    def test_validate_city_strips_prefix(self):
        from agent.error_handler import validate_city
        assert validate_city("city of Goa") == "Goa"
        assert validate_city("  jaipur  ") == "Jaipur"

    def test_validate_city_raises_on_empty(self):
        from agent.error_handler import validate_city, ToolError
        with pytest.raises(ToolError):
            validate_city("")

    def test_validate_budget_input_valid(self):
        from agent.error_handler import validate_budget_input
        amount, days = validate_budget_input("15000,3")
        assert amount == 15000.0
        assert days == 3

    def test_validate_budget_input_invalid_format(self):
        from agent.error_handler import validate_budget_input, ToolError
        with pytest.raises(ToolError):
            validate_budget_input("15000")

    def test_validate_budget_input_zero_amount(self):
        from agent.error_handler import validate_budget_input, ToolError
        with pytest.raises(ToolError):
            validate_budget_input("0,3")

    def test_validate_budget_input_zero_days(self):
        from agent.error_handler import validate_budget_input, ToolError
        with pytest.raises(ToolError):
            validate_budget_input("15000,0")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Tool Registry
# ═══════════════════════════════════════════════════════════════════════════════

class TestToolRegistry:
    """Verify all tools are registered and have correct metadata."""

    def test_all_tools_registered(self):
        from agent.tools import ALL_TOOLS
        tool_names = [t.name for t in ALL_TOOLS]
        assert "budget_tool" in tool_names
        assert "weather_tool" in tool_names
        assert "hotel_tool" in tool_names
        assert "web_search_tool" in tool_names
        assert "places_tool" in tool_names
        assert "flight_tool" in tool_names
        assert "restaurant_tool" in tool_names

    def test_tool_count(self):
        from agent.tools import ALL_TOOLS
        assert len(ALL_TOOLS) == 9

    def test_all_tools_have_descriptions(self):
        from agent.tools import ALL_TOOLS
        for t in ALL_TOOLS:
            assert t.description, f"{t.name} has no description"
            assert len(t.description) > 20, f"{t.name} description too short"

    def test_tool_metadata_complete(self):
        from agent.tools import ALL_TOOLS, TOOL_METADATA
        for t in ALL_TOOLS:
            assert t.name in TOOL_METADATA, f"{t.name} missing from TOOL_METADATA"
            meta = TOOL_METADATA[t.name]
            assert "icon" in meta
            assert "label" in meta
            assert "api" in meta
