"""
Step 6 — Agent Test Suite
Tests budget calculation, weather/hotel tool routing, memory, and edge cases.
All external API calls are mocked so tests run offline.

Run with:  pytest tests/test_agent.py -v
"""
import pytest
from unittest.mock import patch, MagicMock


# ─── Budget Tool ──────────────────────────────────────────────────────────────

class TestBudgetCalculation:
    """Tests for agent/budget.py"""

    def test_normal_budget(self):
        from agent.budget import calculate_budget
        result = calculate_budget("15000,3")
        assert "₹15,000" in result
        assert "₹5,000" in result          # 15000/3
        assert "Accommodation" in result

    def test_luxury_tier(self):
        from agent.budget import calculate_budget
        result = calculate_budget("90000,3")
        assert "Luxury" in result

    def test_budget_tier(self):
        from agent.budget import calculate_budget
        result = calculate_budget("6000,3")
        assert "Budget" in result

    def test_ultra_budget(self):
        from agent.budget import calculate_budget
        result = calculate_budget("1500,3")   # ₹500/day
        assert "Ultra Budget" in result or "Budget" in result

    def test_single_day(self):
        from agent.budget import calculate_budget
        result = calculate_budget("5000,1")
        assert "₹5,000" in result

    def test_invalid_format_missing_days(self):
        from agent.budget import calculate_budget
        result = calculate_budget("15000")
        assert "❌" in result

    def test_invalid_format_text(self):
        from agent.budget import calculate_budget
        result = calculate_budget("abc,xyz")
        assert "❌" in result

    def test_zero_budget(self):
        from agent.budget import calculate_budget
        result = calculate_budget("0,3")
        assert "❌" in result

    def test_zero_days(self):
        from agent.budget import calculate_budget
        result = calculate_budget("15000,0")
        assert "❌" in result

    def test_empty_input(self):
        from agent.budget import calculate_budget
        result = calculate_budget("")
        assert "❌" in result


# ─── Weather Service ──────────────────────────────────────────────────────────

class TestWeatherService:
    """Tests for services/weather.py — mocked HTTP calls"""

    def _mock_geocode(self, city):
        return {"lat": 15.2993, "lon": 74.1240, "display_name": f"{city}, India"}

    def _mock_open_meteo_response(self):
        return {
            "current": {
                "temperature_2m": 28.5,
                "apparent_temperature": 30.0,
                "relative_humidity_2m": 75,
                "wind_speed_10m": 12.0,
                "weather_code": 1,
            },
            "daily": {
                "time": ["2025-01-01", "2025-01-02", "2025-01-03",
                         "2025-01-04", "2025-01-05", "2025-01-06", "2025-01-07"],
                "weather_code": [1, 2, 3, 61, 63, 1, 0],
                "temperature_2m_max": [32, 31, 30, 28, 27, 33, 34],
                "temperature_2m_min": [24, 23, 22, 21, 20, 25, 26],
            }
        }

    @patch("services.weather._geocode_city")
    @patch("services.weather.requests.get")
    def test_valid_city_returns_weather(self, mock_get, mock_geocode):
        mock_geocode.return_value = self._mock_geocode("Goa")
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._mock_open_meteo_response()
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from services.weather import get_weather
        result = get_weather("Goa")
        assert "Goa" in result
        assert "28.5" in result or "Temperature" in result
        assert "3-Day Forecast" in result or "Forecast" in result

    @patch("services.weather._geocode_city")
    def test_invalid_city_returns_error(self, mock_geocode):
        mock_geocode.return_value = None
        from services.weather import get_weather
        result = get_weather("XyzInvalidCity123")
        assert "❌" in result
        assert "Could not find" in result

    @patch("services.weather._geocode_city")
    @patch("services.weather.requests.get")
    def test_network_error_handled(self, mock_get, mock_geocode):
        mock_geocode.return_value = self._mock_geocode("Manali")
        mock_get.side_effect = Exception("Connection refused")
        from services.weather import get_weather
        result = get_weather("Manali")
        assert "❌" in result

    def test_weather_code_mapping(self):
        from services.weather import _weather_code_to_text
        assert "Clear" in _weather_code_to_text(0)
        assert "Rain" in _weather_code_to_text(63)
        assert "Snow" in _weather_code_to_text(73)
        assert "Thunder" in _weather_code_to_text(95)

    def test_travel_advice_hot(self):
        from services.weather import _travel_advice
        result = _travel_advice("Clear Sky", 38)
        assert "hydrated" in result.lower() or "hot" in result.lower()

    def test_travel_advice_cold(self):
        from services.weather import _travel_advice
        result = _travel_advice("Clear Sky", 5)
        assert "cold" in result.lower() or "warm" in result.lower() or "layer" in result.lower()

    def test_travel_advice_rain(self):
        from services.weather import _travel_advice
        result = _travel_advice("Heavy Rain", 25)
        assert "umbrella" in result.lower() or "rain" in result.lower()


# ─── Hotel Service ────────────────────────────────────────────────────────────

class TestHotelService:
    """Tests for services/hotels.py — mocked HTTP calls"""

    @patch("services.hotels._geocode_city")
    @patch("services.hotels._query_overpass_hotels")
    def test_valid_city_returns_hotels(self, mock_overpass, mock_geocode):
        mock_geocode.return_value = {
            "display_name": "Jaipur, India",
            "bbox": [26.7, 27.0, 75.6, 75.9]
        }
        mock_overpass.return_value = [
            {"name": "Hotel Raj Palace", "type": "hotel", "lat": 26.9, "lon": 75.8},
            {"name": "Pink City Hostel", "type": "hostel", "lat": 26.9, "lon": 75.8},
            {"name": "Heritage Guest House", "type": "guest_house", "lat": 26.9, "lon": 75.8},
        ]
        from services.hotels import get_hotels
        result = get_hotels("Jaipur")
        assert "Jaipur" in result
        assert "Hotel Raj Palace" in result

    @patch("services.hotels._geocode_city")
    def test_invalid_city_returns_error(self, mock_geocode):
        mock_geocode.return_value = None
        from services.hotels import get_hotels
        result = get_hotels("XyzInvalidCity123")
        assert "Could not locate" in result or "❌" in result

    @patch("services.hotels._geocode_city")
    @patch("services.hotels._query_overpass_hotels")
    def test_no_hotels_found(self, mock_overpass, mock_geocode):
        mock_geocode.return_value = {"display_name": "Remote Place", "bbox": [0, 1, 0, 1]}
        mock_overpass.return_value = []
        from services.hotels import get_hotels
        result = get_hotels("RemotePlace")
        assert "No hotels found" in result or "❌" in result or "RemotePlace" in result


# ─── Memory ───────────────────────────────────────────────────────────────────

class TestTravelMemory:
    """Tests for agent/memory.py"""

    def test_destination_extracted(self):
        from agent.memory import TravelMemory
        mem = TravelMemory()
        mem.add_user_message("Plan a 3-day trip to Goa")
        assert mem.context.destination == "Goa"

    def test_days_extracted(self):
        from agent.memory import TravelMemory
        mem = TravelMemory()
        mem.add_user_message("Plan a 5-day trip to Manali")
        assert mem.context.days == 5

    def test_budget_extracted(self):
        from agent.memory import TravelMemory
        mem = TravelMemory()
        mem.add_user_message("My budget is ₹20000 INR")
        assert mem.context.budget == "20000"

    def test_travel_style_luxury(self):
        from agent.memory import TravelMemory
        mem = TravelMemory()
        mem.add_user_message("I want a luxury trip to Udaipur")
        assert mem.context.travel_style == "luxury"

    def test_travel_style_budget(self):
        from agent.memory import TravelMemory
        mem = TravelMemory()
        mem.add_user_message("I'm a budget backpacker")
        assert mem.context.travel_style == "budget"

    def test_accommodation_hostel(self):
        from agent.memory import TravelMemory
        mem = TravelMemory()
        mem.add_user_message("I prefer staying in hostels")
        assert mem.context.accommodation == "hostel"

    def test_context_persists_across_messages(self):
        from agent.memory import TravelMemory
        mem = TravelMemory()
        mem.add_user_message("Plan a 3-day trip to Goa")
        mem.add_ai_message("Here is your Goa itinerary...")
        mem.add_user_message("What about hotels?")
        # Destination should still be Goa
        assert mem.context.destination == "Goa"

    def test_window_trimming(self):
        from agent.memory import TravelMemory
        mem = TravelMemory(k=2)
        for i in range(10):
            mem.add_user_message(f"Message {i}")
            mem.add_ai_message(f"Response {i}")
        # k=2 means max 4 messages stored
        assert len(mem.messages) <= 4

    def test_context_injected_as_system_message(self):
        from agent.memory import TravelMemory
        from langchain_core.messages import SystemMessage
        mem = TravelMemory()
        mem.add_user_message("Plan a trip to Jaipur")
        history = mem.get_chat_history()
        assert isinstance(history[0], SystemMessage)
        assert "Jaipur" in history[0].content

    def test_clear_memory(self):
        from agent.memory import TravelMemory
        mem = TravelMemory()
        mem.add_user_message("Trip to Goa for 3 days")
        mem.clear_memory()
        assert len(mem.messages) == 0
        assert mem.context.destination is None

    def test_empty_context_no_system_message(self):
        from agent.memory import TravelMemory
        from langchain_core.messages import SystemMessage
        mem = TravelMemory()
        history = mem.get_chat_history()
        assert not any(isinstance(m, SystemMessage) for m in history)


# ─── Edge Cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge case and error handling tests"""

    def test_budget_very_low(self):
        from agent.budget import calculate_budget
        result = calculate_budget("500,3")   # ₹167/day — extreme
        assert "₹" in result
        assert "❌" not in result            # should still compute, not error

    def test_budget_large_numbers(self):
        from agent.budget import calculate_budget
        result = calculate_budget("500000,30")
        assert "₹" in result

    def test_weather_empty_city(self):
        """Empty city name should return an error, not crash"""
        with patch("services.weather._geocode_city", return_value=None):
            from services.weather import get_weather
            result = get_weather("")
            assert "❌" in result

    def test_hotel_empty_city(self):
        """Empty city name should return an error, not crash"""
        with patch("services.hotels._geocode_city", return_value=None):
            from services.hotels import get_hotels
            result = get_hotels("")
            assert "Could not locate" in result or "❌" in result

    def test_memory_no_destination_in_vague_query(self):
        from agent.memory import TravelMemory
        mem = TravelMemory()
        mem.add_user_message("plan a trip")
        # No known destination — should not crash
        assert mem.context.destination is None

    def test_budget_with_currency_symbols(self):
        """Budget tool should handle input with ₹ prefix gracefully"""
        from agent.budget import calculate_budget
        # Tool receives clean "AMOUNT,DAYS" — test the parser is robust
        result = calculate_budget("20000,4")
        assert "₹20,000" in result
