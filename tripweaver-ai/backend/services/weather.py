import os
import requests
from typing import Dict, List, Optional

USER_AGENT = "TripWeaverAI/1.0 (contact: support@tripweaver.ai)"


def _geocode_city(city: str) -> Optional[Dict]:
    """Geocode a city name to lat/lon using Nominatim"""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": city, "format": "json", "limit": 1}
        headers = {"User-Agent": USER_AGENT}
        r = requests.get(url, params=params, headers=headers, timeout=8)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        return {
            "lat": float(data[0]["lat"]),
            "lon": float(data[0]["lon"]),
            "display_name": data[0]["display_name"]
        }
    except Exception:
        return None


def _weather_code_to_text(code: int) -> str:
    mapping = {
        0: "☀️ Clear Sky",
        1: "🌤 Mainly Clear",
        2: "⛅ Partly Cloudy",
        3: "☁️ Overcast",
        45: "🌫 Foggy",
        48: "🌫 Rime Fog",
        51: "🌦 Light Drizzle",
        53: "🌦 Moderate Drizzle",
        55: "🌧 Dense Drizzle",
        61: "🌧 Slight Rain",
        63: "🌧 Moderate Rain",
        65: "🌧 Heavy Rain",
        71: "🌨 Slight Snowfall",
        73: "🌨 Moderate Snowfall",
        75: "❄️ Heavy Snowfall",
        80: "🌦 Rain Showers",
        81: "🌧 Moderate Showers",
        82: "⛈ Violent Showers",
        95: "⛈ Thunderstorm",
        96: "⛈ Thunderstorm with Hail",
        99: "⛈ Severe Thunderstorm",
    }
    return mapping.get(code, "🌡 Unknown")


def _travel_advice(condition: str, temp: float) -> str:
    """Generate a short travel tip based on weather conditions"""
    condition_lower = condition.lower()
    if "rain" in condition_lower or "drizzle" in condition_lower or "shower" in condition_lower:
        return "🌂 Carry an umbrella and waterproof footwear."
    if "thunder" in condition_lower:
        return "⚠️ Avoid outdoor activities — thunderstorms expected."
    if "snow" in condition_lower:
        return "🧥 Pack heavy winter clothing and snow boots."
    if "fog" in condition_lower:
        return "🚗 Drive carefully — low visibility due to fog."
    if temp is not None and temp >= 35:
        return "🥵 Very hot — stay hydrated and avoid midday sun."
    if temp is not None and temp <= 10:
        return "🧣 Cold weather — pack warm layers."
    return "✅ Good conditions for travel and outdoor activities."


def _format_forecast(dates: List[str], codes: List[int], max_temps: List[float], min_temps: List[float]) -> str:
    """Format a 7-day forecast into a readable string"""
    lines = ["\n📅 7-Day Forecast:"]
    for i in range(min(7, len(dates))):
        condition = _weather_code_to_text(codes[i])
        lines.append(f"  {dates[i]}  {condition}  {min_temps[i]}°C – {max_temps[i]}°C")
    return "\n".join(lines)


def get_weather(city: str) -> str:
    """
    Fetch real-time weather + 7-day forecast for a city.
    Uses Weatherstack if API key is set, otherwise falls back to Open-Meteo (free, no key needed).
    """
    weatherstack_key = os.getenv("WEATHERSTACK_API_KEY") or os.getenv("WEATHER_API_KEY")
    weather_provider = os.getenv("WEATHER_PROVIDER", "").lower()

    # --- Weatherstack (paid, more accurate) ---
    if weather_provider == "weatherstack" or weatherstack_key:
        try:
            url = "http://api.weatherstack.com/current"
            # Append ", India" to avoid Weatherstack mismatching Indian cities
            # e.g. "Goa" → "Genoa, Italy" without this fix
            query = f"{city}, India" if "india" not in city.lower() else city
            params = {"access_key": weatherstack_key, "query": query}
            r = requests.get(url, params=params, timeout=8)
            r.raise_for_status()
            data = r.json()
            if data.get("error"):
                return f"❌ Weather error for {city.title()}: {data['error'].get('info', 'Unknown error')}"
            loc = data.get("location", {})
            cur = data.get("current", {})
            name = loc.get("name") or city
            temp = cur.get("temperature")
            humidity = cur.get("humidity")
            feels_like = cur.get("feelslike")
            wind = cur.get("wind_speed")
            descs = cur.get("weather_descriptions") or []
            condition = descs[0] if descs else "Unknown"
            advice = _travel_advice(condition, temp)
            return (
                f"🌤 **Weather in {name}**\n\n"
                f"🌡 Temperature: {temp}°C (Feels like {feels_like}°C)\n"
                f"🌥 Condition: {condition}\n"
                f"💧 Humidity: {humidity}%\n"
                f"💨 Wind Speed: {wind} km/h\n\n"
                f"{advice}\n\n"
                f"_Source: Weatherstack_"
            )
        except requests.HTTPError as e:
            return f"❌ Weather service error for {city.title()}: HTTP {e.response.status_code}"
        except Exception as e:
            return f"❌ Could not fetch weather for {city.title()}: {str(e)}"

    # --- Open-Meteo (free, no API key needed) ---
    try:
        loc = _geocode_city(city)
        if not loc:
            return f"❌ Could not find location: **{city.title()}**. Please check the city name."

        lat, lon = loc["lat"], loc["lon"]
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min",
            "timezone": "auto",
            "forecast_days": 7,
        }
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()

        # Current weather
        current = data.get("current", {})
        temp = current.get("temperature_2m")
        feels_like = current.get("apparent_temperature")
        humidity = current.get("relative_humidity_2m")
        wind = current.get("wind_speed_10m")
        code = current.get("weather_code")
        condition = _weather_code_to_text(code) if code is not None else "Unknown"
        advice = _travel_advice(condition, temp)

        # 7-day forecast
        daily = data.get("daily", {})
        forecast_str = ""
        if daily:
            forecast_str = _format_forecast(
                dates=daily.get("time", []),
                codes=daily.get("weather_code", []),
                max_temps=daily.get("temperature_2m_max", []),
                min_temps=daily.get("temperature_2m_min", []),
            )

        return (
            f"🌤 **Current Weather in {city.title()}**\n\n"
            f"🌡 Temperature: {temp}°C (Feels like {feels_like}°C)\n"
            f"💧 Humidity: {humidity}%\n"
            f"🌥 Condition: {condition}\n"
            f"💨 Wind Speed: {wind} km/h\n\n"
            f"🧭 Travel Advice: {advice}\n"
            f"{forecast_str}\n\n"
            f"_Source: Open-Meteo (real-time)_"
        )

    except requests.HTTPError as e:
        return f"❌ Weather service error for {city.title()}: HTTP {e.response.status_code}"
    except requests.ConnectionError:
        return f"❌ Network error — could not reach weather service. Please check your connection."
    except Exception as e:
        return f"❌ Could not fetch weather for {city.title()}: {str(e)}"
