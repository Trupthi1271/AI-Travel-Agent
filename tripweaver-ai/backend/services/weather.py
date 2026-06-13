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


def _best_places_for_weather(city: str, condition: str) -> str:
    """Return 3 destination-specific places that suit the current weather."""
    city_key = city.lower().strip()
    rainy = any(word in condition.lower() for word in ["rain", "drizzle", "shower", "thunder", "mist"])

    places = {
        "goa": [
            ("Fort Aguada", "Historic fort with sea views; easier than beach time if rain starts."),
            ("Reis Magos Fort", "Covered heritage stop with scenic views over the Mandovi River."),
            ("Anjuna Market", "Good for shopping, cafes, and short outdoor walks between showers."),
        ] if rainy else [
            ("Palolem Beach", "Calmer beach for swimming, sunset walks, and relaxed cafes."),
            ("Baga Beach", "Best for water sports, beach shacks, and lively evening scenes."),
            ("Fort Aguada", "Open coastal views and an easy sightseeing stop in clear weather."),
        ],
        "manali": [
            ("Hadimba Temple", "Forest setting and short walks work well in cool mountain weather."),
            ("Old Manali", "Cafes, shops, and riverside lanes are easy to explore between showers."),
            ("Museum of Himachal Culture", "Indoor-friendly stop if rain or mist reduces visibility."),
        ],
        "jaipur": [
            ("City Palace", "A mix of indoor galleries and courtyards works well in hot or rainy weather."),
            ("Albert Hall Museum", "Indoor museum stop, useful during heat, rain, or harsh afternoon sun."),
            ("Hawa Mahal", "Quick iconic stop with nearby bazaars and easy photo opportunities."),
        ],
    }
    selected = places.get(city_key, [
        ("Main heritage area", "Good first stop for sightseeing based on current conditions."),
        ("Local market", "Flexible option for food, shopping, and short walks."),
        ("Museum or cultural center", "Useful indoor backup if weather changes."),
    ])

    lines = [
        "### Best Places Given This Weather",
        "",
        "| Place | Reason |",
        "|---|---|",
    ]
    for place, reason in selected[:3]:
        lines.append(f"| {place} | {reason} |")
    return "\n".join(lines)


def _pack_list(condition: str, temp: Optional[float]) -> str:
    """Return a compact pack list for the current weather."""
    condition_lower = condition.lower()
    if any(word in condition_lower for word in ["rain", "drizzle", "shower", "thunder", "mist"]):
        items = ["Umbrella or rain jacket", "Waterproof footwear", "Quick-dry clothing"]
    elif temp is not None and temp >= 35:
        items = ["Cap or hat", "Sunscreen", "Reusable water bottle"]
    elif temp is not None and temp <= 10:
        items = ["Warm jacket", "Thermal layer", "Comfortable closed shoes"]
    else:
        items = ["Light clothing", "Water bottle", "Sunscreen"]

    lines = ["### Pack", ""]
    lines.extend(f"- {item}" for item in items)
    return "\n".join(lines)


def get_weather(city: str) -> str:
    """
    Fetch real-time weather + 7-day forecast for a city.
    Uses Weatherstack if API key is set, otherwise falls back to Open-Meteo (free, no key needed).
    """
    weatherstack_key = os.getenv("WEATHERSTACK_API_KEY") or os.getenv("WEATHER_API_KEY")
    weather_provider = os.getenv("WEATHER_PROVIDER", "").lower()

    # --- Weatherstack (current conditions) + Open-Meteo (3-day forecast) ---
    if weather_provider == "weatherstack" or weatherstack_key:
        try:
            url = "http://api.weatherstack.com/current"
            query = f"{city}, India" if "india" not in city.lower() else city
            params = {"access_key": weatherstack_key, "query": query}
            r = requests.get(url, params=params, timeout=8)
            r.raise_for_status()
            data = r.json()
            if data.get("error"):
                return f"❌ Weather error for {city.title()}: {data['error'].get('info', 'Unknown error')}"
            loc_data = data.get("location", {})
            cur = data.get("current", {})
            name = loc_data.get("name") or city
            temp = cur.get("temperature")
            humidity = cur.get("humidity")
            feels_like = cur.get("feelslike")
            wind = cur.get("wind_speed")
            descs = cur.get("weather_descriptions") or []
            condition = descs[0] if descs else "Unknown"
            advice = _travel_advice(condition, temp)

            # Get 3-day forecast from Open-Meteo (free, always available)
            forecast_table = ""
            try:
                geo = _geocode_city(city)
                if geo:
                    fm_url = "https://api.open-meteo.com/v1/forecast"
                    fm_params = {
                        "latitude": geo["lat"], "longitude": geo["lon"],
                        "daily": "weather_code,temperature_2m_max,temperature_2m_min",
                        "timezone": "auto", "forecast_days": 3,
                    }
                    fm_r = requests.get(fm_url, params=fm_params, timeout=8)
                    fm_r.raise_for_status()
                    fm_data = fm_r.json()
                    daily = fm_data.get("daily", {})
                    if daily:
                        dates = daily.get("time", [])[:3]
                        codes = daily.get("weather_code", [])[:3]
                        highs = daily.get("temperature_2m_max", [])[:3]
                        lows  = daily.get("temperature_2m_min", [])[:3]
                        forecast_table = (
                            "\n\n**📅 3-Day Forecast:**\n\n"
                            "| Date | Condition | High | Low |\n"
                            "|---|---|---|---|\n"
                        )
                        for d, c_code, h, l in zip(dates, codes, highs, lows):
                            cond = _weather_code_to_text(c_code) if c_code is not None else "—"
                            forecast_table += f"| {d} | {cond} | {h}°C | {l}°C |\n"
            except Exception:
                forecast_table = ""

            return (
                f"## Weather in {name}\n\n"
                f"| Parameter | Value |\n"
                f"|---|---|\n"
                f"| Temperature | {temp}°C (Feels like {feels_like}°C) |\n"
                f"| Condition | {condition} |\n"
                f"| Humidity | {humidity}% |\n"
                f"| Wind Speed | {wind} km/h |\n"
                f"{forecast_table}\n\n"
                f"**Travel Advice:** {advice}\n\n"
                f"{_best_places_for_weather(name, condition)}\n\n"
                f"{_pack_list(condition, temp)}\n\n"
                f"_Source: Weatherstack + Open-Meteo_"
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

        # 3-day forecast table
        daily = data.get("daily", {})
        forecast_table = ""
        if daily:
            dates = daily.get("time", [])[:3]
            codes = daily.get("weather_code", [])[:3]
            highs = daily.get("temperature_2m_max", [])[:3]
            lows  = daily.get("temperature_2m_min", [])[:3]
            forecast_table = (
                "\n\n**📅 3-Day Forecast:**\n\n"
                "| Date | Condition | High | Low |\n"
                "|---|---|---|---|\n"
            )
            for d, c, h, l in zip(dates, codes, highs, lows):
                cond = _weather_code_to_text(c) if c is not None else "—"
                forecast_table += f"| {d} | {cond} | {h}°C | {l}°C |\n"

        return (
            f"## Weather in {city.title()}\n\n"
            f"| Parameter | Value |\n"
            f"|---|---|\n"
            f"| Temperature | {temp}°C (Feels like {feels_like}°C) |\n"
            f"| Condition | {condition} |\n"
            f"| Humidity | {humidity}% |\n"
            f"| Wind Speed | {wind} km/h |\n"
            f"{forecast_table}\n\n"
            f"**Travel Advice:** {advice}\n\n"
            f"{_best_places_for_weather(city, condition)}\n\n"
            f"{_pack_list(condition, temp)}\n\n"
            f"_Source: Open-Meteo (real-time)_"
        )

    except requests.HTTPError as e:
        return f"❌ Weather service error for {city.title()}: HTTP {e.response.status_code}"
    except requests.ConnectionError:
        return f"❌ Network error — could not reach weather service. Please check your connection."
    except Exception as e:
        return f"❌ Could not fetch weather for {city.title()}: {str(e)}"
