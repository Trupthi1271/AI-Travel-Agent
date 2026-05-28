"""
Flights Service
---------------
Searches for flights using the Amadeus Flights API (free test tier).
Falls back to a curated static schedule for popular Indian routes
when no API key is configured.

Free Amadeus test account: https://developers.amadeus.com/
Set AMADEUS_API_KEY and AMADEUS_API_SECRET in .env to enable live data.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests

from agent.error_handler import with_retry, ToolError

USER_AGENT = "TripWeaverAI/1.0 (contact: support@tripweaver.ai)"

# ── IATA code lookup for major Indian cities ──────────────────────────────────
CITY_TO_IATA: Dict[str, str] = {
    "delhi": "DEL", "new delhi": "DEL",
    "mumbai": "BOM", "bombay": "BOM",
    "bangalore": "BLR", "bengaluru": "BLR",
    "chennai": "MAA", "madras": "MAA",
    "kolkata": "CCU", "calcutta": "CCU",
    "hyderabad": "HYD",
    "goa": "GOI",
    "jaipur": "JAI",
    "ahmedabad": "AMD",
    "pune": "PNQ",
    "kochi": "COK", "cochin": "COK",
    "lucknow": "LKO",
    "varanasi": "VNS",
    "amritsar": "ATQ",
    "bhubaneswar": "BBI",
    "guwahati": "GAU",
    "patna": "PAT",
    "ranchi": "IXR",
    "srinagar": "SXR",
    "leh": "IXL",
    "jammu": "IXJ",
    "chandigarh": "IXC",
    "nagpur": "NAG",
    "indore": "IDR",
    "bhopal": "BHO",
    "raipur": "RPR",
    "visakhapatnam": "VTZ", "vizag": "VTZ",
    "coimbatore": "CJB",
    "madurai": "IXM",
    "tiruchirappalli": "TRZ", "trichy": "TRZ",
    "mangalore": "IXE",
    "calicut": "CCJ", "kozhikode": "CCJ",
    "port blair": "IXZ",
    "agartala": "IXA",
    "imphal": "IMF",
    "dibrugarh": "DIB",
    "silchar": "IXS",
    "bagdogra": "IXB",
    "dehradun": "DED",
    "shimla": "SLV",
    "kullu": "KUU", "manali": "KUU",
    "dharamsala": "DHM",
    "udaipur": "UDR",
    "jodhpur": "JDH",
    "aurangabad": "IXU",
}

# ── Static fallback: popular routes with typical fares ───────────────────────
_STATIC_ROUTES: Dict[Tuple[str, str], List[Dict]] = {
    ("DEL", "BOM"): [
        {"airline": "IndiGo", "flight": "6E-201", "dep": "06:00", "arr": "08:10", "duration": "2h 10m", "price": 3500, "class": "Economy"},
        {"airline": "Air India", "flight": "AI-101", "dep": "08:30", "arr": "10:45", "duration": "2h 15m", "price": 4200, "class": "Economy"},
        {"airline": "SpiceJet", "flight": "SG-101", "dep": "14:00", "arr": "16:15", "duration": "2h 15m", "price": 3200, "class": "Economy"},
    ],
    ("BOM", "DEL"): [
        {"airline": "IndiGo", "flight": "6E-202", "dep": "07:00", "arr": "09:15", "duration": "2h 15m", "price": 3600, "class": "Economy"},
        {"airline": "Air India", "flight": "AI-102", "dep": "10:00", "arr": "12:20", "duration": "2h 20m", "price": 4500, "class": "Economy"},
        {"airline": "Vistara", "flight": "UK-901", "dep": "18:00", "arr": "20:15", "duration": "2h 15m", "price": 5200, "class": "Economy"},
    ],
    ("DEL", "BLR"): [
        {"airline": "IndiGo", "flight": "6E-501", "dep": "06:30", "arr": "09:15", "duration": "2h 45m", "price": 4200, "class": "Economy"},
        {"airline": "Air India", "flight": "AI-501", "dep": "09:00", "arr": "11:50", "duration": "2h 50m", "price": 5100, "class": "Economy"},
    ],
    ("BOM", "GOI"): [
        {"airline": "IndiGo", "flight": "6E-301", "dep": "07:30", "arr": "08:45", "duration": "1h 15m", "price": 2800, "class": "Economy"},
        {"airline": "SpiceJet", "flight": "SG-301", "dep": "12:00", "arr": "13:20", "duration": "1h 20m", "price": 2500, "class": "Economy"},
    ],
    ("DEL", "GOI"): [
        {"airline": "IndiGo", "flight": "6E-401", "dep": "08:00", "arr": "10:30", "duration": "2h 30m", "price": 4800, "class": "Economy"},
        {"airline": "Air India", "flight": "AI-401", "dep": "14:00", "arr": "16:35", "duration": "2h 35m", "price": 5500, "class": "Economy"},
    ],
    ("DEL", "JAI"): [
        {"airline": "IndiGo", "flight": "6E-601", "dep": "07:00", "arr": "08:05", "duration": "1h 05m", "price": 2200, "class": "Economy"},
        {"airline": "SpiceJet", "flight": "SG-601", "dep": "15:00", "arr": "16:10", "duration": "1h 10m", "price": 1900, "class": "Economy"},
    ],
    ("BOM", "BLR"): [
        {"airline": "IndiGo", "flight": "6E-701", "dep": "06:00", "arr": "07:20", "duration": "1h 20m", "price": 3100, "class": "Economy"},
        {"airline": "Vistara", "flight": "UK-701", "dep": "10:30", "arr": "11:55", "duration": "1h 25m", "price": 4200, "class": "Economy"},
    ],
    ("DEL", "IXL"): [
        {"airline": "IndiGo", "flight": "6E-801", "dep": "06:00", "arr": "07:30", "duration": "1h 30m", "price": 5500, "class": "Economy"},
        {"airline": "Air India", "flight": "AI-801", "dep": "09:00", "arr": "10:35", "duration": "1h 35m", "price": 6200, "class": "Economy"},
    ],
}


def _resolve_iata(city: str) -> Optional[str]:
    """Convert a city name to IATA code."""
    return CITY_TO_IATA.get(city.lower().strip())


def _amadeus_base(env: str) -> str:
    return "https://api.amadeus.com" if (env or "").lower() == "prod" else "https://test.api.amadeus.com"


class _AmadeusTokenCache:
    """Singleton that caches the Amadeus OAuth token and auto-refreshes before expiry."""
    _token: Optional[str] = None
    _expires_at: float = 0.0

    @classmethod
    def get_token(cls, api_key: str, api_secret: str, env: str) -> Optional[str]:
        import time
        if cls._token and time.time() < cls._expires_at - 60:
            return cls._token
        try:
            url = f"{_amadeus_base(env)}/v1/security/oauth2/token"
            data = {"grant_type": "client_credentials", "client_id": api_key, "client_secret": api_secret}
            r = requests.post(url, data=data, timeout=8)
            r.raise_for_status()
            resp = r.json()
            cls._token = resp.get("access_token")
            cls._expires_at = time.time() + int(resp.get("expires_in", 1800))
            return cls._token
        except Exception:
            return None


def _get_amadeus_token(api_key: str, api_secret: str, env: str) -> Optional[str]:
    return _AmadeusTokenCache.get_token(api_key, api_secret, env)


def _search_amadeus_flights(
    origin: str, destination: str, date: str,
    adults: int, token: str, env: str
) -> List[Dict]:
    """Call Amadeus Flight Offers Search API."""
    url = f"{_amadeus_base(env)}/v2/shopping/flight-offers"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "originLocationCode": origin,
        "destinationLocationCode": destination,
        "departureDate": date,
        "adults": adults,
        "max": 5,
        "currencyCode": "INR",
    }
    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    results = []
    for offer in data.get("data", [])[:5]:
        try:
            itinerary = offer["itineraries"][0]
            segment = itinerary["segments"][0]
            price = float(offer["price"]["grandTotal"])
            dep_time = segment["departure"]["at"][11:16]
            arr_time = segment["arrival"]["at"][11:16]
            duration = itinerary["duration"].replace("PT", "").replace("H", "h ").replace("M", "m").strip()
            carrier = segment["carrierCode"]
            flight_num = f"{carrier}-{segment['number']}"
            results.append({
                "airline": carrier,
                "flight": flight_num,
                "dep": dep_time,
                "arr": arr_time,
                "duration": duration,
                "price": price,
                "class": offer.get("travelerPricings", [{}])[0]
                              .get("fareDetailsBySegment", [{}])[0]
                              .get("cabin", "Economy"),
            })
        except (KeyError, IndexError, ValueError):
            continue
    return results


def _static_fallback(origin_iata: str, dest_iata: str) -> List[Dict]:
    """Return static flight data for known routes."""
    key = (origin_iata, dest_iata)
    rev = (dest_iata, origin_iata)
    flights = _STATIC_ROUTES.get(key) or _STATIC_ROUTES.get(rev, [])
    # Reverse dep/arr for reverse routes
    if not _STATIC_ROUTES.get(key) and _STATIC_ROUTES.get(rev):
        flights = [
            {**f, "dep": f["arr"], "arr": f["dep"]}
            for f in flights
        ]
    return flights


def _format_flights(
    flights: List[Dict],
    origin: str, destination: str,
    travel_date: str, source: str
) -> str:
    if not flights:
        return (
            f"✈️ **Flights: {origin.title()} → {destination.title()}**\n\n"
            f"No flights found for this route. Consider checking MakeMyTrip, "
            f"Cleartrip, or Ixigo for the latest options."
        )

    lines = [
        f"✈️ **Flights: {origin.title()} → {destination.title()}**",
        f"📅 Date: {travel_date}\n",
    ]
    for i, f in enumerate(flights, 1):
        price_str = f"₹{f['price']:,.0f}" if isinstance(f['price'], (int, float)) else f"₹{f['price']}"
        lines.append(
            f"**{i}. {f['airline']} {f['flight']}**  "
            f"{f['dep']} → {f['arr']}  ({f['duration']})  "
            f"| {f['class']}  | **{price_str}**"
        )

    lines.append(f"\n_Source: {source} · Prices are indicative_")
    lines.append("💡 Book on MakeMyTrip, Cleartrip, or airline website for confirmed fares.")
    return "\n".join(lines)


@with_retry(max_attempts=2, delay=1.0)
def get_flights(origin: str, destination: str, travel_date: Optional[str] = None) -> str:
    """
    Search for flights between two Indian cities.
    Uses Amadeus API if credentials are set, otherwise static data.
    """
    origin_iata = _resolve_iata(origin)
    dest_iata = _resolve_iata(destination)

    if not origin_iata:
        raise ToolError(f"Could not find airport for '{origin}'. Try a major Indian city.")
    if not dest_iata:
        raise ToolError(f"Could not find airport for '{destination}'. Try a major Indian city.")
    if origin_iata == dest_iata:
        raise ToolError(f"Origin and destination are the same city.")

    # Default to tomorrow if no date given
    if not travel_date:
        travel_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        # Normalise date format
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%B %d %Y", "%d %B %Y"):
            try:
                travel_date = datetime.strptime(travel_date, fmt).strftime("%Y-%m-%d")
                break
            except ValueError:
                continue

    api_key = os.getenv("AMADEUS_API_KEY")
    api_secret = os.getenv("AMADEUS_API_SECRET")
    env = os.getenv("AMADEUS_ENV", "test")

    # Try Amadeus live API
    if api_key and api_secret:
        try:
            token = _get_amadeus_token(api_key, api_secret, env)
            if token:
                flights = _search_amadeus_flights(
                    origin_iata, dest_iata, travel_date, 1, token, env
                )
                if flights:
                    return _format_flights(flights, origin, destination, travel_date, "Amadeus")
        except Exception:
            pass  # Fall through to static data

    # Static fallback
    flights = _static_fallback(origin_iata, dest_iata)
    return _format_flights(flights, origin, destination, travel_date, "indicative schedule")
