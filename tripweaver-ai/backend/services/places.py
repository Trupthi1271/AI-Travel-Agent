"""
Places of Interest Service
---------------------------
Uses the OpenTripMap API (free tier, no credit card required) to fetch
tourist attractions, landmarks, and points of interest for a destination.

API docs: https://dev.opentripmap.org/docs
Free tier: 5 req/sec, no daily cap.
Sign up at https://opentripmap.com/product to get a free API key.
Falls back to a curated static list when no API key is set.
"""

from __future__ import annotations

import os
import requests
from typing import List, Dict, Optional

from agent.error_handler import with_retry, ToolError

USER_AGENT = "TripWeaverAI/1.0 (contact: support@tripweaver.ai)"

# Category codes → human-readable labels
CATEGORY_LABELS: Dict[str, str] = {
    "cultural":       "🏛 Cultural",
    "natural":        "🌿 Nature",
    "historic":       "🏰 Historic",
    "architecture":   "🏗 Architecture",
    "religion":       "🛕 Religious",
    "amusements":     "🎡 Amusements",
    "sport":          "⚽ Sports",
    "foods":          "🍽 Food & Drink",
    "accommodation":  "🏨 Accommodation",
    "shops":          "🛍 Shopping",
    "transport":      "🚌 Transport",
}

# Static fallback data for popular Indian cities (used when no API key)
_STATIC_PLACES: Dict[str, List[Dict]] = {
    "goa": [
        {"name": "Baga Beach",              "category": "natural",      "description": "Popular beach known for water sports and nightlife."},
        {"name": "Palolem Beach",            "category": "natural",      "description": "Secluded crescent beach with calm waters, great for swimming."},
        {"name": "Anjuna Beach",             "category": "natural",      "description": "Famous Wednesday flea market and scenic rocky beach."},
        {"name": "Calangute Beach",          "category": "natural",      "description": "Largest beach in Goa, bustling with restaurants and shops."},
        {"name": "Vagator Beach",            "category": "natural",      "description": "Dramatic red cliffs and clear waters, quieter than Baga."},
        {"name": "Dudhsagar Falls",          "category": "natural",      "description": "One of India's tallest waterfalls, 310m high — stunning after rain."},
        {"name": "Basilica of Bom Jesus",    "category": "historic",     "description": "UNESCO World Heritage Site, 16th-century church with St. Francis Xavier's remains."},
        {"name": "Se Cathedral",             "category": "historic",     "description": "One of Asia's largest churches, built by the Portuguese in 1619."},
        {"name": "Fort Aguada",              "category": "historic",     "description": "17th-century Portuguese fort with lighthouse and Arabian Sea views."},
        {"name": "Chapora Fort",             "category": "historic",     "description": "Iconic hilltop fort with panoramic views — made famous by Dil Chahta Hai."},
        {"name": "Goa State Museum",         "category": "cultural",     "description": "Showcases Goa's history, art, and culture across multiple galleries."},
        {"name": "Anjuna Flea Market",       "category": "shops",        "description": "Famous Wednesday market for handicrafts, clothes, and souvenirs."},
        {"name": "Spice Plantation Tour",    "category": "natural",      "description": "Guided tour through tropical spice gardens with lunch included."},
        {"name": "Dolphin Watching Cruise",  "category": "amusements",   "description": "Boat tour to spot dolphins in the Arabian Sea — best in the morning."},
        {"name": "Mangeshi Temple",          "category": "religion",     "description": "Largest and most visited Hindu temple in Goa, dedicated to Lord Shiva."},
    ],
    "jaipur": [
        {"name": "Amber Fort", "category": "historic", "description": "Majestic hilltop fort with stunning architecture."},
        {"name": "Hawa Mahal", "category": "architecture", "description": "Palace of Winds — iconic 5-storey pink sandstone facade."},
        {"name": "City Palace", "category": "cultural", "description": "Royal palace complex with museums and courtyards."},
        {"name": "Jantar Mantar", "category": "historic", "description": "UNESCO-listed astronomical observatory, 18th century."},
        {"name": "Nahargarh Fort", "category": "historic", "description": "Hilltop fort with panoramic views of Jaipur."},
    ],
    "manali": [
        {"name": "Rohtang Pass", "category": "natural", "description": "High mountain pass at 3,978m, snow activities."},
        {"name": "Solang Valley", "category": "natural", "description": "Adventure hub — skiing, paragliding, zorbing."},
        {"name": "Hadimba Temple", "category": "religion", "description": "Ancient cave temple surrounded by cedar forest."},
        {"name": "Old Manali", "category": "cultural", "description": "Charming village with cafes, shops, and local culture."},
        {"name": "Beas River", "category": "natural", "description": "River rafting and scenic walks along the banks."},
    ],
    "delhi": [
        {"name": "Red Fort", "category": "historic", "description": "UNESCO-listed Mughal fort, symbol of India."},
        {"name": "Qutub Minar", "category": "historic", "description": "UNESCO-listed 73m minaret, 12th century."},
        {"name": "India Gate", "category": "historic", "description": "War memorial and popular evening gathering spot."},
        {"name": "Humayun's Tomb", "category": "historic", "description": "UNESCO-listed Mughal garden tomb."},
        {"name": "Chandni Chowk", "category": "cultural", "description": "Historic bazaar — street food, spices, textiles."},
    ],
    "mumbai": [
        {"name": "Gateway of India", "category": "historic", "description": "Iconic arch monument on the waterfront."},
        {"name": "Marine Drive", "category": "natural", "description": "3km seafront promenade, the 'Queen's Necklace'."},
        {"name": "Elephanta Caves", "category": "historic", "description": "UNESCO-listed rock-cut cave temples, 5th–8th century."},
        {"name": "Chhatrapati Shivaji Terminus", "category": "architecture", "description": "UNESCO-listed Victorian Gothic railway station."},
        {"name": "Juhu Beach", "category": "natural", "description": "Popular beach with street food stalls."},
    ],
    "kerala": [
        {"name": "Alleppey Backwaters", "category": "natural", "description": "Houseboat cruises through scenic canals and lagoons."},
        {"name": "Munnar Tea Gardens", "category": "natural", "description": "Rolling hills covered in tea plantations."},
        {"name": "Periyar Wildlife Sanctuary", "category": "natural", "description": "Tiger reserve with elephant safaris."},
        {"name": "Kovalam Beach", "category": "natural", "description": "Crescent-shaped beach popular for Ayurveda retreats."},
        {"name": "Padmanabhaswamy Temple", "category": "religion", "description": "Ancient Vishnu temple, one of India's wealthiest."},
    ],
}


def _geocode_city(city: str) -> Optional[Dict]:
    """Geocode a city to lat/lon using Nominatim."""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": f"{city}, India", "format": "json", "limit": 1}
        headers = {"User-Agent": USER_AGENT}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"])}
    except Exception:
        return None


def _fetch_opentripmap(lat: float, lon: float, api_key: str, radius: int = 10000) -> List[Dict]:
    """Fetch places from OpenTripMap API."""
    url = "https://api.opentripmap.com/0.1/en/places/radius"
    params = {
        "radius": radius,
        "lon": lon,
        "lat": lat,
        "kinds": "interesting_places",
        "rate": "3",          # only well-known places (rating 3+)
        "format": "json",
        "limit": 15,
        "apikey": api_key,
    }
    r = requests.get(url, params=params, timeout=8)
    r.raise_for_status()
    data = r.json()

    places = []
    for item in data:
        name = item.get("name", "").strip()
        if not name:
            continue
        kinds = item.get("kinds", "")
        # Map first kind to a label
        first_kind = kinds.split(",")[0] if kinds else "cultural"
        category = next(
            (k for k in CATEGORY_LABELS if k in first_kind), "cultural"
        )
        places.append({"name": name, "category": category, "description": ""})
    return places


def _static_fallback(city: str) -> List[Dict]:
    """Return static places for known cities."""
    return _STATIC_PLACES.get(city.lower(), [])


@with_retry(max_attempts=2, delay=1.0)
def get_places(city: str) -> str:
    """
    Get top tourist attractions and points of interest for a city.
    Uses static curated data for well-known cities (better quality),
    falls back to OpenTripMap API for other cities.
    """
    api_key = os.getenv("OPENTRIPMAP_API_KEY")

    # Always try static data first for known cities — it's curated and higher quality
    static = _static_fallback(city)
    if static:
        places = static
    elif api_key:
        # Use OpenTripMap only for cities not in static list
        try:
            loc = _geocode_city(city)
            if loc:
                places = _fetch_opentripmap(loc["lat"], loc["lon"], api_key)
            else:
                places = []
        except Exception:
            places = []
    else:
        places = []

    if not places:
        return (
            f"🗺️ **Places to Visit in {city.title()}**\n\n"
            f"No attraction data found for {city.title()}. "
            "Try a major Indian city like Goa, Jaipur, Manali, Delhi, Mumbai, or Kerala."
        )

    # Group by category
    grouped: Dict[str, List[str]] = {}
    for p in places:
        cat = p["category"]
        label = CATEGORY_LABELS.get(cat, "📍 Other")
        grouped.setdefault(label, []).append(
            f"• **{p['name']}**" + (f" — {p['description']}" if p.get("description") else "")
        )

    lines = [f"## 🗺️ Top Places in {city.title()}\n"]
    for label, items in grouped.items():
        lines.append(f"### {label}")
        lines.extend(items)
        lines.append("")

    source = "curated data" if static else "OpenTripMap"
    lines.append(f"_Source: {source}_")
    return "\n".join(lines).strip()
