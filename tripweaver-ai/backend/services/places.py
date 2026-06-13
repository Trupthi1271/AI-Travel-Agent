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
from services.web_search import search_web

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
    "varanasi": [
        {"name": "Kashi Vishwanath Temple", "category": "religion", "description": "One of India's most sacred Shiva temples and the spiritual heart of Varanasi."},
        {"name": "Dashashwamedh Ghat", "category": "religion", "description": "Famous riverside ghat known for the evening Ganga Aarti ceremony."},
        {"name": "Assi Ghat", "category": "religion", "description": "Popular ghat for sunrise views, boat rides, yoga, and relaxed riverside cafes."},
        {"name": "Manikarnika Ghat", "category": "religion", "description": "Historic cremation ghat that reflects Varanasi's deep spiritual traditions."},
        {"name": "Sarnath", "category": "historic", "description": "Important Buddhist site where Gautama Buddha gave his first sermon."},
        {"name": "Ramnagar Fort", "category": "historic", "description": "18th-century fort and museum on the eastern bank of the Ganga."},
        {"name": "Bharat Mata Mandir", "category": "cultural", "description": "Unique temple featuring a large marble relief map of undivided India."},
        {"name": "Banaras Hindu University", "category": "cultural", "description": "Large historic university campus with museums, gardens, and the Vishwanath Temple."},
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


def _enrich_with_search(city: str, places: List[Dict]) -> List[Dict]:
    """Add descriptions to places that don't have them using DuckDuckGo."""
    try:
        from duckduckgo_search import DDGS
        enriched = []
        for p in places:
            if p.get("description"):
                enriched.append(p)
                continue
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(
                        f"{p['name']} {city} tourist attraction India",
                        max_results=1
                    ))
                desc = results[0]["body"][:120] if results else "—"
                enriched.append({**p, "description": desc})
            except Exception:
                enriched.append({**p, "description": "—"})
        return enriched
    except Exception:
        return places


def _needs_description(place: Dict) -> bool:
    """Return True when a place has no useful description."""
    desc = (place.get("description") or "").strip()
    return not desc or desc in {"—", "-", "N/A"}


def _description_from_duckduckgo(place_name: str, city: str) -> Optional[str]:
    """
    Look up a short public-web description for a place.
    DuckDuckGo enrichment is best-effort so the places tool still works offline.
    """
    query = f"{place_name} {city} tourist attraction description"
    try:
        results = search_web(query, max_results=3)
    except Exception:
        return None

    for result in results:
        snippet = (result.get("snippet") or "").strip()
        if not snippet:
            continue
        if len(snippet) > 180:
            snippet = snippet[:177].rsplit(" ", 1)[0].rstrip(".,;:") + "..."
        return snippet
    return None


def _enrich_missing_descriptions(places: List[Dict], city: str, limit: int = 10) -> bool:
    """
    Fill missing place descriptions from DuckDuckGo snippets.
    Returns True if at least one place was enriched.
    """
    enriched = False
    checked = 0
    for place in places:
        if checked >= limit:
            break
        if not _needs_description(place):
            continue

        checked += 1
        desc = _description_from_duckduckgo(place["name"], city)
        if desc:
            place["description"] = desc
            enriched = True

    return enriched


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

    enriched_from_web = _enrich_missing_descriptions(places, city)

    # Group by raw category key (not label) for ordered display
    grouped: Dict[str, List[Dict]] = {}
    for p in places:
        cat = p["category"]
        grouped.setdefault(cat, []).append(p)

    lines = [f"## 🗺️ Top Places in {city.title()}\n"]

    # Category display names and order
    category_order = [
        ("historic",      "🏰 Historical Sites"),
        ("religion",      "🛕 Religious & Pilgrimage"),
        ("natural",       "🌿 Nature & Outdoors"),
        ("cultural",      "🏛️ Cultural"),
        ("architecture",  "🏗️ Architecture"),
        ("amusements",    "🎡 Entertainment"),
        ("shops",         "🛍️ Markets & Shopping"),
        ("foods",         "🍽️ Food & Drink"),
    ]

    for cat_key, cat_label in category_order:
        if cat_key not in grouped:
            continue
        items = grouped[cat_key]
        lines.append(f"### {cat_label}")
        lines.append("| Place | Description |")
        lines.append("|---|---|")
        for item in items:
            name = str(item.get("name", "")).strip().replace("|", "\\|")
            desc = str(item.get("description") or "—").strip().replace("|", "\\|")
            lines.append(f"| **{name}** | {desc} |")
        lines.append("")

    source = "curated data" if static else "OpenTripMap"
    if enriched_from_web:
        source += " + DuckDuckGo descriptions"
    lines.append(f"_Source: {source}_")
    return "\n".join(lines).strip()
