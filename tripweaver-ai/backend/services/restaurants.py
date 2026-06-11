"""
Restaurant Service
------------------
Fetches restaurants, cafes, and local eateries for a destination.
Uses OpenTripMap API (foods category) with a curated static fallback.
Same mechanism as places.py but filtered to food & drink only.
"""

from __future__ import annotations

import os
import requests
from typing import List, Dict, Optional

from agent.error_handler import with_retry

USER_AGENT = "TripWeaverAI/1.0 (contact: support@tripweaver.ai)"

# Cuisine types → emoji labels
CUISINE_LABELS: Dict[str, str] = {
    "seafood":    "🦞 Seafood",
    "indian":     "🍛 Indian",
    "street":     "🌮 Street Food",
    "cafe":       "☕ Cafes",
    "fast_food":  "🍔 Fast Food",
    "vegetarian": "🥗 Vegetarian",
    "chinese":    "🥢 Chinese",
    "italian":    "🍕 Italian",
    "bar":        "🍺 Bars & Pubs",
    "other":      "🍽️ Restaurants",
}

# Curated static restaurant data for popular cities
_STATIC_RESTAURANTS: Dict[str, List[Dict]] = {
    "goa": [
        {"name": "Fisherman's Wharf",      "cuisine": "seafood",    "description": "Popular waterfront seafood restaurant, great for Goan fish curry."},
        {"name": "Thalassa",               "cuisine": "other",      "description": "Scenic Greek restaurant on a hilltop with ocean views."},
        {"name": "Britto's",               "cuisine": "seafood",    "description": "Iconic beachside restaurant in Baga, known for fresh seafood."},
        {"name": "Vinayak Family Restaurant","cuisine": "indian",   "description": "Authentic Goan thali — must-try for local flavors."},
        {"name": "Cafe Bodega",            "cuisine": "cafe",       "description": "Charming heritage cafe in Fontainhas with fresh bakes and coffee."},
        {"name": "Infantaria Pastry Shop", "cuisine": "cafe",       "description": "Best croissants and pastries in Goa, Calangute landmark."},
        {"name": "Anjuna Flea Market Food Stalls", "cuisine": "street", "description": "Wednesday market with diverse street food from across India."},
    ],
    "jaipur": [
        {"name": "Laxmi Misthan Bhandar",  "cuisine": "indian",     "description": "Jaipur institution — best kachori, samosas, and sweets since 1954."},
        {"name": "Chokhi Dhani",           "cuisine": "indian",     "description": "Rajasthani village resort with traditional thali and folk performances."},
        {"name": "Niros Restaurant",       "cuisine": "other",      "description": "Oldest restaurant on MI Road, famous for Rajasthani cuisine."},
        {"name": "Tapri Central",          "cuisine": "cafe",       "description": "Trendy rooftop cafe with chai, coffee, and city views."},
        {"name": "Masala Chowk",           "cuisine": "street",     "description": "Popular food court with 25+ stalls serving Rajasthani street food."},
        {"name": "Hotel Rawat",            "cuisine": "street",     "description": "Famous for pyaaz kachori — a Jaipur breakfast staple."},
    ],
    "manali": [
        {"name": "Johnson's Cafe",         "cuisine": "other",      "description": "Cozy cafe with excellent trout, Himachali cuisine, and mountain views."},
        {"name": "Drifters' Inn & Cafe",   "cuisine": "cafe",       "description": "Backpacker favorite in Old Manali with momos, thukpa, and chai."},
        {"name": "Café 1947",              "cuisine": "cafe",       "description": "Popular hangout for trekkers — great pasta, pancakes, and lassi."},
        {"name": "Lazy Dog Lounge",        "cuisine": "bar",        "description": "Best bar in Manali with live music and Himalayan cocktails."},
        {"name": "Manali Momo Corner",     "cuisine": "street",     "description": "Local street stall with the best steamed and fried momos."},
    ],
    "delhi": [
        {"name": "Karim's",                "cuisine": "indian",     "description": "Legendary Mughlai restaurant near Jama Masjid, since 1913."},
        {"name": "Indian Accent",          "cuisine": "other",      "description": "Award-winning modern Indian cuisine, one of Asia's best restaurants."},
        {"name": "Paranthe Wali Gali",     "cuisine": "street",     "description": "Historic lane in Chandni Chowk famous for stuffed parathas since 1875."},
        {"name": "Bukhara",                "cuisine": "indian",     "description": "ITC Maurya's iconic restaurant, famous for dal bukhara and tandoori."},
        {"name": "Saravana Bhavan",        "cuisine": "vegetarian", "description": "South Indian vegetarian chain — best idli, dosa, and filter coffee."},
    ],
    "mumbai": [
        {"name": "Leopold Cafe",           "cuisine": "other",      "description": "Historic Colaba cafe and bar, Mumbai landmark since 1871."},
        {"name": "Trishna",                "cuisine": "seafood",    "description": "Legendary seafood restaurant in Fort, famous for butter garlic crab."},
        {"name": "Vada Pav at Ashok",      "cuisine": "street",     "description": "Mumbai's most iconic street food — best vada pav in the city."},
        {"name": "The Table",              "cuisine": "other",      "description": "Top-rated contemporary restaurant in Colaba with global menu."},
        {"name": "Cafe Mondegar",          "cuisine": "bar",        "description": "Famous Colaba bar with jukebox, murals, and cold beers."},
    ],
    "kerala": [
        {"name": "Alle Spice",             "cuisine": "seafood",    "description": "Alleppey's best for Kerala fish curry, karimeen, and prawn dishes."},
        {"name": "Paragon Restaurant",     "cuisine": "seafood",    "description": "Kozhikode institution — best Malabar biryani and seafood in Kerala."},
        {"name": "Dhe Puttu",              "cuisine": "indian",     "description": "Innovative Kerala restaurant known for creative puttu variations."},
        {"name": "Hotel Saravana Bhavan",  "cuisine": "vegetarian", "description": "South Indian vegetarian — excellent Kerala sadya on Sundays."},
        {"name": "Tea Garden Cafe",        "cuisine": "cafe",       "description": "Munnar hillside cafe with fresh tea, local snacks, and valley views."},
    ],
    "varanasi": [
        {"name": "Kashi Chat Bhandar",     "cuisine": "street",     "description": "Famous for tamatar chaat and Banarasi street food, Godaulia area."},
        {"name": "Aadha-Aadha",           "cuisine": "cafe",       "description": "Popular rooftop cafe near the ghats with Ganga views and lassi."},
        {"name": "Keshari Restaurant",     "cuisine": "vegetarian", "description": "Best thali in Varanasi — pure vegetarian Banarasi food."},
        {"name": "Blue Lassi Shop",        "cuisine": "street",     "description": "Iconic 70-year-old shop serving the finest lassi in India."},
    ],
}


def _geocode_city(city: str) -> Optional[Dict]:
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": f"{city}, India", "format": "json", "limit": 1}
        headers = {"User-Agent": USER_AGENT}
        r = requests.get(url, params=params, headers=headers, timeout=8)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"])}
    except Exception:
        return None


def _fetch_opentripmap_restaurants(lat: float, lon: float, api_key: str) -> List[Dict]:
    """Fetch food & drink places from OpenTripMap."""
    url = "https://api.opentripmap.com/0.1/en/places/radius"
    params = {
        "radius": 5000,
        "lon": lon,
        "lat": lat,
        "kinds": "foods",
        "rate": "2",
        "format": "json",
        "limit": 15,
        "apikey": api_key,
    }
    r = requests.get(url, params=params, timeout=8)
    r.raise_for_status()
    data = r.json()
    results = []
    for item in data:
        name = item.get("name", "").strip()
        if not name:
            continue
        results.append({"name": name, "cuisine": "other", "description": ""})
    return results


@with_retry(max_attempts=2, delay=1.0)
def get_restaurants(city: str) -> str:
    """
    Get top restaurants and food spots for a city.
    Uses curated static data first, falls back to OpenTripMap.
    """
    api_key = os.getenv("OPENTRIPMAP_API_KEY")

    static = _STATIC_RESTAURANTS.get(city.lower(), [])
    if static:
        restaurants = static
        source = "curated data"
    elif api_key:
        try:
            loc = _geocode_city(city)
            if loc:
                restaurants = _fetch_opentripmap_restaurants(loc["lat"], loc["lon"], api_key)
                source = "OpenTripMap"
            else:
                restaurants = []
                source = "unavailable"
        except Exception:
            restaurants = []
            source = "unavailable"
    else:
        restaurants = []
        source = "unavailable"

    if not restaurants:
        return (
            f"## 🍽️ Restaurants in {city.title()}\n\n"
            f"No restaurant data available for {city.title()}. "
            f"Try searching on Zomato or Google Maps for live recommendations."
        )

    # Group by cuisine
    grouped: Dict[str, List[str]] = {}
    for r in restaurants:
        cuisine = r.get("cuisine", "other")
        label = CUISINE_LABELS.get(cuisine, "🍽️ Restaurants")
        entry = f"• **{r['name']}**"
        if r.get("description"):
            entry += f" — {r['description']}"
        grouped.setdefault(label, []).append(entry)

    lines = [f"## 🍽️ Where to Eat in {city.title()}\n"]
    for label, items in grouped.items():
        lines.append(f"### {label}")
        lines.extend(items)
        lines.append("")

    lines.append(f"_Source: {source}_")
    lines.append("\n💡 **Tip:** Always check current ratings on Zomato or Google Maps before visiting.")
    return "\n".join(lines).strip()
