import requests
from typing import Dict, List, Optional

USER_AGENT = "TripWeaverAI/1.0 (contact: support@tripweaver.ai)"

def _geocode_city(city: str) -> Optional[Dict]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": city, "format": "json", "limit": 1}
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, params=params, headers=headers, timeout=8)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    bbox = [float(x) for x in data[0]["boundingbox"]]  # [south, north, west, east]
    return {"display_name": data[0]["display_name"], "bbox": bbox}

def _query_overpass_hotels(bbox: List[float]) -> List[Dict]:
    south, north, west, east = bbox
    query = f"""
    [out:json][timeout:10];
    (node["tourism"~"hotel|guest_house|hostel"]({south},{west},{north},{east});
     way["tourism"~"hotel|guest_house|hostel"]({south},{west},{north},{east});
     relation["tourism"~"hotel|guest_house|hostel"]({south},{west},{north},{east});
    );
    out center 20;
    """
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    headers = {"User-Agent": USER_AGENT}
    for url in endpoints:
        try:
            r = requests.post(url, data=query, headers=headers, timeout=12)
            r.raise_for_status()
            data = r.json()
            break
        except Exception:
            data = {}
            continue
    elements = data.get("elements", [])
    results = []
    seen = set()
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        typ = tags.get("tourism")
        lat = el.get("lat") or (el.get("center", {}) or {}).get("lat")
        lon = el.get("lon") or (el.get("center", {}) or {}).get("lon")
        results.append({"name": name, "type": typ, "lat": lat, "lon": lon})
        if len(results) >= 20:
            break
    return results

def _amadeus_base(env: str) -> str:
    return "https://api.amadeus.com" if (env or "").lower() == "prod" else "https://test.api.amadeus.com"

def _amadeus_token(api_key: str, api_secret: str, env: str) -> Optional[str]:
    if not (api_key and api_secret):
        return None
    url = f"{_amadeus_base(env)}/v1/security/oauth2/token"
    data = {"grant_type": "client_credentials", "client_id": api_key, "client_secret": api_secret}
    r = requests.post(url, data=data, timeout=20)
    r.raise_for_status()
    return r.json().get("access_token")

def _amadeus_hotels_by_geocode(lat: float, lon: float, token: str, env: str) -> List[Dict]:
    url = f"{_amadeus_base(env)}/v1/reference-data/locations/hotels/by-geocode"
    headers = {"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT}
    params = {"latitude": lat, "longitude": lon, "radius": 20, "radiusUnit": "KM"}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    items = data.get("data", []) or []
    out = []
    for it in items:
        name = it.get("name")
        if not name:
            continue
        out.append({"name": name, "type": "hotel"})
        if len(out) >= 18:
            break
    return out

def get_hotels(city: str) -> str:
    # Read env at call time to ensure .env has been loaded
    import os
    hotels_provider = os.getenv("HOTELS_PROVIDER")
    amadeus_api_key = os.getenv("AMADEUS_API_KEY")
    amadeus_api_secret = os.getenv("AMADEUS_API_SECRET")
    amadeus_env = os.getenv("AMADEUS_ENV", "test")

    if (hotels_provider and hotels_provider.lower() == "amadeus") or (amadeus_api_key and amadeus_api_secret):
        try:
            loc = _geocode_city(city)
            if not loc:
                return f"🏨 Hotels in {city.title()}\n\nCould not locate this city."
            token = _amadeus_token(amadeus_api_key, amadeus_api_secret, amadeus_env)
            if not token:
                return f"❌ Hotel service error for {city.title()}: Missing Amadeus credentials"
            bbox = loc["bbox"]
            lat = (bbox[0] + bbox[1]) / 2.0
            lon = (bbox[2] + bbox[3]) / 2.0
            hotels = _amadeus_hotels_by_geocode(lat, lon, token, amadeus_env)
            if not hotels:
                return f"🏨 Hotels in {city.title()} (Amadeus)\n\nNo hotels found near this area."
            lines = [f"🏨 Hotels in {city.title()} (Amadeus)"]
            lines.append("\nHotels")
            for h in hotels[:10]:
                lines.append(f"• {h['name']}")
            return "\n".join(lines)
        except requests.HTTPError as e:
            return f"❌ Hotel service error for {city.title()}: {e.response.status_code}"
        except Exception as e:
            return f"❌ Could not fetch hotels for {city.title()}: {str(e)}"
    try:
        loc = _geocode_city(city)
        if not loc:
            return f"🏨 Hotels in {city.title()}\n\nCould not locate this city."
        hotels = _query_overpass_hotels(loc["bbox"])
        if not hotels:
            return f"🏨 Hotels in {city.title()}\n\nNo hotels found via OpenStreetMap for this area."
        # Simple grouping
        hostels = [h for h in hotels if (h.get("type") or "").lower() == "hostel"]
        guests = [h for h in hotels if (h.get("type") or "").lower() == "guest_house"]
        normal = [h for h in hotels if (h.get("type") or "").lower() == "hotel"]
        result = [f"🏨 Hotels in {city.title()} (data: OpenStreetMap)"]
        if normal:
            result.append("\nHotels")
            for h in normal[:6]:
                result.append(f"• {h['name']}")
        if guests:
            result.append("\nGuest Houses")
            for h in guests[:6]:
                result.append(f"• {h['name']}")
        if hostels:
            result.append("\nHostels")
            for h in hostels[:6]:
                result.append(f"• {h['name']}")
        return "\n".join(result)
    except requests.HTTPError as e:
        return f"❌ Hotel service error for {city.title()}: {e.response.status_code}"
    except Exception as e:
        return f"❌ Could not fetch hotels for {city.title()}: {str(e)} (Overpass may be busy; try again in a minute)"
