"""Open-Meteo weather pre-send plugin.

This plugin performs geocoding via Open-Meteo's geocoding API and
fetches current weather and precipitation probability, returning a
sanitized string suitable for injection into the agent prompt.
"""
from typing import Optional
from urllib.parse import quote_plus
import aiohttp
import datetime


async def _geocode(city: str, debug: bool = False) -> Optional[dict]:
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={quote_plus(city)}&count=1"
    try:
        timeout = aiohttp.ClientTimeout(total=8)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.get(url) as resp:
                if resp.status != 200:
                    if debug:
                        print(f"[debug] geocode non-200 {resp.status} for {url}")
                    return None
                data = await resp.json()
    except Exception as e:
        if debug:
            import traceback

            print(f"[debug] geocode exception: {e!r}\n", traceback.format_exc())
        return None

    results = data.get("results") or []
    if not results:
        return None
    return results[0]


async def _fetch_open_meteo(lat: float, lon: float, debug: bool = False) -> Optional[dict]:
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&current_weather=true&hourly=precipitation_probability&timezone=UTC"
    )
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.get(url) as resp:
                if resp.status != 200:
                    if debug:
                        print(f"[debug] open-meteo non-200 {resp.status} for {url}")
                    return None
                data = await resp.json()
    except Exception as e:
        if debug:
            import traceback

            print(f"[debug] open-meteo exception: {e!r}\n", traceback.format_exc())
        return None

    return data


def _nearest_hour_index(times: list, debug: bool = False) -> Optional[int]:
    if not times:
        return None
    # Align to current UTC hour
    now = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    target = now.strftime("%Y-%m-%dT%H:00")
    try:
        return times.index(target)
    except ValueError:
        # fallback: try without :00 (some formats)
        target2 = now.strftime("%Y-%m-%dT%H:00Z")
        try:
            return times.index(target2)
        except ValueError:
            # last resort: nearest by scanning
            best = None
            best_diff = None
            for i, t in enumerate(times):
                try:
                    dt = datetime.datetime.fromisoformat(t)
                except Exception:
                    continue
                diff = abs((dt - now).total_seconds())
                if best is None or diff < best_diff:
                    best = i
                    best_diff = diff
            return best


async def _prefetch_open_meteo(user_input: str, history, debug: bool = False, settings: Optional[dict] = None) -> Optional[str]:
    low = (user_input or "").lower()
    if "weather" not in low:
        return None

    # extract city like previous plugin
    tokens = low.split()
    city = None
    if "in" in tokens:
        try:
            idx = tokens.index("in")
            candidate = tokens[idx + 1]
            candidate = candidate.strip(".,?;!")
            if candidate:
                city = candidate
        except Exception:
            city = None

    if not city and settings:
        city = settings.get("default_location")

    if not city:
        return None

    if debug:
        print(f"[debug] open-meteo plugin: geocoding {city}")

    geo = await _geocode(city, debug=debug)
    if not geo:
        return None

    lat = geo.get("latitude")
    lon = geo.get("longitude")
    name = geo.get("name") or city

    data = await _fetch_open_meteo(lat, lon, debug=debug)
    if not data:
        return None

    current = data.get("current_weather", {})
    temp = current.get("temperature")
    # map weathercode to simple word (very small mapping)
    wc = current.get("weathercode")
    code_map = {
        0: "clear",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "foggy",
        48: "depositing rime fog",
        51: "drizzle",
        61: "rain",
        71: "snow",
        80: "rain showers",
    }
    cond = code_map.get(wc, "")

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    pops = hourly.get("precipitation_probability", [])
    pop = None
    idx = _nearest_hour_index(times, debug=debug)
    if idx is not None and idx < len(pops):
        pop = pops[idx]

    # Build a sanitized short string
    parts = []
    parts.append(f"{name}: {cond}" if cond else f"{name}")
    if temp is not None:
        parts.append(f"{int(round(float(temp)))}°C")
    if pop is not None:
        parts.append(f"{int(round(float(pop)))}% chance of rain")

    summary = " ".join(parts)
    return f"Weather (sanitized): {summary}"


def register():
    return {"actions": {}, "pre_send": [_prefetch_open_meteo], "provider": "open_meteo"}
