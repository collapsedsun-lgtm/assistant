"""Open-Meteo weather pre-send plugin.

This plugin performs geocoding via Open-Meteo's geocoding API and fetches:
- current weather
- today's min/max temperatures
- next 24h hourly temperature/rain probability
- tomorrow summary
- 7-day outlook summary

The plugin returns sanitized factual text for prompt injection.
The assistant can keep responses short by default while still having
rich data available for follow-up questions.
"""
from typing import Optional
from urllib.parse import quote_plus
import aiohttp
import datetime
import re


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


def _wcode_to_text(code: Optional[int]) -> str:
    code_map = {
        0: "clear",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "foggy",
        48: "rime fog",
        51: "light drizzle",
        53: "drizzle",
        55: "dense drizzle",
        61: "light rain",
        63: "rain",
        65: "heavy rain",
        71: "light snow",
        73: "snow",
        75: "heavy snow",
        80: "rain showers",
        81: "rain showers",
        82: "heavy rain showers",
        95: "thunderstorm",
    }
    return code_map.get(code, "")


def _is_weather_query(text: str) -> bool:
    low = (text or "").lower()
    weather_terms = (
        "weather",
        "forecast",
        "rain",
        "raining",
        "temperature",
        "temp",
        "tomorrow",
        "this week",
        "next week",
        "sunny",
    )
    return any(term in low for term in weather_terms)


def _extract_city(user_input: str) -> Optional[str]:
    low = (user_input or "").strip().lower()
    if not low:
        return None

    # Simple heuristics: "weather in bristol", "forecast for london"
    patterns = [r"\bin\s+([a-zA-Z\-\s']+)", r"\bfor\s+([a-zA-Z\-\s']+)"]
    for pat in patterns:
        m = re.search(pat, low)
        if m:
            cand = m.group(1).strip(" .,!?:;\"")
            # Trim common trailing words
            for stop in ("today", "tomorrow", "this week", "next week", "please"):
                cand = re.sub(rf"\b{re.escape(stop)}\b", "", cand).strip()
            if cand:
                return cand

    return None


async def _fetch_open_meteo(lat: float, lon: float, timezone: str, debug: bool = False) -> Optional[dict]:
    # Request current weather, daily min/max and hourly temperature + precipitation probability
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&current_weather=true"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode"
        f"&hourly=temperature_2m,precipitation_probability"
        f"&timezone={quote_plus(timezone)}"
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
    if not _is_weather_query(low):
        return None

    city = _extract_city(user_input)

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
    timezone = "UTC"
    if settings and settings.get("local_timezone"):
        timezone = settings.get("local_timezone")

    data = await _fetch_open_meteo(lat, lon, timezone=timezone, debug=debug)
    if not data:
        return None

    current = data.get("current_weather", {})
    temp = current.get("temperature")
    wc = current.get("weathercode")
    cond = _wcode_to_text(wc)

    # Daily min/max
    daily = data.get("daily", {})
    daily_dates = daily.get("time", [])
    temp_maxs = daily.get("temperature_2m_max", [])
    temp_mins = daily.get("temperature_2m_min", [])
    daily_rain_max = daily.get("precipitation_probability_max", [])
    daily_codes = daily.get("weathercode", [])
    max_temp = temp_maxs[0] if temp_maxs else None
    min_temp = temp_mins[0] if temp_mins else None

    # Hourly temps and precipitation probability
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    pops = hourly.get("precipitation_probability", [])

    # Find current hour index in hourly times
    idx = _nearest_hour_index(times, debug=debug)
    curr_pop = None
    if idx is not None and idx < len(pops):
        curr_pop = pops[idx]

    # Build short summary for default spoken answer
    summary_parts = []
    summary_parts.append(f"{name}: {cond}" if cond else f"{name}")
    if temp is not None:
        summary_parts.append(f"current {int(round(float(temp)))}°C")
    if max_temp is not None and min_temp is not None:
        summary_parts.append(f"max {int(round(float(max_temp)))}°C")
        summary_parts.append(f"min {int(round(float(min_temp)))}°C")
    if curr_pop is not None:
        summary_parts.append(f"{int(round(float(curr_pop)))}% chance of rain now")

    summary = ", ".join(summary_parts)

    # Build hourly table for the next 24 hours starting at current hour
    table_lines = []
    if idx is None:
        # fallback: start from beginning
        start = 0
    else:
        start = idx

    end = min(start + 24, len(times))
    table_lines.append(f"Hour ({timezone}) | Temp°C | Rain%")
    for i in range(start, end):
        t = times[i]
        # Extract hour component for readability
        hour_str = t
        try:
            # ISO format: YYYY-MM-DDTHH:MM[:SS][Z]
            hour_str = t.replace("T", " ")
        except Exception:
            pass
        tt = None
        pp = None
        try:
            if i < len(temps):
                tt = temps[i]
            if i < len(pops):
                pp = pops[i]
        except Exception:
            pass
        temp_display = f"{int(round(float(tt)))}" if tt is not None else "-"
        pop_display = f"{int(round(float(pp)))}" if pp is not None else "-"
        table_lines.append(f"{hour_str} | {temp_display} | {pop_display}")

    table = "\n".join(table_lines)

    # Tomorrow summary when available
    tomorrow_line = "tomorrow: unavailable"
    if len(daily_dates) > 1:
        dmax = temp_maxs[1] if len(temp_maxs) > 1 else None
        dmin = temp_mins[1] if len(temp_mins) > 1 else None
        drain = daily_rain_max[1] if len(daily_rain_max) > 1 else None
        dcode = daily_codes[1] if len(daily_codes) > 1 else None
        dcond = _wcode_to_text(dcode)
        tparts = [f"tomorrow ({daily_dates[1]})"]
        if dcond:
            tparts.append(dcond)
        if dmax is not None and dmin is not None:
            tparts.append(f"max {int(round(float(dmax)))}°C")
            tparts.append(f"min {int(round(float(dmin)))}°C")
        if drain is not None:
            tparts.append(f"rain up to {int(round(float(drain)))}%")
        tomorrow_line = ", ".join(tparts)

    # 7-day compact outlook
    week_lines = ["7-day outlook:"]
    week_count = min(7, len(daily_dates))
    for i in range(week_count):
        d = daily_dates[i]
        dmax = temp_maxs[i] if i < len(temp_maxs) else None
        dmin = temp_mins[i] if i < len(temp_mins) else None
        drain = daily_rain_max[i] if i < len(daily_rain_max) else None
        dcode = daily_codes[i] if i < len(daily_codes) else None
        dcond = _wcode_to_text(dcode)
        parts = [d]
        if dcond:
            parts.append(dcond)
        if dmax is not None and dmin is not None:
            parts.append(f"{int(round(float(dmin)))}-{int(round(float(dmax)))}°C")
        if drain is not None:
            parts.append(f"rain {int(round(float(drain)))}%")
        week_lines.append(" | ".join(parts))

    week_text = "\n".join(week_lines)

    return (
        "Weather (sanitized): " + summary + "\n"
        "Response guidance: reply short by default in 1 to 2 sentences. "
        "Use the detailed sections only when user asks for detail, tomorrow, hourly timing, or weekly outlook.\n"
        "Today details:\n"
        + summary
        + "\n"
        + tomorrow_line
        + "\n"
        + week_text
        + "\nHourly (next 24h):\n"
        + table
    )


def register():
    return {"actions": {}, "pre_send": [_prefetch_open_meteo], "provider": "open_meteo"}
