"""Example plugin that demonstrates a safe pre_send weather fetch.

This plugin registers a `pre_send` hook which, when the user asks
about weather, will fetch a public weather page and return a
sanitized short fact string. All raw HTML is sanitized by
`web_sanitizer.fetch_and_sanitize` before returning.

The agent will then inject the sanitized string into the prompt; the
LLM will never see the raw fetched content.
"""
import asyncio
import re
from typing import Optional
from urllib.parse import quote_plus

from web_sanitizer import fetch_and_sanitize


async def _prefetch_weather(user_input: str, history, debug: bool = False, settings: Optional[dict] = None) -> Optional[str]:
    low = (user_input or "").lower()
    if "weather" not in low:
        return None

    # Try to detect a city name naively (very simple heuristic)
    tokens = low.split()
    city = None
    if "in" in tokens:
        try:
            idx = tokens.index("in")
            # strip punctuation from the token (e.g., 'bristol?' -> 'bristol')
            candidate = tokens[idx + 1]
            candidate = re.sub(r"^[^\w]+|[^\w]+$", "", candidate)
            if candidate:
                city = candidate
        except Exception:
            city = None

    # If no city was parsed, try default from settings
    if not city and settings:
        try:
            default = settings.get("default_location")
            if default:
                city = default
        except Exception:
            pass

    # Use wttr.in (text) for a quick example; the sanitizer will strip
    # anything unsafe. In production you may prefer a structured API.
    urls_to_try = []
    if city:
        q = quote_plus(city)
        urls_to_try.append(f"https://wttr.in/{q}?format=3")
    # Fallback: general location
    urls_to_try.append("https://wttr.in/?format=3")

    for url in urls_to_try:
        if debug:
            print(f"[debug] weather_plugin: trying {url}")
        try:
            sanitized = await fetch_and_sanitize(url, max_chars=300, timeout=5, debug=debug)
            if sanitized:
                if debug:
                    print(f"[debug] weather_plugin: sanitized result: {sanitized}")
                return f"Weather (sanitized): {sanitized}"
        except Exception as e:
            if debug:
                import traceback

                print(f"[debug] weather_plugin exception for {url}: {e!r}\n", traceback.format_exc())
            # swallow; try next fallback
            continue

    return None


def register():
    # This plugin does not register action handlers, only a pre_send hook.
    return {"actions": {}, "pre_send": [_prefetch_weather]}
