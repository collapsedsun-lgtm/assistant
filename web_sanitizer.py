"""Safe web fetch and sanitizer utilities.

Plugins that need to fetch web data MUST use these helpers to ensure the
LLM never sees raw web content. The sanitizer strips HTML, removes URLs
and suspicious instruction-like fragments, and truncates long content.

The goal is conservative: drop anything that looks like programmatic or
instruction-bearing text and return a short neutral summary string.
"""
from typing import Optional
import re
import asyncio

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

import aiohttp


URL_RE = re.compile(r"https?://\S+", re.I)
BRACE_RE = re.compile(r"[{}<>]")
CODE_FENCE_RE = re.compile(r"```+")


def _sanitize_text(raw: str, max_chars: int = 800) -> str:
    if not raw:
        return ""

    text = raw

    # Remove common code fences and braces
    text = CODE_FENCE_RE.sub(" ", text)
    text = BRACE_RE.sub(" ", text)

    # Drop URLs
    text = URL_RE.sub(" ", text)

    # Collapse whitespace and normalize
    text = re.sub(r"\s+", " ", text).strip()

    # Remove lines that look like system instructions
    lines = []
    for ln in text.split("\n"):
        l = ln.strip()
        low = l.lower()
        if not l:
            continue
        if low.startswith("you are") or low.startswith("system:") or low.startswith("assistant:"):
            continue
        if any(tok in low for tok in ("do not", "should", "must", "avoid", "instruction")) and len(l) < 200:
            # conservative: skip short imperative lines
            continue
        lines.append(l)

    out = " ".join(lines)

    # Final truncate
    if len(out) > max_chars:
        out = out[: max_chars - 1].rsplit(" ", 1)[0] + "..."

    return out


async def fetch_and_sanitize(url: str, max_chars: int = 800, timeout: int = 10, debug: bool = False) -> Optional[str]:
    """Fetch a URL and return a sanitized short string or None on failure.

    This helper uses `aiohttp` and will never return raw HTML; HTML is
    stripped (via BeautifulSoup when available) and then further
    sanitized by heuristics above.
    """
    try:
        timeout_cfg = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_cfg) as sess:
            async with sess.get(url) as resp:
                if resp.status != 200:
                    if debug:
                        print(f"[debug] fetch_and_sanitize: non-200 status {resp.status} for {url}")
                    return None
                raw = await resp.text()
    except Exception as e:
        if debug:
            import traceback

            print(f"[debug] fetch_and_sanitize exception for {url}: {e!r}\n", traceback.format_exc())
        return None

    # If BeautifulSoup is available, extract visible text
    if BeautifulSoup:
        try:
            doc = BeautifulSoup(raw, "html.parser")
            for s in doc(["script", "style"]):
                s.decompose()
            visible = doc.get_text(separator=" ")
        except Exception:
            visible = raw
    else:
        # Fallback: very conservative strip of tags
        visible = re.sub(r"<[^>]+>", " ", raw)

    return _sanitize_text(visible, max_chars=max_chars)


def sanitize_snippet(snippet: str, max_chars: int = 800) -> str:
    """Sanitize an already-retrieved text snippet (no network calls).

    Always returns a short, neutral string suitable for injection into
    an LLM system message.
    """
    return _sanitize_text(snippet, max_chars=max_chars)


def humanize_weather(snippet: str) -> str:
    """Convert a terse weather snippet (from wttr.in style) into a
    human-friendly sentence suitable for TTS.

    Example input: "bristol: 🌫 +10°C" -> "In Bristol it's foggy and 10°C."
    The function is conservative and returns a readable fallback when
    parsing fails.
    """
    if not snippet:
        return "I don't have weather information right now."

    s = snippet.strip()

    # Remove leading 'Weather (sanitized):' if present
    if s.lower().startswith("weather (sanitized):"):
        s = s[len("weather (sanitized):") :].strip()

    # Try to split city and rest
    parts = s.split(":", 1)
    if len(parts) == 2:
        city, rest = parts[0].strip(), parts[1].strip()
    else:
        city, rest = None, parts[0].strip()

    # Map common weather emojis to words
    emoji_map = {
        "🌫": "foggy",
        "🌁": "foggy",
        "🌧": "rainy",
        "☔": "rainy",
        "☀": "sunny",
        "☀️": "sunny",
        "🌤": "partly sunny",
        "⛈": "thunderstorms",
        "🌩": "thunderstorms",
        "❄️": "snowy",
        "🌨": "snowy",
        "🌬": "windy",
        "💨": "windy",
    }

    # Find temperature (e.g., +10°C or 10 C)
    temp = None
    import re

    m = re.search(r"([+-]?\d{1,3})\s*°?\s*C", rest, re.I)
    if m:
        temp = m.group(1)
    else:
        # Try to find a bare number
        m2 = re.search(r"([+-]?\d{1,3})\b", rest)
        if m2:
            temp = m2.group(1)

    # Find first emoji present
    weather_word = None
    for ch in rest:
        if ch in emoji_map:
            weather_word = emoji_map[ch]
            break

    # If no emoji, try to find words like 'fog', 'rain', 'wind'
    low = rest.lower()
    if not weather_word:
        if "fog" in low or "mist" in low:
            weather_word = "foggy"
        elif "rain" in low or "showers" in low:
            weather_word = "rainy"
        elif "sun" in low or "clear" in low:
            weather_word = "sunny"
        elif "snow" in low:
            weather_word = "snowy"
        elif "wind" in low:
            weather_word = "windy"

    # Build human-friendly sentence
    parts = []
    if city:
        parts.append(f"In {city.capitalize()}")

    if weather_word:
        if city:
            parts.append(f"it's {weather_word}")
        else:
            parts.append(f"it's {weather_word}")

    if temp is not None:
        parts.append(f"and the temperature is {temp}°C")

    if not parts:
        # fallback to the raw rest text but cleaned
        return rest

    sentence = " ".join(parts).strip()
    # Ensure punctuation
    if not sentence.endswith("."):
        sentence = sentence + "."

    return sentence
