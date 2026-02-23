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
