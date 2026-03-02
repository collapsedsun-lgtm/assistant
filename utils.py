import json
import os
from typing import Any


def _extract_texts(obj: Any):
    out = []
    if obj is None:
        return out
    if isinstance(obj, str):
        out.append(obj)
        return out
    if isinstance(obj, dict):
        message = obj.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content:
                out.append(content)

        response = obj.get("response")
        if isinstance(response, str) and response:
            out.append(response)

        delta = obj.get("delta")
        if isinstance(delta, dict):
            dcontent = delta.get("content")
            if isinstance(dcontent, str) and dcontent:
                out.append(dcontent)

        text = obj.get("text")
        if isinstance(text, str) and text:
            out.append(text)

        content = obj.get("content")
        if isinstance(content, str) and content:
            out.append(content)

        if out:
            return out

        for v in obj.values():
            out.extend(_extract_texts(v))
        return out
    if isinstance(obj, list):
        for it in obj:
            out.extend(_extract_texts(it))
        return out
    return out


def _sanitize_assistant_output(text: str) -> str:
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    import re

    cleaned = re.sub(r"^\s*(assistant)\s*[:\-–—]\s*", "", text, flags=re.I)
    cleaned = re.sub(r"^\s*(assistant)\s+", "", cleaned, flags=re.I)
    return cleaned


def load_system_prompt() -> str:
    path = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_settings() -> dict:
    path = os.path.join(os.path.dirname(__file__), "settings.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
