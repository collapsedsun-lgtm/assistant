import argparse
import asyncio
import json
import os
from typing import List

import aiohttp
import inspect
import time

from llm_config import MODEL, OLLAMA_URL, LLM_OPTIONS
from llm_config import LLM_STREAM, PARTIAL_TTL, FINAL_TTL, PARTIAL_SAVE_THRESHOLD, LLM_TIMEOUT
import actions
from plugin_loader import load_plugins
import memory_summarizer
import web_sanitizer
from kv_store import get_default_kv, response_cache_key
import session as session_module
from llm_client import call_llm, check_model_endpoint
from llm_config import LLM_STREAM

# Session mode flag: when True we send a short hint per request and rely on
# Ollama to retain the full system prompt in its session memory after a
# one-time bootstrap. When enabled via settings we assume session support.
OLLAMA_SESSION_ENABLED = False
OLLAMA_SESSION_BOOTSTRAPPED = False


def _extract_texts(obj):
    """Extract assistant text pieces from streaming JSON payloads.

    Prefer known content-bearing keys and avoid model metadata fields such as
    `model`, `created_at`, `role`, and stop reasons.
    """
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
    # other primitive (int/float/bool) -> ignore
    return out


def _sanitize_assistant_output(text: str) -> str:
    """Remove leading speaker labels that models sometimes emit (e.g. "Assistant:").

    This prevents duplicate printed labels like "Assistant: Assistant: ...".
    """
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    import re

    # Remove common leading speaker prefixes (case-insensitive), e.g.
    # "Assistant: ", "assistant -", "Assistant —".
    cleaned = re.sub(r"^\s*(assistant)\s*[:\-–—]\s*", "", text, flags=re.I)
    # Also remove a bare leading 'assistant' followed by whitespace/newline
    cleaned = re.sub(r"^\s*(assistant)\s+", "", cleaned, flags=re.I)
    return cleaned

# Number of most-recent exchanges (user + assistant) to include in context.
# This is a hardcoded constant for now; make configurable later if needed.
ROLLING_WINDOW = 5


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


def build_messages(
    system_prompt: str,
    actions_desc: str,
    history: List[dict],
    user_input: str,
    rolling_window: int,
    summarize_memory: bool = False,
    prefetch_texts: List[str] | None = None,
) -> List[dict]:
    # Compose system prompt. Avoid duplicating action instructions when the
    # base prompt already includes an "Available actions" section.
    if actions_desc and "available actions" not in system_prompt.lower():
        system_content = system_prompt + "\n\nAvailable actions:\n" + actions_desc
    else:
        system_content = system_prompt


    msgs = [{"role": "system", "content": system_content}]
    # Optionally include a memory summary as a system-level note
    if summarize_memory and history:
        summary = memory_summarizer.summarize(history, rolling_window)
        msgs.append({"role": "system", "content": "Memory summary: " + summary})

    # Optionally include sanitized pre-fetched facts (from pre_send hooks)
    if prefetch_texts:
        combined = "\n\n".join(t for t in prefetch_texts if t)
        if combined:
            msgs.append({"role": "system", "content": "Pre-fetched facts (sanitized):\n" + combined})

    # Include rolling history (already formatted as role/content dicts)
    start = max(0, len(history) - (rolling_window * 2))
    # Ensure message contents are strings when sent to the LLM
    for m in history[start:]:
        content = m.get("content")
        if isinstance(content, dict):
            try:
                content_str = json.dumps(content)
            except Exception:
                content_str = str(content)
        else:
            content_str = str(content)
        msgs.append({"role": m.get("role", "user"), "content": content_str})

    # Add current user input
    msgs.append({"role": "user", "content": user_input})
    return msgs








def try_parse_tool_call(llm_output: str, actions_list: List[actions.ActionSpec]):
    # Be permissive: the model may include extra text around the JSON
    # tool call (or prefixes like model names). Try direct parse first,
    # then search for a JSON object substring.
    try:
        parsed = json.loads(llm_output)
        return actions.validate_tool_call(parsed, actions_list)
    except Exception:
        pass

    # Attempt to find first {...} JSON object in the output
    start = llm_output.find("{")
    end = llm_output.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = llm_output[start : end + 1]
    try:
        parsed = json.loads(candidate)
        return actions.validate_tool_call(parsed, actions_list)
    except Exception:
        return None





# REPL/CLI moved to cli.py; keep this module focused on helper logic.

