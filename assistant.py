import argparse
import asyncio
import json
import os
from typing import List

import aiohttp
import inspect
import time
from assistant_core import build_messages, try_parse_tool_call

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


from utils import _extract_texts, _sanitize_assistant_output, load_system_prompt, load_settings

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


from assistant_core import build_messages, try_parse_tool_call








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

