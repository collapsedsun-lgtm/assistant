"""Basic LLM configuration for the agent.

This file centralizes the model name, endpoint URL, and a small
options dictionary used for calls. Keeping these values separate from
the main agent code makes it easier to adjust model settings or swap
endpoints later.
"""

import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.getenv("MODEL", "gemma:2b")

# Options used when calling the LLM. Kept simple and hardcoded for now.
# `LLM_STREAM` environment variable or `settings.json` can enable streaming.
LLM_OPTIONS = {
    "stream": False,
    "options": {
        "temperature": 0.1,
        "top_p": 0.9,
        "num_predict": 150,
    },
}

# Runtime toggle to enable streaming mode independent of LLM_OPTIONS.
LLM_STREAM = os.getenv("LLM_STREAM", "false").lower() in ("1", "true", "yes")

# Partial and final response cache TTLs (seconds) and partial-save threshold.
# Configure via environment variables: LLM_PARTIAL_TTL, LLM_FINAL_TTL,
# LLM_PARTIAL_SAVE_THRESHOLD.
PARTIAL_TTL = int(os.getenv("LLM_PARTIAL_TTL", "30"))
FINAL_TTL = int(os.getenv("LLM_FINAL_TTL", "60"))
PARTIAL_SAVE_THRESHOLD = int(os.getenv("LLM_PARTIAL_SAVE_THRESHOLD", "200"))

# Request timeout (seconds) for LLM calls. Increase when running on CPU-backed models.
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))


# Override from settings.json when present. This allows persisted configuration
# in addition to environment variables. `settings.json` keys (if present):
# `llm_stream`, `partial_ttl`, `final_ttl`, `partial_save_threshold`.
try:
    import json
    here = os.path.dirname(__file__)
    settings_path = os.path.join(here, "settings.json")
    if os.path.exists(settings_path):
        with open(settings_path, "r", encoding="utf-8") as f:
            _settings = json.load(f)
        if isinstance(_settings, dict):
            if "llm_stream" in _settings:
                try:
                    LLM_STREAM = bool(_settings.get("llm_stream"))
                except Exception:
                    pass
            if "partial_ttl" in _settings:
                try:
                    PARTIAL_TTL = int(_settings.get("partial_ttl"))
                except Exception:
                    pass
            if "final_ttl" in _settings:
                try:
                    FINAL_TTL = int(_settings.get("final_ttl"))
                except Exception:
                    pass
            if "partial_save_threshold" in _settings:
                try:
                    PARTIAL_SAVE_THRESHOLD = int(_settings.get("partial_save_threshold"))
                except Exception:
                    pass
            if "llm_timeout" in _settings:
                try:
                    LLM_TIMEOUT = int(_settings.get("llm_timeout"))
                except Exception:
                    pass
except Exception:
    # If settings can't be read or parsed, silently continue using env/defaults
    pass

