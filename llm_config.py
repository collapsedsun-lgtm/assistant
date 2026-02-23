"""Basic LLM configuration for the agent.

This file centralizes the model name, endpoint URL, and a small
options dictionary used for calls. Keeping these values separate from
the main agent code makes it easier to adjust model settings or swap
endpoints later.
"""

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma:2b"

# Options used when calling the LLM. Kept simple and hardcoded for now.
LLM_OPTIONS = {
    "stream": False,
    "options": {
        "temperature": 0.1,
        "top_p": 0.9,
        "num_predict": 150,
    },
}
