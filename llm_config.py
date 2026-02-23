OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma:2b"

# Options used when calling the LLM. Kept simple and hardcoded for now.
LLM_OPTIONS = {
    "stream": False,
    "options": {
        "temperature": 0.1,
        "top_p": 0.9,
        "num_predict": 150
    }
}
