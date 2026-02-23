# Assistant (home automation agent)

Minimal agent for generating JSON-encoded actions from an LLM. The agent is intentionally separated from executors/runners: it will emit actions as JSON only and a separate runner/plugin can consume those actions to interact with devices.

Quickstart
1. Clone the repo.
2. Create a per-project Python virtual environment and install dependencies (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run (mock LLM for safe testing):

```bash
python assistant.py --mock
```

Run (real LLM - requires Ollama or configured endpoint):

```bash
python assistant.py
```

Project layout
- `assistant.py` — async REPL that talks to the LLM and emits JSON actions.
- `actions.json` — single source of truth describing available actions and their argument shapes.
- `actions.py` — helpers and pydantic validation for action specs and tool calls.
- `system_prompt.txt` — system prompt provided to the model.
- `llm_config.py` — LLM endpoint and options (hardcoded for now).
- `plugin_loader.py` — discovers plugins in `plugins/`.
- `plugins/` — example plugins live here; plugins should implement a `register()` returning a mapping of action_name -> async handler.

Contributing
- Add new actions by editing `actions.json` (this is the single source of truth for both the LLM and runners).
- Implement a plugin in `plugins/` and expose handlers via `register()`.
- Keep secrets out of the repo (use environment variables or a local `.env` file listed in `.gitignore`).

Notes
- The agent uses a rolling context window (last few exchanges) to provide short-term memory.
- The agent will never execute actions itself; it prints validated action JSON. Build a separate runner to consume and safely execute those actions.
