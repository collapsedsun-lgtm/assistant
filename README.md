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

Recent changes (2026-03-01)
- **Improve latency by using KV store**: a new `kv_store.py` and related changes reduce repeated computation and speed up repeated queries.
- **Session / bootstrap behavior**: the assistant now attempts a one-time bootstrap with the model when the backend supports server-side sessions; note that `/api/chat` endpoints do not provide persistent session semantics, so session-mode is disabled for that endpoint and the assistant falls back to the per-request prompt flow.
- **Keep-alive and health-checks**: requests include a `keep_alive` option (configurable via `settings.json`) to reduce cold-starts; health checks were changed to connectivity-only checks to avoid contaminating session state.
- **Streaming parsing fix**: streaming output parsing was tightened to avoid model metadata leaking into assistant text (you should see cleaner streaming outputs now).
- **Files changed**: `assistant.py`, `kv_store.py`, `llm_config.py`, `plugin_loader.py`, plugins under `plugins/`, `system_prompt.txt`, and `settings.json` were touched in the recent PR.

Where to look next
- `assistant.py`: main changes to session handling, bootstrap ordering, and streaming parsing.
- `settings.json`: new/updated keys control `ollama_use_session` and `ollama_keep_alive`.
- `kv_store.py`: caching/kv logic used to improve latency on repeated operations.


CLI Flags
- `--mock`: Run with a simple local mock LLM (useful for testing without a model endpoint).
- `--run-plugins`: Allow executing discovered plugins for validated actions (optional; disabled by default).
- `--debug`: Enable debug output; prints the full LLM payload and tracebacks on error.
- `--rolling-window N`: Include up to N recent exchanges (user+assistant pairs) in the context sent to the model. Default is 5.
- `--summarize-memory`: Enable a local memory summarizer that condenses recent exchanges into a short summary included in the prompt (disabled by default).

Examples
Run with mock LLM and plugin execution (safe for debugging):
```bash
python assistant.py --mock --run-plugins
```

Run with the real LLM, 3-exchange context and memory summarization enabled:
```bash
python assistant.py --rolling-window 3 --summarize-memory --run-plugins --debug
```

Plugin example
--------------
Create a file `plugins/console_plugin.py` with the following contents:

```py
def register():
	async def turn_on_light(args: dict):
		room = args.get("room", "unknown")
		print(f"[console_plugin] ACTION: turn_on_light -> room={room}")
		return {"status": "printed", "message": f"turned on {room}"}

	return {"turn_on_light": turn_on_light}
```

Run the agent (mock or real) and enable plugin execution:

```bash
python assistant.py --run-plugins --mock
```

The agent will validate model outputs against `actions.json` and call
the matching plugin handler when `--run-plugins` is provided.

Commercial licensing
--------------------
This project is licensed under the GNU Affero General Public License v3
(AGPL-3.0). If you wish to use this code in a commercial product but do
not want to comply with the AGPL terms (for example, to avoid releasing
source for a hosted service), commercial licensing may be available on
request. Contact the repository owner for details.
