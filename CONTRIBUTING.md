Plugin conventions and contributing notes
=====================================

This file documents how to add plugins for the agent and the expected
plugin interface so contributors can implement handlers that will be
discovered and optionally executed by the agent.

Plugin location
- Place plugins in the `plugins/` directory in the repository root.
- Each plugin is a Python module (file) that exposes a `register()`
  function.

`register()` function
- Signature: `def register() -> dict`
- `register()` should return a mapping: `{ "action_name": async_handler, ... }`.
  - `action_name` must match the `name` field in `actions.json`.
  - `async_handler` must be an async function (e.g. `async def handler(args: dict) -> Any`).

Handler convention
- Handlers receive a single `args` dictionary with argument names and
  values from the validated tool call.
- Handlers should be `async` and may perform I/O (HTTP, MQTT, etc.).
- Return value: any serializable Python object (the agent will `print`
  the returned value when plugins are run). For production runners you
  may want to standardize on a dict like `{"status": "ok", "message": "..."}`.

Error handling
- Plugins should catch and handle errors where possible. If an
  exception bubbles to the agent, it will be caught and printed, and
  other plugins will not be affected.

Example plugin (plugins/console_plugin.py)
```py
def register():
    async def turn_on_light(args):
        room = args.get("room", "unknown")
        print(f"[console_plugin] ACTION: turn_on_light -> room={room}")
        return {"status": "printed", "message": f"turned on {room}"}

    return {"turn_on_light": turn_on_light}
```

Testing plugins locally
- Run the agent with `--run-plugins` and a mock LLM for safe testing:
  ```bash
  python assistant.py --mock --run-plugins
  ```
- The agent validates the model output against `actions.json` before
  calling the plugin handler. This keeps the plugin layer insulated
  from malformed model responses.

Contributing
- Fork the repo and submit a pull request with your plugin or action
  additions. Update `actions.json` for new actions and include tests
  where appropriate.
