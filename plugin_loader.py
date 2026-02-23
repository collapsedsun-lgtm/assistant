"""Small plugin discovery utility.

This module scans the `plugins/` directory for Python modules and
invokes their `register()` function (if present). `register()` should
return a mapping of `action_name` -> async handler function which will
be used by the main agent when `--run-plugins` is enabled.

Plugins are intentionally lightweight: they run inside the same
process and must be asynchronous (async def) so they can be awaited by
the agent.
"""

import importlib
import os
import pkgutil
from typing import Dict, Callable, Awaitable, Any, List, Tuple


# Directory where plugins live
PLUGIN_DIR = os.path.join(os.path.dirname(__file__), "plugins")


def load_plugins() -> Tuple[Dict[str, Callable[[dict], Awaitable[Any]]], List[Callable]]:
    """Discover and load plugin handlers and optional hooks.

    Returns a tuple `(handlers, pre_send_hooks)` where `handlers` maps
    action name -> async handler, and `pre_send_hooks` is a list of
    async callables that accept `(user_input, history)` and return a
    sanitized string (or None) to be injected into the prompt.

    Faulty plugins are ignored and do not prevent other plugins from loading.
    """
    handlers: Dict[str, Callable[[dict], Awaitable[Any]]] = {}
    pre_send_hooks: List[Callable] = []
    if not os.path.isdir(PLUGIN_DIR):
        return handlers, pre_send_hooks

    # Discover Python files in the plugins directory (load by path).
    for entry in os.listdir(PLUGIN_DIR):
        if not entry.endswith(".py"):
            continue
        if entry.startswith("__"):
            continue
        path = os.path.join(PLUGIN_DIR, entry)
        name = os.path.splitext(entry)[0]
        try:
            # Load module from file path to avoid requiring plugins/ to be a package
            spec = importlib.util.spec_from_file_location(f"plugins.{name}", path)
            if spec is None:
                continue
            module = importlib.util.module_from_spec(spec)
            loader = spec.loader
            if loader is None:
                continue
            loader.exec_module(module)
        except Exception:
            # Ignore modules that fail to import
            continue

        # Each plugin may implement a `register()` function. It may return
        # either a simple mapping {action: handler} (legacy style) or a
        # dict with optional keys: 'actions' -> mapping and 'pre_send' -> list
        # of async hook callables.
        register = getattr(module, "register", None)
        if callable(register):
            try:
                regs = register()
                if isinstance(regs, dict):
                    # New style: {'actions': {...}, 'pre_send': [callables]}
                    if "actions" in regs and isinstance(regs["actions"], dict):
                        handlers.update(regs["actions"])
                    # Legacy style: top-level mapping of actions
                    elif all(callable(v) for v in regs.values()):
                        handlers.update(regs)

                    # Optional pre_send hooks
                    if "pre_send" in regs and isinstance(regs["pre_send"], list):
                        for h in regs["pre_send"]:
                            if callable(h):
                                pre_send_hooks.append(h)
                # If register returned a tuple (compatibility), try to unpack
                elif isinstance(regs, tuple) and len(regs) == 2:
                    a, hooks = regs
                    if isinstance(a, dict):
                        handlers.update(a)
                    if isinstance(hooks, list):
                        for h in hooks:
                            if callable(h):
                                pre_send_hooks.append(h)
            except Exception:
                # ignore faulty plugins for now
                continue

    return handlers, pre_send_hooks
