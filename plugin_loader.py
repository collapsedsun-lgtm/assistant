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
from typing import Dict, Callable, Awaitable, Any


# Directory where plugins live
PLUGIN_DIR = os.path.join(os.path.dirname(__file__), "plugins")


def load_plugins() -> Dict[str, Callable[[dict], Awaitable[Any]]]:
    """Discover and load plugin handlers.

    Returns a dict mapping action name -> async handler. Faulty plugins
    are ignored and do not prevent other plugins from loading.
    """
    handlers = {}
    if not os.path.isdir(PLUGIN_DIR):
        return handlers

    # Discover modules in the plugins directory
    for finder, name, ispkg in pkgutil.iter_modules([PLUGIN_DIR]):
        module_name = f"plugins.{name}"
        try:
            module = importlib.import_module(module_name)
        except Exception:
            # Ignore modules that fail to import
            continue

        # Each plugin may implement a `register()` function that returns
        # a mapping of action_name -> async handler(args)
        register = getattr(module, "register", None)
        if callable(register):
            try:
                regs = register()
                if isinstance(regs, dict):
                    handlers.update(regs)
            except Exception:
                # ignore faulty plugins for now
                continue

    return handlers
