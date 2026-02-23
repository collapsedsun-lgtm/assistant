import importlib
import os
import pkgutil
from typing import Dict, Callable, Awaitable, Any

PLUGIN_DIR = os.path.join(os.path.dirname(__file__), "plugins")


def load_plugins() -> Dict[str, Callable[[dict], Awaitable[Any]]]:
    handlers = {}
    if not os.path.isdir(PLUGIN_DIR):
        return handlers

    # Discover modules in the plugins directory
    for finder, name, ispkg in pkgutil.iter_modules([PLUGIN_DIR]):
        module_name = f"plugins.{name}"
        try:
            module = importlib.import_module(module_name)
        except Exception:
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
