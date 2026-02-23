import json
import os
from typing import Dict, List

_ACTIONS_PATH = os.path.join(os.path.dirname(__file__), "actions.json")


def load_actions() -> List[Dict]:
    with open(_ACTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("actions", [])


def actions_description(actions: List[Dict]) -> str:
    lines = []
    for a in actions:
        name = a.get("name")
        desc = a.get("description", "")
        args = a.get("args", {})
        args_str = ", ".join(f"{k}: {v}" for k, v in args.items()) if args else "(no args)"
        lines.append(f"- {name}: {desc} Args: {args_str}")
    return "\n".join(lines)


def validate_tool_call(parsed: Dict, actions: List[Dict]) -> bool:
    if not isinstance(parsed, dict):
        return False
    if "tool" not in parsed:
        return False
    tool = parsed.get("tool")
    for a in actions:
        if a.get("name") == tool:
            return True
    return False
