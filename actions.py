import json
import os
from typing import Dict, List, Any, Optional

from pydantic import BaseModel, ValidationError


_ACTIONS_PATH = os.path.join(os.path.dirname(__file__), "actions.json")


class ActionSpec(BaseModel):
    name: str
    description: Optional[str] = None
    args: Dict[str, str] = {}


class ToolCall(BaseModel):
    tool: str
    args: Dict[str, Any] = {}


def load_actions() -> List[ActionSpec]:
    with open(_ACTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [ActionSpec(**a) for a in data.get("actions", [])]


def actions_description(actions: List[ActionSpec]) -> str:
    lines = []
    for a in actions:
        name = a.name
        desc = a.description or ""
        args = a.args or {}
        args_str = ", ".join(f"{k}: {v}" for k, v in args.items()) if args else "(no args)"
        lines.append(f"- {name}: {desc} Args: {args_str}")
    return "\n".join(lines)


def validate_tool_call(parsed: Any, actions: List[ActionSpec]) -> Optional[ToolCall]:
    """
    Parse and validate a tool call JSON against the known actions.

    Returns a `ToolCall` pydantic model if valid, otherwise None.
    """
    if not isinstance(parsed, dict):
        return None
    try:
        call = ToolCall(**parsed)
    except ValidationError:
        return None

    # Ensure the tool exists in the action specs
    for a in actions:
        if a.name == call.tool:
            # Basic argument key validation: provided keys must be in spec args
            if call.args:
                for k in call.args.keys():
                    if k not in a.args:
                        return None
            return call
    return None
