"""Helpers for loading and validating action (tool) specifications.

This module treats `actions.json` as the single source of truth for the
available tools the agent can request. It exposes lightweight
Pydantic-based models for the action spec and a tool-call instance so
the agent can validate model output before any runner attempts to
execute it.
"""

import json
import os
from typing import Dict, List, Any, Optional

from pydantic import BaseModel, ValidationError


# Path to the JSON file describing available actions
_ACTIONS_PATH = os.path.join(os.path.dirname(__file__), "actions.json")


class ActionSpec(BaseModel):
    """Specification for a single action/tool.

    Fields:
    - name: action identifier
    - description: human-friendly description (for prompts)
    - args: mapping of argument name -> type (string description only)
    """
    name: str
    description: Optional[str] = None
    args: Dict[str, str] = {}


class ToolCall(BaseModel):
    """Model of a tool call emitted by the LLM.

    This mirrors the JSON contract we expect the model to follow.
    """
    tool: str
    args: Dict[str, Any] = {}


def load_actions() -> List[ActionSpec]:
    """Load and return the list of configured action specs."""
    with open(_ACTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [ActionSpec(**a) for a in data.get("actions", [])]


def actions_description(actions: List[ActionSpec]) -> str:
    """Render a short human-readable description of each action.

    This string is appended to the system prompt so the model can see
    the available tools and their argument names.
    """
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
    The validation checks that the emitted JSON has the expected keys
    and that the `tool` exists in the action specs. It also ensures the
    provided argument keys are present in the spec (simple key-level
    validation only).
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
