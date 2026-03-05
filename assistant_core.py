import json
from typing import List

import actions
import memory_summarizer
import db_adapter


def build_messages(
    system_prompt: str,
    actions_desc: str,
    history: List[dict],
    user_input: str,
    rolling_window: int,
    summarize_memory: bool = False,
    prefetch_texts: List[str] | None = None,
) -> List[dict]:
    # Compose system prompt. Avoid duplicating action instructions when the
    # base prompt already includes an "Available actions" section.
    if actions_desc and "available actions" not in system_prompt.lower():
        system_content = system_prompt + "\n\nAvailable actions:\n" + actions_desc
    else:
        system_content = system_prompt

    msgs = [{"role": "system", "content": system_content}]
    # Optionally include a memory summary as a system-level note
    if summarize_memory and history:
        summary = memory_summarizer.summarize(history, rolling_window)
        msgs.append({"role": "system", "content": "Memory summary: " + summary})

    # Optionally include sanitized pre-fetched facts (from pre_send hooks)
    if prefetch_texts:
        combined = "\n\n".join(t for t in prefetch_texts if t)
        if combined:
            msgs.append({"role": "system", "content": "Pre-fetched facts (sanitized):\n" + combined})

    # Include rolling history (already formatted as role/content dicts)
    start = max(0, len(history) - (rolling_window * 2))
    # Ensure message contents are strings when sent to the LLM
    for m in history[start:]:
        content = m.get("content")
        if isinstance(content, dict):
            try:
                content_str = json.dumps(content)
            except Exception:
                content_str = str(content)
        else:
            content_str = str(content)
        msgs.append({"role": m.get("role", "user"), "content": content_str})

    # Add current user input
    msgs.append({"role": "user", "content": user_input})
    return msgs


def try_parse_tool_call(llm_output: str, actions_list: List[actions.ActionSpec]):
    # Be permissive: the model may include extra text around the JSON
    # tool call (or prefixes like model names). Try direct parse first,
    # then search for a JSON object substring.
    try:
        parsed = json.loads(llm_output)
        return actions.validate_tool_call(parsed, actions_list)
    except Exception:
        pass

    # Attempt to find first {...} JSON object in the output (compat shim)
    start = llm_output.find("{")
    end = llm_output.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = llm_output[start : end + 1]
    try:
        parsed = json.loads(candidate)
        return actions.validate_tool_call(parsed, actions_list)
    except Exception:
        return None


def parse_tool_calls(llm_output: str, actions_list: List[actions.ActionSpec]):
    """Extract leading text and all valid tool calls (JSON objects) from LLM output.

    Returns (text_prefix, [ToolCall, ...]) where text_prefix is the text before
    the first JSON object (trimmed) and the list contains validated `ToolCall`
    pydantic models for each JSON object found.
    """
    calls = []
    first_brace = llm_output.find("{")
    text_prefix = llm_output[:first_brace].strip() if first_brace != -1 else llm_output.strip()
    idx = 0
    L = len(llm_output)
    while True:
        start = llm_output.find("{", idx)
        if start == -1:
            break
        depth = 0
        end = -1
        for j in range(start, L):
            ch = llm_output[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end == -1:
            break
        candidate = llm_output[start : end + 1]
        try:
            parsed_json = json.loads(candidate)
            validated = actions.validate_tool_call(parsed_json, actions_list)
            if validated:
                calls.append(validated)
        except Exception:
            pass
        idx = end + 1
    return text_prefix, calls


def save_memo_via_db(origin: str | None, destination: str | None, content: str, metadata: dict | None = None) -> int:
    """Helper exposed for other modules to persist a memo."""
    return db_adapter.save_memo(origin=origin, destination=destination, content=content, metadata=metadata)


def list_memos_via_db(filters: dict | None = None, limit: int = 100):
    return db_adapter.list_memos(filters=filters, limit=limit)
