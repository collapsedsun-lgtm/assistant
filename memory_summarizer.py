"""Lightweight, local memory summarizer.

The summarizer converts recent user/assistant exchanges into short,
human-friendly sentences so that small models can more reliably use
the content when answering memory-related questions. It is intentionally
simple (no external LLM calls) so it remains fast and deterministic.
"""

from typing import List
import json


def summarize(history: List[dict], rolling_window: int, max_chars: int = 600) -> str:
    """Produce a concise, human-readable summary of recent exchanges.

    The function takes the most recent `rolling_window` exchanges
    (each exchange is a user message + assistant reply) and converts
    them into sentences such as:

      "User: turn off the lights in the bedroom. Assistant emitted action null (room=bedroom)."

    This format is easier for small models to parse and reason about
    than raw JSON blobs.
    """
    # take the last rolling_window * 2 messages (user + assistant pairs)
    start = max(0, len(history) - (rolling_window * 2))
    recent = history[start:]

    sentences = []
    # iterate in pairs (user then assistant)
    i = 0
    while i < len(recent):
        user_msg = recent[i]
        assistant_msg = recent[i + 1] if i + 1 < len(recent) else None

        utext = str(user_msg.get("content", "")).strip()
        # Normalize assistant content
        acontent = assistant_msg.get("content") if assistant_msg else None
        if isinstance(acontent, dict):
            try:
                atext = json.dumps(acontent)
            except Exception:
                atext = str(acontent)
        else:
            atext = str(acontent) if acontent is not None else ""

        # If assistant content looks like a tool call JSON, try to parse
        try:
            parsed = json.loads(atext) if atext else None
        except Exception:
            parsed = None

        if parsed and isinstance(parsed, dict) and "tool" in parsed:
            tool = parsed.get("tool")
            args = parsed.get("args", {})
            args_str = ", ".join(f"{k}={v}" for k, v in args.items()) if args else "no args"
            sentences.append(f"User: {utext}. Assistant emitted action {tool} ({args_str}).")
        else:
            sentences.append(f"User: {utext}. Assistant: {atext}.")

        i += 2

    joined = " \n ".join(sentences)
    if len(joined) <= max_chars:
        return joined
    return joined[: max_chars - 3] + "..."
