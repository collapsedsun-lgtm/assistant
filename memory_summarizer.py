from typing import List
import json


def summarize(history: List[dict], rolling_window: int, max_chars: int = 600) -> str:
    """
    Produce a minimal summary of the most recent exchanges.

    This is a lightweight, local summarizer used to compress memory before
    sending to the model. It simply concatenates the last N exchanges into
    a short text blob and truncates to `max_chars`.
    """
    # take the last rolling_window * 2 messages (user + assistant pairs)
    start = max(0, len(history) - (rolling_window * 2))
    recent = history[start:]

    parts = []
    for m in recent:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, dict):
            try:
                content = json.dumps(content)
            except Exception:
                content = str(content)
        parts.append(f"{role}: {content}")

    joined = " \n ".join(parts)
    if len(joined) <= max_chars:
        return joined
    # truncate cleanly
    return joined[: max_chars - 3] + "..."
