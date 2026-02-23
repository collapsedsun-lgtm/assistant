from typing import List
import json


def summarize(history: List[dict], rolling_window: int, max_chars: int = 600) -> str:
    """
    Produce a human-friendly summary of the most recent exchanges.

    The summarizer converts recent user/assistant pairs into short sentences
    (e.g. "User asked X. Assistant emitted action Y with args Z.") so the LLM
    can read and use the memory more reliably.
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
