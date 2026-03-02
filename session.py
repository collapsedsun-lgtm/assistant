import json
from typing import List, Tuple, Optional

from kv_store import get_default_kv, messages_list_key


async def load_history(session_id: str = "default") -> Tuple[Optional[object], List[dict]]:
    """Return (kv, history_list).

    If KV is unavailable, returns (None, []).
    """
    history: List[dict] = []
    try:
        kv = await get_default_kv()
    except Exception:
        return None, []

    try:
        raw = await kv.lrange(messages_list_key(session_id), 0, -1)
        if raw:
            for item in raw:
                try:
                    history.append(json.loads(item))
                except Exception:
                    history.append({"role": "assistant", "content": item})
    except Exception:
        # If KV is unavailable for listing, return empty history but keep kv
        return kv, []

    return kv, history


async def persist_message(session_id: str, kv: Optional[object], role: str, content) -> None:
    """Append a message (role/content) to the session message list in KV.

    Swallows errors to avoid crashing the REPL when KV is unavailable.
    """
    if kv is None:
        return
    try:
        await kv.rpush(messages_list_key(session_id), json.dumps({"role": role, "content": content}))
    except Exception:
        return
