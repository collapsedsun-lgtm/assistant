"""Thin adapter layer exposing simple assistant-facing DB helpers."""
from typing import Any, Dict, List, Optional

import db


def init(db_path: Optional[str] = None, echo: bool = False):
    """Initialize the database. Call once at startup if desired."""
    return db.init_db(db_path=db_path, echo=echo)


def save_memo(origin: Optional[str], destination: Optional[str], content: str, metadata: Optional[Dict] = None, title: Optional[str] = None, due_date: Optional[str] = None) -> int:
    # due_date may be an ISO string; db.create_message expects datetime or None
    dd = None
    if due_date:
        try:
            import datetime as _dt

            dd = _dt.datetime.fromisoformat(due_date)
        except Exception:
            dd = None
    return db.create_message(type="memo", origin=origin, destination=destination, content=content, metadata=metadata, title=title, due_date=dd)


def list_memos(filters: Optional[Dict] = None, limit: int = 100) -> List[Dict]:
    if filters is None:
        filters = {"type": "memo"}
    else:
        filters = {**filters, "type": filters.get("type", "memo")}
    return db.list_messages(filters=filters, limit=limit)


def mark_memo_read(message_id: int) -> bool:
    return db.update_message_state(message_id, "read")


def create_list(parent_origin: Optional[str], destination: Optional[str], items: List[str], metadata: Optional[Dict] = None, title: Optional[str] = None) -> int:
    msg_id = db.create_message(type="list", origin=parent_origin, destination=destination, content="", metadata=metadata, title=title)
    for idx, text in enumerate(items):
        db.create_list_item(message_id=msg_id, position=idx, text=text, checked=False)
    return msg_id


def add_list_item(list_id: int = None, origin: Optional[str] = None, destination: Optional[str] = None, title: Optional[str] = None, item_text: str = "", create_if_missing: bool = False) -> Dict:
    """Add an item to a list identified by `list_id` or by origin/destination/title.

    If the list cannot be found and `create_if_missing` is True, a new list is created.
    Returns dict with `list_id` and `item_id`.
    """
    target_id = list_id
    if target_id is None:
        # try to find most recent matching list
        filters = {"type": "list"}
        if origin:
            filters["origin"] = origin
        if destination:
            filters["destination"] = destination
        if title:
            filters["title"] = title
        candidates = db.list_messages(filters=filters, limit=1)
        if candidates:
            target_id = candidates[0]["id"]
    if target_id is None and create_if_missing:
        target_id = create_list(parent_origin=origin, destination=destination, items=[], metadata=None, title=title)

    if target_id is None:
        raise ValueError("List not found and create_if_missing=False")

    # determine position: append at end
    existing = db.list_messages(filters={"type": "list", "id": target_id}, limit=1)
    # we can't get item count easily from list_messages; fallback to 0
    # use db.get_message to inspect
    msg = db.get_message(target_id)
    pos = 0
    if msg and "list_items" in msg:
        pos = len(msg["list_items"]) if msg["list_items"] else 0

    item_id = db.create_list_item(message_id=target_id, position=pos, text=item_text, checked=False)
    return {"list_id": target_id, "item_id": item_id}


def check_item(item_id: int, checked: bool = True) -> bool:
    return db.mark_item_checked(item_id, checked=checked)
