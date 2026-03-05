"""Plugin exposing actions to create and list memos and lists.

Actions:
- create_memo: {origin, destination, title, content, metadata, due_date}
- list_memos: {filters?, limit?}
- create_list: {origin, destination, title, items (list of strings), metadata}
- list_lists: {filters?, limit?}

These are lightweight wrappers around `db_adapter` and intended to be
invocable by the assistant's action execution path.
"""
from typing import Any, Dict, List, Optional
import datetime

import db_adapter


def _parse_iso(dt: Optional[str]) -> Optional[datetime.datetime]:
    if not dt:
        return None
    try:
        return datetime.datetime.fromisoformat(dt)
    except Exception:
        return None


async def create_memo(args: Dict[str, Any]):
    origin = args.get("origin")
    destination = args.get("destination")
    title = args.get("title")
    content = args.get("content", "")
    metadata = args.get("metadata")
    due_date = _parse_iso(args.get("due_date"))
    mid = db_adapter.save_memo(origin=origin, destination=destination, content=content, metadata=metadata, title=title, due_date=args.get("due_date"))
    return {"id": mid}


async def list_memos(args: Dict[str, Any]):
    filters = args.get("filters")
    limit = int(args.get("limit", 100))
    memos = db_adapter.list_memos(filters=filters, limit=limit)
    return {"memos": memos}


async def create_list(args: Dict[str, Any]):
    origin = args.get("origin")
    destination = args.get("destination")
    title = args.get("title")
    items = args.get("items") or []
    metadata = args.get("metadata")
    mid = db_adapter.create_list(parent_origin=origin, destination=destination, items=items, metadata=metadata, title=title)
    return {"id": mid}


async def add_list_item(args: Dict[str, Any]):
    # Params: list_id OR origin,destination,title; item_text; create_if_missing
    list_id = args.get("list_id")
    origin = args.get("origin")
    destination = args.get("destination")
    title = args.get("title")
    item_text = args.get("item_text") or args.get("text") or ""
    create_if_missing = bool(args.get("create_if_missing", False))
    try:
        res = db_adapter.add_list_item(list_id=list_id, origin=origin, destination=destination, title=title, item_text=item_text, create_if_missing=create_if_missing)
        return res
    except Exception as e:
        return {"error": str(e)}


async def get_lists_for_user(args: Dict[str, Any]):
    # params: destination (user name), limit
    destination = args.get("destination")
    limit = int(args.get("limit", 10))
    filters = {"type": "list"}
    if destination:
        filters["destination"] = destination
    import db
    lists = db.list_messages(filters=filters, limit=limit)
    return {"lists": lists}


async def get_list(args: Dict[str, Any]):
    list_id = args.get("list_id")
    if not list_id:
        return {"error": "list_id required"}
    import db
    msg = db.get_message(int(list_id))
    if not msg:
        return {"error": "not found"}
    return {"list": msg}


async def list_lists(args: Dict[str, Any]):
    filters = args.get("filters")
    limit = int(args.get("limit", 100))
    # reuse list_messages but filter for type='list'
    if filters is None:
        filters = {"type": "list"}
    else:
        filters = {**filters, "type": "list"}
    import db
    items = db.list_messages(filters=filters, limit=limit)
    return {"lists": items}


def load_plugin():
    return {
        "create_memo": create_memo,
        "list_memos": list_memos,
        "create_list": create_list,
        "list_lists": list_lists,
        "add_list_item": add_list_item,
        "get_lists_for_user": get_lists_for_user,
        "get_list": get_list,
    }


def register():
    """Plugin loader compatibility: return actions mapping under 'actions' key."""
    return {"actions": load_plugin(), "provider": "memos_plugin"}
