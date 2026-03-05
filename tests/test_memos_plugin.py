import importlib.util
from pathlib import Path
import asyncio


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_plugin_create_list_and_memo(tmp_path):
    # initialize DB in temp path
    import sys
    sys.path.insert(0, str(Path('.').resolve()))
    # load `db` as the canonical module name so plugin imports use same module
    import importlib
    db = importlib.import_module("db")
    db.init_db(db_path=str(tmp_path / "p.db"))

    plugin = load_module(Path("./plugins/memos_plugin.py"), "memos_plugin")
    # run create_list
    coro = plugin.create_list({"origin": "alice", "destination": "bob", "title": "Shop", "items": ["eggs", "milk"]})
    res = asyncio.get_event_loop().run_until_complete(coro)
    assert "id" in res

    # run list_lists
    res2 = asyncio.get_event_loop().run_until_complete(plugin.list_lists({}))
    assert "lists" in res2

    # create memo
    res3 = asyncio.get_event_loop().run_until_complete(plugin.create_memo({"origin": "carol", "destination": "dave", "title": "Note", "content": "remember"}))
    assert "id" in res3

    res4 = asyncio.get_event_loop().run_until_complete(plugin.list_memos({}))
    assert "memos" in res4

    # add items incrementally
    add1 = asyncio.get_event_loop().run_until_complete(plugin.add_list_item({"list_id": res["id"], "item_text": "dishwasher"}))
    assert "item_id" in add1
    add2 = asyncio.get_event_loop().run_until_complete(plugin.add_list_item({"list_id": res["id"], "item_text": "bread"}))
    assert "item_id" in add2
    # read list
    got = asyncio.get_event_loop().run_until_complete(plugin.get_list({"list_id": res["id"]}))
    assert "list" in got and len(got["list"]["list_items"]) >= 2
