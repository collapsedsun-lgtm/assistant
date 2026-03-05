import importlib.util
from pathlib import Path


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_db_crud(tmp_path):
    db_path = tmp_path / "test_assistant.db"
    db = load_module(Path("./db.py"), "db_test")

    # initialize DB at test path
    engine, Session = db.init_db(db_path=str(db_path))

    # create a memo
    mid = db.create_message(type="memo", origin="alice", destination="bob", content="hello", metadata={"k": "v"})
    assert isinstance(mid, int) and mid > 0

    # list messages and check metadata
    msgs = db.list_messages()
    assert any(m["id"] == mid for m in msgs)

    msg = db.get_message(mid)
    assert msg is not None
    assert msg["origin"] == "alice"
    assert msg["destination"] == "bob"
    assert msg["metadata"] == {"k": "v"}


def test_lists_and_items(tmp_path):
    db_path = tmp_path / "test_assistant2.db"
    db = load_module(Path("./db.py"), "db_test2")

    db.init_db(db_path=str(db_path))

    # create a list message and items
    mid = db.create_message(type="list", origin="carol", destination="dave", content="")
    assert mid > 0

    # add items
    itm1 = db.create_list_item(message_id=mid, position=0, text="first", checked=False)
    itm2 = db.create_list_item(message_id=mid, position=1, text="second", checked=False)
    assert itm1 > 0 and itm2 > 0

    msg = db.get_message(mid)
    assert msg is not None
    assert len(msg["list_items"]) == 2

    # mark item checked
    ok = db.mark_item_checked(itm1, checked=True)
    assert ok is True
