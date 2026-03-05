import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import (Boolean, Column, DateTime, ForeignKey, Integer,
                        String, Text, create_engine)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    title = Column(String)
    type = Column(String, nullable=False)
    origin = Column(String)
    destination = Column(String)
    state = Column(String, default="unread")
    content = Column(Text)
    metadata_json = Column("metadata", Text)  # JSON stored as text; use different attribute name
    due_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    list_items = relationship("ListItem", back_populates="message", cascade="all, delete-orphan")


class ListItem(Base):
    __tablename__ = "list_items"

    id = Column(Integer, primary_key=True)
    message_id = Column(Integer, ForeignKey("messages.id", ondelete="CASCADE"))
    position = Column(Integer)
    text = Column(Text)
    title = Column(String)
    notes = Column(Text)
    sent_at = Column(DateTime)
    checked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    message = relationship("Message", back_populates="list_items")


_engine = None
_SessionLocal = None


def init_db(db_path: Optional[str] = None, echo: bool = False) -> Tuple[Any, Any]:
    """Initialize the SQLite database and return (engine, SessionLocal).

    If `db_path` is None, a file `assistant.db` is created next to this module.
    """
    global _engine, _SessionLocal
    if db_path is None:
        root = os.path.dirname(__file__)
        db_path = os.path.join(root, "assistant.db")
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url, echo=echo, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)

    # Ensure migrations for additive columns on existing DBs (safe no-op if columns exist).
    try:
        with engine.connect() as conn:
            # messages table columns to ensure
            res = conn.execute("PRAGMA table_info(messages)")
            cols = [row[1] for row in res.fetchall()] if res is not None else []
            if "title" not in cols:
                try:
                    conn.execute("ALTER TABLE messages ADD COLUMN title TEXT")
                except Exception:
                    pass
            if "due_date" not in cols:
                try:
                    conn.execute("ALTER TABLE messages ADD COLUMN due_date DATETIME")
                except Exception:
                    pass

            # list_items additions
            res2 = conn.execute("PRAGMA table_info(list_items)")
            cols2 = [row[1] for row in res2.fetchall()] if res2 is not None else []
            if "title" not in cols2:
                try:
                    conn.execute("ALTER TABLE list_items ADD COLUMN title TEXT")
                except Exception:
                    pass
            if "notes" not in cols2:
                try:
                    conn.execute("ALTER TABLE list_items ADD COLUMN notes TEXT")
                except Exception:
                    pass
            if "sent_at" not in cols2:
                try:
                    conn.execute("ALTER TABLE list_items ADD COLUMN sent_at DATETIME")
                except Exception:
                    pass
    except Exception:
        # best-effort migration; ignore failures
        pass
    _engine = engine
    _SessionLocal = SessionLocal
    return engine, SessionLocal


def _get_session():
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()


def _now():
    return datetime.utcnow()


def create_message(type: str, origin: Optional[str] = None, destination: Optional[str] = None,
                   state: str = "unread", content: str = "", metadata: Optional[Dict] = None,
                   title: Optional[str] = None, due_date: Optional[datetime] = None) -> int:
    session = _get_session()
    msg = Message(
        type=type,
        origin=origin,
        destination=destination,
        state=state,
        content=content,
        metadata_json=json.dumps(metadata) if metadata is not None else None,
        title=title,
        due_date=due_date,
        created_at=_now(),
        updated_at=_now(),
    )
    session.add(msg)
    session.commit()
    session.refresh(msg)
    session.close()
    return msg.id


def get_message(message_id: int) -> Optional[Dict]:
    session = _get_session()
    msg = session.query(Message).filter(Message.id == message_id).one_or_none()
    if not msg:
        session.close()
        return None
    result = {
        "id": msg.id,
        "title": msg.title,
        "type": msg.type,
        "origin": msg.origin,
        "destination": msg.destination,
        "state": msg.state,
        "content": msg.content,
        "metadata": json.loads(msg.metadata_json) if msg.metadata_json else None,
        "due_date": msg.due_date,
        "created_at": msg.created_at,
        "updated_at": msg.updated_at,
        "list_items": [
            {"id": it.id, "position": it.position, "title": it.title, "text": it.text, "checked": bool(it.checked), "notes": it.notes, "sent_at": it.sent_at}
            for it in msg.list_items
        ],
    }
    session.close()
    return result


def list_messages(filters: Optional[Dict] = None, limit: int = 100) -> List[Dict]:
    session = _get_session()
    q = session.query(Message)
    if filters:
        if "type" in filters:
            q = q.filter(Message.type == filters["type"])
        if "origin" in filters:
            q = q.filter(Message.origin == filters["origin"])
        if "destination" in filters:
            q = q.filter(Message.destination == filters["destination"])
        if "state" in filters:
            q = q.filter(Message.state == filters["state"])
    q = q.order_by(Message.created_at.desc()).limit(limit)
    results = []
    for msg in q:
        results.append({
            "id": msg.id,
            "title": msg.title,
            "type": msg.type,
            "origin": msg.origin,
            "destination": msg.destination,
            "state": msg.state,
            "content": msg.content,
            "metadata": json.loads(msg.metadata_json) if msg.metadata_json else None,
            "due_date": msg.due_date,
            "created_at": msg.created_at,
            "updated_at": msg.updated_at,
        })
    session.close()
    return results


def update_message_state(message_id: int, new_state: str) -> bool:
    session = _get_session()
    msg = session.query(Message).filter(Message.id == message_id).one_or_none()
    if not msg:
        session.close()
        return False
    msg.state = new_state
    msg.updated_at = _now()
    session.commit()
    session.close()
    return True


def create_list_item(message_id: int, position: int, text: str, checked: bool = False, title: Optional[str] = None, notes: Optional[str] = None, sent_at: Optional[datetime] = None) -> int:
    session = _get_session()
    itm = ListItem(message_id=message_id, position=position, title=title, text=text, checked=checked, notes=notes, sent_at=sent_at, created_at=_now())
    session.add(itm)
    session.commit()
    session.refresh(itm)
    session.close()
    return itm.id


def mark_item_checked(item_id: int, checked: bool = True) -> bool:
    session = _get_session()
    itm = session.query(ListItem).filter(ListItem.id == item_id).one_or_none()
    if not itm:
        session.close()
        return False
    itm.checked = checked
    session.commit()
    session.close()
    return True
