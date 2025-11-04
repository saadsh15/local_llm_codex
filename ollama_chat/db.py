# ollama_chat/db.py
import datetime as dt
from pathlib import Path
from typing import List, Tuple

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    create_engine,
    select,
    func,
)
from sqlalchemy.orm import declarative_base, Session, sessionmaker

BASE_DIR = Path.home() / ".ollama_chat"
BASE_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = BASE_DIR / "chat_history.db"
ENGINE = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
SessionLocal = sessionmaker(bind=ENGINE, future=True)

Base = declarative_base()


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True)          # UUID per chat session
    role = Column(String, index=True)               # "user" | "assistant"
    content = Column(Text)
    created_at = Column(DateTime, default=func.now(), index=True)


def init_db() -> None:
    """Create tables if they don't exist."""
    Base.metadata.create_all(ENGINE)


def add_message(session_id: str, role: str, content: str) -> None:
    with SessionLocal() as db:
        db.add(Message(session_id=session_id, role=role, content=content))
        db.commit()


def get_history(session_id: str) -> List[Tuple[str, str]]:
    """Return ordered list of (role, content) for a given session."""
    with SessionLocal() as db:
        stmt = select(Message.role, Message.content).where(
            Message.session_id == session_id
        ).order_by(Message.created_at)
        return db.execute(stmt).all()


def list_sessions() -> List[Tuple[str, dt.datetime]]:
    """Return all distinct session ids with the timestamp of the latest message."""
    with SessionLocal() as db:
        stmt = (
            select(Message.session_id, func.max(Message.created_at))
            .group_by(Message.session_id)
            .order_by(func.max(Message.created_at).desc())
        )
        return db.execute(stmt).all()

