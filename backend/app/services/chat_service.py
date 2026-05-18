from __future__ import annotations

from app.extensions import db
from app.models import ChatMessage, ChatSession
from app.services.common import to_iso


def list_sessions(dataset_id: int | None) -> list[dict]:
    query = ChatSession.query
    if dataset_id is not None:
        query = query.filter_by(dataset_id=dataset_id)
    sessions = query.order_by(ChatSession.updated_at.desc()).all()
    return [session_dto(item) for item in sessions]


def create_session(dataset_id: int, title: str) -> dict:
    session = ChatSession(dataset_id=dataset_id, title=title.strip() or "新会话")
    db.session.add(session)
    db.session.commit()
    return session_dto(session)


def list_messages(session_id: int) -> list[dict]:
    messages = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.created_at.asc()).all()
    return [message_dto(item) for item in messages]


def append_message(
    *,
    session_id: int,
    role: str,
    content: str | None,
    assistant_payload: dict | None = None,
    model_name: str | None = None,
) -> ChatMessage:
    message = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        assistant_payload=assistant_payload,
        model_name=model_name,
        tokens_used=None,
    )
    db.session.add(message)
    db.session.commit()
    return message


def session_dto(record: ChatSession) -> dict:
    return {
        "id": record.id,
        "dataset_id": record.dataset_id,
        "title": record.title or "未命名会话",
        "created_at": to_iso(record.created_at),
        "updated_at": to_iso(record.updated_at),
    }


def message_dto(record: ChatMessage) -> dict:
    return {
        "id": record.id,
        "session_id": record.session_id,
        "role": record.role,
        "content": record.content or "",
        "assistant_payload": record.assistant_payload,
        "content_path": record.content_path,
        "model_name": record.model_name,
        "tokens_used": record.tokens_used,
        "created_at": to_iso(record.created_at),
    }

