from __future__ import annotations

from flask import Blueprint, request

from app.api import success
from app.services.chat_service import create_session, list_messages, list_sessions


chat_bp = Blueprint("chat", __name__)


@chat_bp.get("/chat/sessions")
def get_sessions():
    dataset_id = request.args.get("dataset_id", type=int)
    return success({"items": list_sessions(dataset_id)})


@chat_bp.post("/chat/sessions")
def post_session():
    payload = request.get_json(silent=True) or {}
    session = create_session(
        dataset_id=int(payload.get("dataset_id")),
        title=payload.get("title", "新会话"),
    )
    return success({"session": session})


@chat_bp.get("/chat/sessions/<int:session_id>/messages")
def get_messages(session_id: int):
    return success({"items": list_messages(session_id)})

