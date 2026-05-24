from __future__ import annotations

from flask import Blueprint, request

from app.api import success
from app.services.agent_service import ask_agent


agent_bp = Blueprint("agent", __name__)


@agent_bp.post("/agent/ask")
def post_ask():
    """智能问答接口。

    路由层只负责读取请求参数，具体会话保存、上下文构造和 LLM 调用由 agent_service 完成。
    """

    payload = request.get_json(silent=True) or {}
    result = ask_agent(
        dataset_id=int(payload["dataset_id"]),
        question=payload.get("question", ""),
        session_id=payload.get("session_id"),
        history=payload.get("history") or [],
    )
    return success(result)
