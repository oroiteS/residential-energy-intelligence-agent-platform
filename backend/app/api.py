from __future__ import annotations

import uuid
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from flask import Flask, Response, g, jsonify, request

from app.errors import AppError


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def get_request_id() -> str:
    request_id = getattr(g, "request_id", None)
    if request_id:
        return request_id
    return uuid.uuid4().hex


def envelope(*, code: str = "OK", message: str = "success", data: Any = None) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "data": data,
        "request_id": get_request_id(),
        "timestamp": now_iso(),
    }


def success(data: Any = None, *, message: str = "success", code: str = "OK", status: int = 200) -> Response:
    return jsonify(envelope(code=code, message=message, data=data)), status


def register_hooks(app: Flask) -> None:
    @app.before_request
    def assign_request_id() -> None:
        g.request_id = uuid.uuid4().hex
        g.request_started_at = perf_counter()
        app.logger.info(
            "[api] 收到请求 request_id=%s method=%s path=%s remote=%s",
            g.request_id,
            request.method,
            request.path,
            request.remote_addr,
        )

    @app.after_request
    def append_request_id(response: Response) -> Response:
        response.headers["X-Request-ID"] = get_request_id()
        elapsed_ms = (perf_counter() - getattr(g, "request_started_at", perf_counter())) * 1000
        app.logger.info(
            "[api] 完成请求 request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
            get_request_id(),
            request.method,
            request.path,
            response.status_code,
            elapsed_ms,
        )
        return response


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(AppError)
    def handle_app_error(error: AppError) -> tuple[Response, int]:
        return jsonify(envelope(code=error.code, message=error.message, data=None)), error.status_code

    @app.errorhandler(404)
    def handle_404(_error: Exception) -> tuple[Response, int]:
        return jsonify(envelope(code="NOT_FOUND", message="接口不存在", data=None)), 404

    @app.errorhandler(500)
    def handle_500(error: Exception) -> tuple[Response, int]:
        app.logger.exception("未处理异常: %s", error)
        return jsonify(envelope(code="INTERNAL_ERROR", message="服务器内部错误", data=None)), 500
