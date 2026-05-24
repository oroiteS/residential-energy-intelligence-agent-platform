from __future__ import annotations

import uuid
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from flask import Flask, Response, g, jsonify, request
from flask.typing import ResponseReturnValue

from app.errors import AppError


def now_iso() -> str:
    """生成接口响应使用的本地 ISO 时间戳。"""

    return datetime.now(timezone.utc).astimezone().isoformat()


def get_request_id() -> str:
    """获取当前请求的追踪 ID。

    请求钩子会优先写入 g.request_id；如果在非请求上下文中调用，
    则临时生成一个 ID，保证响应 envelope 始终包含 request_id 字段。
    """

    request_id = getattr(g, "request_id", None)
    if request_id:
        return request_id
    return uuid.uuid4().hex


def envelope(*, code: str = "OK", message: str = "success", data: Any = None) -> dict[str, Any]:
    """构造统一 API 响应外壳。

    返回结构固定包含 code、message、data、request_id 和 timestamp，
    前端可以用同一套逻辑处理成功响应和错误响应。
    """

    return {
        "code": code,
        "message": message,
        "data": data,
        "request_id": get_request_id(),
        "timestamp": now_iso(),
    }


def success(data: Any = None, *, message: str = "success", code: str = "OK", status: int = 200) -> ResponseReturnValue:
    """返回统一格式的成功响应。"""

    return jsonify(envelope(code=code, message=message, data=data)), status


def register_hooks(app: Flask) -> None:
    """注册请求生命周期钩子。

    before_request 负责生成 request_id 并记录开始时间；
    after_request 负责写回响应头并输出耗时日志，便于答辩时说明接口可追踪性。
    """

    @app.before_request
    def assign_request_id() -> None:
        # request_id 会贯穿日志、响应体和响应头，便于定位某一次具体请求。
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
        # 将 request_id 暴露给前端或调试工具，方便接口联调时关联后端日志。
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
    """注册统一错误处理。

    业务主动抛出的 AppError 会保留自定义 code 和状态码；
    未命中路由和未处理异常也会转换为统一响应结构。
    """

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
