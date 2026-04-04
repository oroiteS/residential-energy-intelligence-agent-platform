"""HTTP 相关辅助。"""

from __future__ import annotations

import json
from typing import Any

from robyn import Request, Response, status_codes

from app.errors import AppError, ValidationError


JSON_HEADERS = {"Content-Type": "application/json; charset=utf-8"}


def json_response(payload: dict[str, Any], status_code: int = 200) -> Response:
    return Response(status_code=status_code, headers=JSON_HEADERS, description=json.dumps(payload, ensure_ascii=False))


def success(payload: dict[str, Any]) -> Response:
    return json_response(payload, status_codes.HTTP_200_OK)


def error_response(error: AppError) -> Response:
    return json_response({"code": error.code, "message": error.message}, error.status_code)


def load_request_json(request: Request) -> dict[str, Any]:
    try:
        payload = request.json
        if callable(payload):
            payload = payload()
        if payload is None:
            raise ValidationError("请求体不能为空")
        if isinstance(payload, bytes):
            payload = json.loads(payload.decode("utf-8"))
        if isinstance(payload, str):
            payload = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValidationError("请求体不是合法 JSON") from exc
    if not isinstance(payload, dict):
        raise ValidationError("请求体必须是 JSON 对象")
    return payload
