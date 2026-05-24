from __future__ import annotations


class AppError(Exception):
    """后端业务异常基类。

    业务层通过抛出该异常把错误信息、错误码和 HTTP 状态码交给统一错误处理器，
    避免各个接口重复拼装错误响应。
    """

    def __init__(self, message: str, *, code: str = "BAD_REQUEST", status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class NotFoundError(AppError):
    """资源不存在错误，对应 HTTP 404。"""

    def __init__(self, message: str, *, code: str = "NOT_FOUND") -> None:
        super().__init__(message, code=code, status_code=404)


class ValidationError(AppError):
    """请求参数或业务前置条件不满足，对应 HTTP 422。"""

    def __init__(self, message: str, *, code: str = "VALIDATION_ERROR") -> None:
        super().__init__(message, code=code, status_code=422)


class ServiceUnavailableError(AppError):
    """外部服务或可选能力不可用，对应 HTTP 503。"""

    def __init__(self, message: str, *, code: str = "SERVICE_UNAVAILABLE") -> None:
        super().__init__(message, code=code, status_code=503)
