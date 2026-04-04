"""服务异常定义。"""

from __future__ import annotations


class AppError(Exception):
    """应用层异常。"""

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class ValidationError(AppError):
    """请求校验异常。"""

    def __init__(self, message: str) -> None:
        super().__init__("INVALID_REQUEST", message, status_code=400)


class ServiceUnavailableError(AppError):
    """外部能力不可用异常。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message, status_code=503)
