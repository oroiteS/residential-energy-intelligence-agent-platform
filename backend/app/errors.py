from __future__ import annotations


class AppError(Exception):
    def __init__(self, message: str, *, code: str = "BAD_REQUEST", status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class NotFoundError(AppError):
    def __init__(self, message: str, *, code: str = "NOT_FOUND") -> None:
        super().__init__(message, code=code, status_code=404)


class ValidationError(AppError):
    def __init__(self, message: str, *, code: str = "VALIDATION_ERROR") -> None:
        super().__init__(message, code=code, status_code=422)


class ServiceUnavailableError(AppError):
    def __init__(self, message: str, *, code: str = "SERVICE_UNAVAILABLE") -> None:
        super().__init__(message, code=code, status_code=503)
