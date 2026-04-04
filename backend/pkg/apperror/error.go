package apperror

import "net/http"

type AppError struct {
	Code       string
	Message    string
	HTTPStatus int
	Data       any
	Err        error
}

func (e *AppError) Error() string {
	return e.Message
}

func Internal(err error) *AppError {
	return &AppError{
		Code:       "INTERNAL_ERROR",
		Message:    "服务内部错误",
		HTTPStatus: http.StatusInternalServerError,
		Err:        err,
	}
}

func InvalidRequest(message string, data any) *AppError {
	return &AppError{
		Code:       "INVALID_REQUEST",
		Message:    message,
		HTTPStatus: http.StatusBadRequest,
		Data:       data,
	}
}

func Unprocessable(code, message string, data any) *AppError {
	return &AppError{
		Code:       code,
		Message:    message,
		HTTPStatus: http.StatusUnprocessableEntity,
		Data:       data,
	}
}

func NotFound(code, message string, data any) *AppError {
	return &AppError{
		Code:       code,
		Message:    message,
		HTTPStatus: http.StatusNotFound,
		Data:       data,
	}
}

func Conflict(code, message string, data any) *AppError {
	return &AppError{
		Code:       code,
		Message:    message,
		HTTPStatus: http.StatusConflict,
		Data:       data,
	}
}

func ServiceUnavailable(code, message string, data any) *AppError {
	return &AppError{
		Code:       code,
		Message:    message,
		HTTPStatus: http.StatusServiceUnavailable,
		Data:       data,
	}
}
