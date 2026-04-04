package response

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"

	"residential-energy-intelligence-agent-platform/pkg/apperror"
)

type Envelope struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	Data      any    `json:"data"`
	RequestID string `json:"request_id"`
	Timestamp string `json:"timestamp"`
}

func Success(c *gin.Context, status int, code, message string, data any) {
	c.JSON(status, Envelope{
		Code:      code,
		Message:   message,
		Data:      data,
		RequestID: requestID(c),
		Timestamp: now(),
	})
}

func OK(c *gin.Context, data any) {
	Success(c, http.StatusOK, "OK", "success", data)
}

func Error(c *gin.Context, appErr *apperror.AppError) {
	if appErr == nil {
		appErr = apperror.Internal(nil)
	}
	c.JSON(appErr.HTTPStatus, Envelope{
		Code:      appErr.Code,
		Message:   appErr.Message,
		Data:      appErr.Data,
		RequestID: requestID(c),
		Timestamp: now(),
	})
}

func requestID(c *gin.Context) string {
	value, exists := c.Get("request_id")
	if !exists {
		return ""
	}
	id, ok := value.(string)
	if !ok {
		return ""
	}
	return id
}

func now() string {
	return time.Now().Format(time.RFC3339)
}
