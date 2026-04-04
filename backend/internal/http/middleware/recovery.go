package middleware

import (
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"residential-energy-intelligence-agent-platform/pkg/apperror"
	"residential-energy-intelligence-agent-platform/pkg/response"
)

func Recovery(logger *zap.Logger) gin.HandlerFunc {
	return gin.CustomRecovery(func(c *gin.Context, recovered any) {
		if logger != nil {
			logger.Error("请求处理发生 panic", zap.Any("panic", recovered))
		}
		response.Error(c, apperror.Internal(nil))
	})
}
