package handler

import (
	"strconv"

	"github.com/gin-gonic/gin"

	"residential-energy-intelligence-agent-platform/pkg/apperror"
	"residential-energy-intelligence-agent-platform/pkg/response"
)

func parseUint64Param(c *gin.Context, key string) (uint64, bool) {
	raw := c.Param(key)
	value, err := strconv.ParseUint(raw, 10, 64)
	if err != nil {
		response.Error(c, apperror.InvalidRequest("路径参数非法", map[string]any{key: raw}))
		return 0, false
	}
	return value, true
}
