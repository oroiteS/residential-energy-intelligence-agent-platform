package handler

import (
	"github.com/gin-gonic/gin"

	"residential-energy-intelligence-agent-platform/internal/service"
	"residential-energy-intelligence-agent-platform/pkg/apperror"
	"residential-energy-intelligence-agent-platform/pkg/response"
)

type SystemConfigHandler struct {
	service *service.SystemConfigService
}

func NewSystemConfigHandler(service *service.SystemConfigService) *SystemConfigHandler {
	return &SystemConfigHandler{service: service}
}

func (h *SystemConfigHandler) Get(c *gin.Context) {
	config, err := h.service.Get(c.Request.Context())
	if err != nil {
		response.Error(c, apperror.Internal(err))
		return
	}
	response.OK(c, config)
}

func (h *SystemConfigHandler) Patch(c *gin.Context) {
	var input service.UpdateSystemConfigInput
	if err := c.ShouldBindJSON(&input); err != nil {
		response.Error(c, apperror.InvalidRequest("请求体格式错误", map[string]any{
			"error": err.Error(),
		}))
		return
	}

	config, appErr := h.service.Update(c.Request.Context(), input)
	if appErr != nil {
		response.Error(c, appErr)
		return
	}

	response.OK(c, config)
}
