package handler

import (
	"github.com/gin-gonic/gin"

	"residential-energy-intelligence-agent-platform/internal/service"
	"residential-energy-intelligence-agent-platform/pkg/apperror"
	"residential-energy-intelligence-agent-platform/pkg/response"
)

type AgentHandler struct {
	service *service.AgentService
}

func NewAgentHandler(service *service.AgentService) *AgentHandler {
	return &AgentHandler{service: service}
}

func (h *AgentHandler) Ask(c *gin.Context) {
	var payload service.AgentAskInput
	if err := c.ShouldBindJSON(&payload); err != nil {
		response.Error(c, apperror.InvalidRequest("请求体格式错误", map[string]any{"error": err.Error()}))
		return
	}

	data, appErr := h.service.Ask(c.Request.Context(), payload)
	if appErr != nil {
		response.Error(c, appErr)
		return
	}
	response.OK(c, data)
}
