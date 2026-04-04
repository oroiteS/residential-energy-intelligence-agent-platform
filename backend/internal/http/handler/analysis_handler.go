package handler

import (
	"github.com/gin-gonic/gin"

	"residential-energy-intelligence-agent-platform/internal/service"
	"residential-energy-intelligence-agent-platform/pkg/response"
)

type AnalysisHandler struct {
	service *service.AnalysisService
}

func NewAnalysisHandler(service *service.AnalysisService) *AnalysisHandler {
	return &AnalysisHandler{service: service}
}

func (h *AnalysisHandler) Get(c *gin.Context) {
	id, ok := parseUint64Param(c, "id")
	if !ok {
		return
	}

	data, appErr := h.service.Get(c.Request.Context(), id)
	if appErr != nil {
		response.Error(c, appErr)
		return
	}
	response.OK(c, data)
}

func (h *AnalysisHandler) Recompute(c *gin.Context) {
	id, ok := parseUint64Param(c, "id")
	if !ok {
		return
	}

	data, appErr := h.service.Recompute(c.Request.Context(), id)
	if appErr != nil {
		response.Error(c, appErr)
		return
	}
	response.OK(c, data)
}
