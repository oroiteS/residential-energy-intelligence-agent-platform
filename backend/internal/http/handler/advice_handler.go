package handler

import (
	"github.com/gin-gonic/gin"

	"residential-energy-intelligence-agent-platform/internal/service"
	"residential-energy-intelligence-agent-platform/pkg/response"
)

type AdviceHandler struct {
	service *service.AdviceService
}

func NewAdviceHandler(service *service.AdviceService) *AdviceHandler {
	return &AdviceHandler{service: service}
}

func (h *AdviceHandler) Generate(c *gin.Context) {
	id, ok := parseUint64Param(c, "id")
	if !ok {
		return
	}

	advice, appErr := h.service.Generate(c.Request.Context(), id)
	if appErr != nil {
		response.Error(c, appErr)
		return
	}
	response.OK(c, advice)
}

func (h *AdviceHandler) List(c *gin.Context) {
	id, ok := parseUint64Param(c, "id")
	if !ok {
		return
	}

	items, appErr := h.service.List(c.Request.Context(), id, c.Query("advice_type"))
	if appErr != nil {
		response.Error(c, appErr)
		return
	}
	response.OK(c, gin.H{"items": items})
}

func (h *AdviceHandler) Get(c *gin.Context) {
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
