package handler

import (
	"strconv"

	"github.com/gin-gonic/gin"

	"residential-energy-intelligence-agent-platform/internal/domain"
	"residential-energy-intelligence-agent-platform/internal/service"
	"residential-energy-intelligence-agent-platform/pkg/apperror"
	"residential-energy-intelligence-agent-platform/pkg/response"
)

type LLMConfigHandler struct {
	service *service.LLMConfigService
}

func NewLLMConfigHandler(service *service.LLMConfigService) *LLMConfigHandler {
	return &LLMConfigHandler{service: service}
}

func (h *LLMConfigHandler) List(c *gin.Context) {
	items, appErr := h.service.List(c.Request.Context())
	if appErr != nil {
		response.Error(c, appErr)
		return
	}
	response.OK(c, gin.H{"items": items})
}

func (h *LLMConfigHandler) Create(c *gin.Context) {
	payload, ok := bindLLMConfigPayload(c)
	if !ok {
		return
	}

	item, appErr := h.service.Create(c.Request.Context(), payload)
	if appErr != nil {
		response.Error(c, appErr)
		return
	}
	response.Success(c, 201, "OK", "success", item)
}

func (h *LLMConfigHandler) Update(c *gin.Context) {
	id, ok := parseUint64Param(c, "id")
	if !ok {
		return
	}

	payload, ok := bindLLMConfigPayload(c)
	if !ok {
		return
	}

	item, appErr := h.service.Update(c.Request.Context(), id, payload)
	if appErr != nil {
		response.Error(c, appErr)
		return
	}
	response.OK(c, item)
}

func (h *LLMConfigHandler) Delete(c *gin.Context) {
	id, ok := parseUint64Param(c, "id")
	if !ok {
		return
	}

	if appErr := h.service.Delete(c.Request.Context(), id); appErr != nil {
		response.Error(c, appErr)
		return
	}
	response.OK(c, gin.H{"id": id})
}

func (h *LLMConfigHandler) SetDefault(c *gin.Context) {
	id, ok := parseUint64Param(c, "id")
	if !ok {
		return
	}

	item, appErr := h.service.SetDefault(c.Request.Context(), id)
	if appErr != nil {
		response.Error(c, appErr)
		return
	}
	response.OK(c, item)
}

func bindLLMConfigPayload(c *gin.Context) (domain.LLMConfigPayload, bool) {
	var payload domain.LLMConfigPayload
	if err := c.ShouldBindJSON(&payload); err != nil {
		response.Error(c, apperror.InvalidRequest("请求体格式错误", map[string]any{
			"error": err.Error(),
		}))
		return domain.LLMConfigPayload{}, false
	}
	return payload, true
}

func parseUint64Param(c *gin.Context, key string) (uint64, bool) {
	raw := c.Param(key)
	id, err := strconv.ParseUint(raw, 10, 64)
	if err != nil || id == 0 {
		response.Error(c, apperror.InvalidRequest("路径参数非法", map[string]any{
			key: raw,
		}))
		return 0, false
	}
	return id, true
}
