package handler

import (
	"strconv"

	"github.com/gin-gonic/gin"

	"residential-energy-intelligence-agent-platform/internal/service"
	"residential-energy-intelligence-agent-platform/pkg/apperror"
	"residential-energy-intelligence-agent-platform/pkg/response"
)

type ChatHandler struct {
	service *service.ChatService
}

func NewChatHandler(service *service.ChatService) *ChatHandler {
	return &ChatHandler{service: service}
}

func (h *ChatHandler) CreateSession(c *gin.Context) {
	var payload service.CreateChatSessionInput
	if err := c.ShouldBindJSON(&payload); err != nil {
		response.Error(c, apperror.InvalidRequest("请求体格式错误", map[string]any{"error": err.Error()}))
		return
	}

	data, appErr := h.service.CreateSession(c.Request.Context(), payload)
	if appErr != nil {
		response.Error(c, appErr)
		return
	}
	response.OK(c, data)
}

func (h *ChatHandler) ListSessions(c *gin.Context) {
	page, appErr := parsePositiveIntQuery(c, "page", 1)
	if appErr != nil {
		response.Error(c, appErr)
		return
	}
	pageSize, appErr := parsePositiveIntQuery(c, "page_size", 20)
	if appErr != nil {
		response.Error(c, appErr)
		return
	}

	var datasetID uint64
	if raw := c.Query("dataset_id"); raw != "" {
		value, ok := parseOptionalUint64Query(c, "dataset_id")
		if !ok {
			return
		}
		datasetID = value
	}

	data, serviceErr := h.service.ListSessions(c.Request.Context(), service.ChatSessionListParams{
		Page:      page,
		PageSize:  pageSize,
		DatasetID: datasetID,
	})
	if serviceErr != nil {
		response.Error(c, serviceErr)
		return
	}
	response.OK(c, data)
}

func (h *ChatHandler) DeleteSession(c *gin.Context) {
	id, ok := parseUint64Param(c, "id")
	if !ok {
		return
	}

	if appErr := h.service.DeleteSession(c.Request.Context(), id); appErr != nil {
		response.Error(c, appErr)
		return
	}
	response.OK(c, gin.H{"id": id})
}

func (h *ChatHandler) ListMessages(c *gin.Context) {
	id, ok := parseUint64Param(c, "id")
	if !ok {
		return
	}

	page, appErr := parsePositiveIntQuery(c, "page", 1)
	if appErr != nil {
		response.Error(c, appErr)
		return
	}
	pageSize, appErr := parsePositiveIntQuery(c, "page_size", 50)
	if appErr != nil {
		response.Error(c, appErr)
		return
	}

	data, serviceErr := h.service.ListMessages(c.Request.Context(), id, service.ChatMessageListParams{
		Page:     page,
		PageSize: pageSize,
	})
	if serviceErr != nil {
		response.Error(c, serviceErr)
		return
	}
	response.OK(c, data)
}

func parseOptionalUint64Query(c *gin.Context, key string) (uint64, bool) {
	raw := c.Query(key)
	if raw == "" {
		return 0, true
	}
	value, err := strconv.ParseUint(raw, 10, 64)
	if err != nil {
		response.Error(c, apperror.InvalidRequest("查询参数非法", map[string]any{key: raw}))
		return 0, false
	}
	return value, true
}
