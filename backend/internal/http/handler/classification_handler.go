package handler

import (
	"time"

	"github.com/gin-gonic/gin"

	"residential-energy-intelligence-agent-platform/internal/service"
	"residential-energy-intelligence-agent-platform/pkg/apperror"
	"residential-energy-intelligence-agent-platform/pkg/response"
)

type ClassificationHandler struct {
	service *service.ClassificationService
}

func NewClassificationHandler(service *service.ClassificationService) *ClassificationHandler {
	return &ClassificationHandler{service: service}
}

func (h *ClassificationHandler) Predict(c *gin.Context) {
	id, ok := parseUint64Param(c, "id")
	if !ok {
		return
	}

	var payload service.ClassificationPredictInput
	if err := c.ShouldBindJSON(&payload); err != nil {
		response.Error(c, apperror.InvalidRequest("请求体格式错误", map[string]any{"error": err.Error()}))
		return
	}

	data, appErr := h.service.Predict(c.Request.Context(), id, payload)
	if appErr != nil {
		response.Error(c, appErr)
		return
	}
	response.OK(c, data)
}

func (h *ClassificationHandler) GetLatest(c *gin.Context) {
	id, ok := parseUint64Param(c, "id")
	if !ok {
		return
	}
	data, appErr := h.service.GetLatest(c.Request.Context(), id, c.Query("model_type"))
	if appErr != nil {
		response.Error(c, appErr)
		return
	}
	response.OK(c, data)
}

func (h *ClassificationHandler) List(c *gin.Context) {
	id, ok := parseUint64Param(c, "id")
	if !ok {
		return
	}

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

	data, serviceErr := h.service.List(c.Request.Context(), id, service.ClassificationListParams{
		Page:      page,
		PageSize:  pageSize,
		ModelType: c.Query("model_type"),
	})
	if serviceErr != nil {
		response.Error(c, serviceErr)
		return
	}
	response.OK(c, data)
}

func parseOptionalTime(c *gin.Context, raw string) (*time.Time, *apperror.AppError) {
	if raw == "" {
		return nil, nil
	}
	value, err := time.Parse(time.RFC3339, raw)
	if err != nil {
		return nil, apperror.InvalidRequest("时间格式非法", map[string]any{"value": raw})
	}
	return &value, nil
}
