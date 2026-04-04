package handler

import (
	"github.com/gin-gonic/gin"

	"residential-energy-intelligence-agent-platform/internal/service"
	"residential-energy-intelligence-agent-platform/pkg/apperror"
	"residential-energy-intelligence-agent-platform/pkg/response"
)

type ForecastHandler struct {
	service *service.ForecastService
}

func NewForecastHandler(service *service.ForecastService) *ForecastHandler {
	return &ForecastHandler{service: service}
}

func (h *ForecastHandler) Predict(c *gin.Context) {
	id, ok := parseUint64Param(c, "id")
	if !ok {
		return
	}

	var payload service.ForecastPredictInput
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

func (h *ForecastHandler) List(c *gin.Context) {
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

	data, serviceErr := h.service.List(c.Request.Context(), id, service.ForecastListParams{
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

func (h *ForecastHandler) Get(c *gin.Context) {
	id, ok := parseUint64Param(c, "forecast_id")
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

func (h *ForecastHandler) Backtest(c *gin.Context) {
	id, ok := parseUint64Param(c, "id")
	if !ok {
		return
	}

	var payload service.ForecastBacktestInput
	if err := c.ShouldBindJSON(&payload); err != nil {
		response.Error(c, apperror.InvalidRequest("请求体格式错误", map[string]any{"error": err.Error()}))
		return
	}

	data, appErr := h.service.Backtest(c.Request.Context(), id, payload)
	if appErr != nil {
		response.Error(c, appErr)
		return
	}
	response.OK(c, data)
}
