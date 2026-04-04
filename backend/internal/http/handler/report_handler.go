package handler

import (
	"github.com/gin-gonic/gin"

	"residential-energy-intelligence-agent-platform/internal/service"
	"residential-energy-intelligence-agent-platform/pkg/apperror"
	"residential-energy-intelligence-agent-platform/pkg/response"
)

type ReportHandler struct {
	service *service.ReportService
}

func NewReportHandler(service *service.ReportService) *ReportHandler {
	return &ReportHandler{service: service}
}

func (h *ReportHandler) Export(c *gin.Context) {
	id, ok := parseUint64Param(c, "id")
	if !ok {
		return
	}

	var request struct {
		ReportType string `json:"report_type"`
	}
	if err := c.ShouldBindJSON(&request); err != nil {
		response.Error(c, apperror.InvalidRequest("请求体格式错误", map[string]any{"error": err.Error()}))
		return
	}

	report, appErr := h.service.Export(c.Request.Context(), id, request.ReportType)
	if appErr != nil {
		response.Error(c, appErr)
		return
	}
	response.Success(c, 202, "ACCEPTED", "导出任务已受理", gin.H{"report": report})
}

func (h *ReportHandler) List(c *gin.Context) {
	id, ok := parseUint64Param(c, "id")
	if !ok {
		return
	}

	items, appErr := h.service.List(c.Request.Context(), id)
	if appErr != nil {
		response.Error(c, appErr)
		return
	}
	response.OK(c, gin.H{"items": items})
}

func (h *ReportHandler) Download(c *gin.Context) {
	id, ok := parseUint64Param(c, "id")
	if !ok {
		return
	}

	path, filename, appErr := h.service.GetDownloadPath(c.Request.Context(), id)
	if appErr != nil {
		response.Error(c, appErr)
		return
	}
	c.FileAttachment(path, filename)
}
