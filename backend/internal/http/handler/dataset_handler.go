package handler

import (
	"strconv"

	"github.com/gin-gonic/gin"

	"residential-energy-intelligence-agent-platform/internal/service"
	"residential-energy-intelligence-agent-platform/pkg/apperror"
	"residential-energy-intelligence-agent-platform/pkg/response"
)

type DatasetHandler struct {
	service *service.DatasetService
}

func NewDatasetHandler(service *service.DatasetService) *DatasetHandler {
	return &DatasetHandler{service: service}
}

func (h *DatasetHandler) Import(c *gin.Context) {
	file, err := c.FormFile("file")
	if err != nil {
		response.Error(c, apperror.InvalidRequest("缺少上传文件", nil))
		return
	}

	data, appErr := h.service.Import(c.Request.Context(), service.ImportDatasetInput{
		Name:          c.PostForm("name"),
		Description:   c.PostForm("description"),
		HouseholdID:   c.PostForm("household_id"),
		Unit:          c.PostForm("unit"),
		ColumnMapping: c.PostForm("column_mapping"),
		File:          file,
	})
	if appErr != nil {
		response.Error(c, appErr)
		return
	}

	response.Success(c, 202, "ACCEPTED", "导入任务已受理", gin.H{"dataset": data})
}

func (h *DatasetHandler) List(c *gin.Context) {
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

	result, serviceErr := h.service.List(c.Request.Context(), service.DatasetListParams{
		Page:     page,
		PageSize: pageSize,
		Status:   c.Query("status"),
		Keyword:  c.Query("keyword"),
	})
	if serviceErr != nil {
		response.Error(c, serviceErr)
		return
	}
	response.OK(c, result)
}

func (h *DatasetHandler) Get(c *gin.Context) {
	id, ok := parseUint64Param(c, "id")
	if !ok {
		return
	}

	result, appErr := h.service.GetDetail(c.Request.Context(), id)
	if appErr != nil {
		response.Error(c, appErr)
		return
	}
	response.OK(c, result)
}

func (h *DatasetHandler) Delete(c *gin.Context) {
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

func parsePositiveIntQuery(c *gin.Context, key string, defaultValue int) (int, *apperror.AppError) {
	raw := c.Query(key)
	if raw == "" {
		return defaultValue, nil
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value <= 0 {
		return 0, apperror.InvalidRequest("查询参数非法", map[string]any{key: raw})
	}
	return value, nil
}
