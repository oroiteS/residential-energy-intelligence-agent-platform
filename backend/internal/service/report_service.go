package service

import (
	"context"
	"errors"
	"go.uber.org/zap"
	"gorm.io/gorm"
	"os"
	"path/filepath"
	"strings"

	"residential-energy-intelligence-agent-platform/internal/config"
	"residential-energy-intelligence-agent-platform/internal/domain"
	"residential-energy-intelligence-agent-platform/internal/repository"
	"residential-energy-intelligence-agent-platform/pkg/apperror"
)

type ReportService struct {
	cfg          *config.AppConfig
	datasetRepo  repository.DatasetRepository
	analysisRepo repository.AnalysisResultRepository
	adviceRepo   repository.EnergyAdviceRepository
	reportRepo   repository.ReportRepository
	agentService *AgentService
	logger       *zap.Logger
}

func NewReportService(
	cfg *config.AppConfig,
	datasetRepo repository.DatasetRepository,
	analysisRepo repository.AnalysisResultRepository,
	adviceRepo repository.EnergyAdviceRepository,
	reportRepo repository.ReportRepository,
	agentService *AgentService,
	logger *zap.Logger,
) *ReportService {
	return &ReportService{
		cfg:          cfg,
		datasetRepo:  datasetRepo,
		analysisRepo: analysisRepo,
		adviceRepo:   adviceRepo,
		reportRepo:   reportRepo,
		agentService: agentService,
		logger:       logger,
	}
}

func (s *ReportService) Export(ctx context.Context, datasetID uint64, reportType string) (map[string]any, *apperror.AppError) {
	if s.datasetRepo == nil || s.analysisRepo == nil || s.adviceRepo == nil || s.reportRepo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持导出报告", nil)
	}
	if strings.TrimSpace(reportType) == "" {
		reportType = "pdf"
	}
	if reportType != "pdf" {
		return nil, apperror.Unprocessable("INVALID_REQUEST", "当前仅支持 pdf 报告导出", nil)
	}

	dataset, err := s.datasetRepo.GetByID(ctx, datasetID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, apperror.NotFound("DATASET_NOT_FOUND", "数据集不存在", map[string]any{"id": datasetID})
		}
		return nil, apperror.Internal(err)
	}
	if dataset.Status != "ready" {
		return nil, apperror.Conflict("DATASET_NOT_READY", "数据集尚未处理完成", map[string]any{"id": datasetID})
	}

	analysis, err := s.analysisRepo.GetByDatasetID(ctx, datasetID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, apperror.NotFound("ANALYSIS_NOT_FOUND", "统计分析结果不存在", map[string]any{"dataset_id": datasetID})
		}
		return nil, apperror.Internal(err)
	}

	advices, err := s.adviceRepo.ListByDatasetID(ctx, datasetID, "rule")
	if err != nil {
		return nil, apperror.Internal(err)
	}

	var (
		filePath string
		fileSize uint64
	)
	if s.agentService == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "智能体服务未配置，暂不支持 PDF 导出", nil)
	}
	summary, appErr := s.agentService.GenerateReportSummary(ctx, dataset, analysis)
	if appErr != nil {
		return nil, appErr
	}
	filePath, fileSize, err = s.writePDFReport(ctx, dataset, analysis, advices, summary)
	if err != nil {
		return nil, apperror.ServiceUnavailable("EXPORT_FAILED", "报告导出失败", map[string]any{
			"error":       err.Error(),
			"report_type": reportType,
		})
	}

	record := &domain.ReportRecord{
		DatasetID:  datasetID,
		ReportType: reportType,
		FilePath:   filePath,
		FileSize:   fileSize,
	}
	if err := s.reportRepo.Create(ctx, record); err != nil {
		return nil, apperror.Internal(err)
	}

	s.logInfo("报告导出完成", zap.Uint64("dataset_id", datasetID), zap.Uint64("report_id", record.ID), zap.String("report_type", reportType))
	return reportDTO(record), nil
}

func (s *ReportService) List(ctx context.Context, datasetID uint64) ([]map[string]any, *apperror.AppError) {
	if s.reportRepo == nil || s.datasetRepo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持读取报告", nil)
	}
	if _, err := s.datasetRepo.GetByID(ctx, datasetID); err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, apperror.NotFound("DATASET_NOT_FOUND", "数据集不存在", map[string]any{"id": datasetID})
		}
		return nil, apperror.Internal(err)
	}

	records, err := s.reportRepo.ListByDatasetID(ctx, datasetID)
	if err != nil {
		return nil, apperror.Internal(err)
	}

	items := make([]map[string]any, 0, len(records))
	for _, record := range records {
		recordCopy := record
		items = append(items, reportDTO(&recordCopy))
	}
	return items, nil
}

func (s *ReportService) GetDownloadPath(ctx context.Context, reportID uint64) (string, string, *apperror.AppError) {
	if s.reportRepo == nil {
		return "", "", apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持下载报告", nil)
	}

	record, err := s.reportRepo.GetByID(ctx, reportID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return "", "", apperror.NotFound("REPORT_NOT_FOUND", "报告不存在", map[string]any{"id": reportID})
		}
		return "", "", apperror.Internal(err)
	}

	if _, err := os.Stat(record.FilePath); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", "", apperror.NotFound("REPORT_NOT_FOUND", "报告文件不存在", map[string]any{"id": reportID})
		}
		return "", "", apperror.Internal(err)
	}

	return record.FilePath, filepath.Base(record.FilePath), nil
}

func reportDTO(record *domain.ReportRecord) map[string]any {
	return map[string]any{
		"id":          record.ID,
		"dataset_id":  record.DatasetID,
		"report_type": record.ReportType,
		"file_path":   record.FilePath,
		"file_size":   record.FileSize,
		"created_at":  record.CreatedAt,
	}
}

func (s *ReportService) logInfo(message string, fields ...zap.Field) {
	if s.logger != nil {
		s.logger.Info(message, fields...)
	}
}
