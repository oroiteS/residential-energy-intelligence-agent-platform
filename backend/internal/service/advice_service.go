package service

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"go.uber.org/zap"
	"gorm.io/gorm"

	"residential-energy-intelligence-agent-platform/internal/config"
	"residential-energy-intelligence-agent-platform/internal/domain"
	"residential-energy-intelligence-agent-platform/internal/repository"
	"residential-energy-intelligence-agent-platform/pkg/apperror"
)

type AdviceService struct {
	cfg          *config.AppConfig
	datasetRepo  repository.DatasetRepository
	analysisRepo repository.AnalysisResultRepository
	adviceRepo   repository.EnergyAdviceRepository
	logger       *zap.Logger
}

func NewAdviceService(
	cfg *config.AppConfig,
	datasetRepo repository.DatasetRepository,
	analysisRepo repository.AnalysisResultRepository,
	adviceRepo repository.EnergyAdviceRepository,
	logger *zap.Logger,
) *AdviceService {
	return &AdviceService{
		cfg:          cfg,
		datasetRepo:  datasetRepo,
		analysisRepo: analysisRepo,
		adviceRepo:   adviceRepo,
		logger:       logger,
	}
}

func (s *AdviceService) Generate(ctx context.Context, datasetID uint64) (map[string]any, *apperror.AppError) {
	if s.datasetRepo == nil || s.analysisRepo == nil || s.adviceRepo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持生成节能建议", nil)
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

	analysisRecord, err := s.analysisRepo.GetByDatasetID(ctx, datasetID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, apperror.NotFound("ANALYSIS_NOT_FOUND", "统计分析结果不存在", map[string]any{"dataset_id": datasetID})
		}
		return nil, apperror.Internal(err)
	}

	items := buildRuleAdviceItems(analysisRecord)
	content := domain.AdviceContent{Items: items}
	contentPath, err := s.writeAdviceContent(datasetID, content)
	if err != nil {
		return nil, apperror.Internal(err)
	}

	summary := buildAdviceSummary(items)
	record := &domain.EnergyAdviceRecord{
		DatasetID:   datasetID,
		AdviceType:  "rule",
		ContentPath: contentPath,
		Summary:     &summary,
	}
	if err := s.adviceRepo.Create(ctx, record); err != nil {
		return nil, apperror.Internal(err)
	}

	s.logInfo("规则建议生成完成", zap.Uint64("dataset_id", datasetID), zap.Uint64("advice_id", record.ID))
	return adviceSummaryDTO(record), nil
}

func (s *AdviceService) List(ctx context.Context, datasetID uint64, adviceType string) ([]map[string]any, *apperror.AppError) {
	if s.adviceRepo == nil || s.datasetRepo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持读取节能建议", nil)
	}
	if _, err := s.datasetRepo.GetByID(ctx, datasetID); err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, apperror.NotFound("DATASET_NOT_FOUND", "数据集不存在", map[string]any{"id": datasetID})
		}
		return nil, apperror.Internal(err)
	}

	records, err := s.adviceRepo.ListByDatasetID(ctx, datasetID, strings.TrimSpace(adviceType))
	if err != nil {
		return nil, apperror.Internal(err)
	}

	items := make([]map[string]any, 0, len(records))
	for _, record := range records {
		recordCopy := record
		items = append(items, adviceSummaryDTO(&recordCopy))
	}
	return items, nil
}

func (s *AdviceService) Get(ctx context.Context, id uint64) (map[string]any, *apperror.AppError) {
	if s.adviceRepo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持读取节能建议", nil)
	}

	record, err := s.adviceRepo.GetByID(ctx, id)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, apperror.NotFound("ADVICE_NOT_FOUND", "节能建议不存在", map[string]any{"id": id})
		}
		return nil, apperror.Internal(err)
	}

	content, err := os.ReadFile(record.ContentPath)
	if err != nil {
		return nil, apperror.Internal(err)
	}
	var adviceContent domain.AdviceContent
	if err := json.Unmarshal(content, &adviceContent); err != nil {
		return nil, apperror.Internal(err)
	}

	return map[string]any{
		"advice":  adviceDetailDTO(record),
		"content": adviceContent,
	}, nil
}

func (s *AdviceService) writeAdviceContent(datasetID uint64, content domain.AdviceContent) (string, error) {
	adviceDir := filepath.Join(s.cfg.OutputRootDir, "advices")
	if err := os.MkdirAll(adviceDir, 0o755); err != nil {
		return "", err
	}
	path := filepath.ToSlash(filepath.Join(adviceDir, fmt.Sprintf("dataset_%d_%d.json", datasetID, time.Now().UnixNano())))
	data, err := json.MarshalIndent(content, "", "  ")
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return "", err
	}
	return path, nil
}

func buildRuleAdviceItems(analysis *domain.AnalysisResultRecord) []domain.AdviceItem {
	items := make([]domain.AdviceItem, 0, 5)

	if analysis.PeakRatio >= 0.40 {
		items = append(items, domain.AdviceItem{
			Reason: fmt.Sprintf("峰时占比 %.2f%%，高于建议阈值 40%%", analysis.PeakRatio*100),
			Action: "将洗衣、热水器加热、充电等可延后任务尽量安排到谷时段执行",
		})
	}

	if analysis.ValleyRatio < 0.20 {
		items = append(items, domain.AdviceItem{
			Reason: fmt.Sprintf("谷时占比仅 %.2f%%，低于建议利用水平", analysis.ValleyRatio*100),
			Action: "增加夜间低价时段的设备排程，提升谷时电价利用率",
		})
	}

	if analysis.MaxLoadW >= 3500 && analysis.MaxLoadTime != nil {
		items = append(items, domain.AdviceItem{
			Reason: fmt.Sprintf("最高负荷达到 %.0fW，出现在 %s", analysis.MaxLoadW, analysis.MaxLoadTime.Format("15:04")),
			Action: "避免在该时段同时启用多个高功率设备，分散热水器、厨房设备和烘干类负载",
		})
	}

	if analysis.DailyAvgKWH >= 10 {
		items = append(items, domain.AdviceItem{
			Reason: fmt.Sprintf("日均用电量 %.2fkWh，整体基线偏高", analysis.DailyAvgKWH),
			Action: "优先排查长期待机设备、老旧冰箱和持续运行负载，降低全天基础负荷",
		})
	}

	if len(items) < 3 {
		items = append(items, domain.AdviceItem{
			Reason: "统计结果显示负荷结构相对平稳，仍存在优化空间",
			Action: "保持当前用电规律的同时，定期检查待机负载并优先在低电价时段安排弹性用电任务",
		})
	}

	if len(items) < 3 {
		items = append(items, domain.AdviceItem{
			Reason: "典型日曲线需要持续观察高负荷时段变化",
			Action: "建议每周复查一次统计分析，观察峰谷平占比和最大负荷时刻是否改善",
		})
	}

	if len(items) > 8 {
		return items[:8]
	}
	return items
}

func buildAdviceSummary(items []domain.AdviceItem) string {
	if len(items) == 0 {
		return "保持当前用电结构，持续关注峰谷时段分布"
	}
	return items[0].Action
}

func adviceSummaryDTO(record *domain.EnergyAdviceRecord) map[string]any {
	return map[string]any{
		"id":                record.ID,
		"dataset_id":        record.DatasetID,
		"classification_id": record.ClassificationID,
		"advice_type":       record.AdviceType,
		"summary":           nullableString(record.Summary),
		"created_at":        record.CreatedAt,
	}
}

func adviceDetailDTO(record *domain.EnergyAdviceRecord) map[string]any {
	data := adviceSummaryDTO(record)
	data["content_path"] = record.ContentPath
	return data
}

func (s *AdviceService) logInfo(message string, fields ...zap.Field) {
	if s.logger != nil {
		s.logger.Info(message, fields...)
	}
}
