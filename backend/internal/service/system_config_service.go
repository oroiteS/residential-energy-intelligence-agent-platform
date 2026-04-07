package service

import (
	"context"
	"encoding/json"
	"errors"
	"strings"

	"go.uber.org/zap"
	"gorm.io/gorm"

	"residential-energy-intelligence-agent-platform/internal/domain"
	"residential-energy-intelligence-agent-platform/internal/repository"
	"residential-energy-intelligence-agent-platform/pkg/apperror"
)

type SystemConfigService struct {
	repo   repository.SystemConfigRepository
	db     *gorm.DB
	logger *zap.Logger
}

type UpdateSystemConfigInput struct {
	PeakValleyConfig           *domain.PeakValleyConfig         `json:"peak_valley_config"`
	ModelHistoryWindowConfig   *domain.ModelHistoryWindowConfig `json:"model_history_window_config"`
	EnergyAdvicePromptTemplate *string                          `json:"energy_advice_prompt_template"`
	DataUploadDir              *string                          `json:"data_upload_dir"`
	ReportOutputDir            *string                          `json:"report_output_dir"`
}

func NewSystemConfigService(db *gorm.DB, repo repository.SystemConfigRepository, logger *zap.Logger) *SystemConfigService {
	return &SystemConfigService{
		repo:   repo,
		db:     db,
		logger: logger,
	}
}

func (s *SystemConfigService) Get(ctx context.Context) (*domain.SystemRuntimeConfig, error) {
	config := defaultSystemRuntimeConfig()
	if s.repo == nil {
		return config, nil
	}

	records, err := s.repo.List(ctx)
	if err != nil {
		if s.logger != nil {
			s.logger.Warn("读取系统配置失败，已回退默认配置", zap.Error(err))
		}
		if err == gorm.ErrInvalidDB {
			return config, nil
		}
		return config, nil
	}

	for _, record := range records {
		switch record.ConfigKey {
		case "peak_valley_config":
			decodeJSON(record.ConfigValue, &config.PeakValleyConfig, s.logger, record.ConfigKey)
		case "model_history_window_config":
			decodeJSON(record.ConfigValue, &config.ModelHistoryWindowConfig, s.logger, record.ConfigKey)
		case "energy_advice_prompt_template":
			config.EnergyAdvicePromptTemplate = record.ConfigValue
		case "data_upload_dir":
			config.DataUploadDir = record.ConfigValue
		case "report_output_dir":
			config.ReportOutputDir = record.ConfigValue
		}
	}

	normalizeFrozenModelWindow(&config.ModelHistoryWindowConfig)
	return config, nil
}

func defaultSystemRuntimeConfig() *domain.SystemRuntimeConfig {
	return &domain.SystemRuntimeConfig{
		PeakValleyConfig: domain.PeakValleyConfig{
			Peak:   []string{"07:00-11:00", "18:00-23:00"},
			Valley: []string{"23:00-07:00"},
		},
		ModelHistoryWindowConfig: domain.ModelHistoryWindowConfig{
			ClassificationDays:  1,
			ForecastHistoryDays: 3,
		},
		EnergyAdvicePromptTemplate: "这是居民过去{{history_days}}天的实际用电情况、未来一段时间的预测用电情况，以及居民用电行为分类。请基于统计分析结果、历史用电摘要、未来预测摘要和分类结果，给出具体、可执行、可解释的节能建议，并指出关键依据。",
		DataUploadDir:              "./uploads/datasets",
		ReportOutputDir:            "./outputs/reports",
	}
}

func (s *SystemConfigService) Update(ctx context.Context, input UpdateSystemConfigInput) (*domain.SystemRuntimeConfig, *apperror.AppError) {
	if s.db == nil || s.repo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持修改系统配置", nil)
	}

	current, err := s.Get(ctx)
	if err != nil {
		return nil, apperror.Internal(err)
	}

	if input.PeakValleyConfig != nil {
		if err := validatePeakValleyConfig(*input.PeakValleyConfig); err != nil {
			return nil, apperror.Unprocessable("INVALID_REQUEST", err.Error(), nil)
		}
		current.PeakValleyConfig = *input.PeakValleyConfig
	}

	if input.ModelHistoryWindowConfig != nil {
		normalizeFrozenModelWindow(input.ModelHistoryWindowConfig)
		current.ModelHistoryWindowConfig = *input.ModelHistoryWindowConfig
	}

	if input.EnergyAdvicePromptTemplate != nil {
		prompt := strings.TrimSpace(*input.EnergyAdvicePromptTemplate)
		if prompt == "" {
			return nil, apperror.Unprocessable("INVALID_REQUEST", "energy_advice_prompt_template 不能为空", nil)
		}
		current.EnergyAdvicePromptTemplate = prompt
	}

	if input.DataUploadDir != nil {
		dir := strings.TrimSpace(*input.DataUploadDir)
		if dir == "" {
			return nil, apperror.Unprocessable("INVALID_REQUEST", "data_upload_dir 不能为空", nil)
		}
		current.DataUploadDir = dir
	}

	if input.ReportOutputDir != nil {
		dir := strings.TrimSpace(*input.ReportOutputDir)
		if dir == "" {
			return nil, apperror.Unprocessable("INVALID_REQUEST", "report_output_dir 不能为空", nil)
		}
		current.ReportOutputDir = dir
	}

	records, err := buildSystemConfigRecords(current)
	if err != nil {
		return nil, apperror.Internal(err)
	}

	for _, record := range records {
		recordCopy := record
		if err := s.repo.Upsert(ctx, &recordCopy); err != nil {
			return nil, apperror.Internal(err)
		}
	}

	return current, nil
}

func decodeJSON(raw string, target any, logger *zap.Logger, key string) {
	if err := json.Unmarshal([]byte(raw), target); err != nil && logger != nil {
		logger.Warn("系统配置 JSON 解析失败", zap.String("config_key", key), zap.Error(err))
	}
}

func buildSystemConfigRecords(config *domain.SystemRuntimeConfig) ([]domain.SystemConfigRecord, error) {
	if config == nil {
		return nil, errors.New("system config 不能为空")
	}

	peakValleyJSON, err := json.Marshal(config.PeakValleyConfig)
	if err != nil {
		return nil, err
	}

	modelWindow := config.ModelHistoryWindowConfig
	normalizeFrozenModelWindow(&modelWindow)
	modelWindowJSON, err := json.Marshal(modelWindow)
	if err != nil {
		return nil, err
	}

	records := []domain.SystemConfigRecord{
		{
			ConfigKey:   "peak_valley_config",
			ConfigValue: string(peakValleyJSON),
			Description: stringPtr("峰谷时段配置（JSON 格式）"),
		},
		{
			ConfigKey:   "model_history_window_config",
			ConfigValue: string(modelWindowJSON),
			Description: stringPtr("模型历史窗口配置（分类/预测）"),
		},
		{
			ConfigKey:   "energy_advice_prompt_template",
			ConfigValue: config.EnergyAdvicePromptTemplate,
			Description: stringPtr("节能建议智能体提示词模板"),
		},
		{
			ConfigKey:   "data_upload_dir",
			ConfigValue: config.DataUploadDir,
			Description: stringPtr("数据集上传目录"),
		},
		{
			ConfigKey:   "report_output_dir",
			ConfigValue: config.ReportOutputDir,
			Description: stringPtr("报告输出目录"),
		},
	}

	return records, nil
}

func validatePeakValleyConfig(config domain.PeakValleyConfig) error {
	if len(config.Peak) == 0 {
		return errors.New("peak_valley_config.peak 至少需要一个时间段")
	}
	if len(config.Valley) == 0 {
		return errors.New("peak_valley_config.valley 至少需要一个时间段")
	}

	for _, period := range append(config.Peak, config.Valley...) {
		if strings.TrimSpace(period) == "" {
			return errors.New("peak_valley_config 中不允许空时间段")
		}
	}
	return nil
}

func normalizeFrozenModelWindow(config *domain.ModelHistoryWindowConfig) {
	if config == nil {
		return
	}
	config.ClassificationDays = 1
	config.ForecastHistoryDays = 3
}

func stringPtr(value string) *string {
	return &value
}
