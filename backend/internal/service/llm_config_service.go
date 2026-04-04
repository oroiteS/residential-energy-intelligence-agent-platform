package service

import (
	"context"
	"errors"
	"strconv"
	"strings"

	"go.uber.org/zap"
	"gorm.io/gorm"

	"residential-energy-intelligence-agent-platform/internal/domain"
	"residential-energy-intelligence-agent-platform/internal/repository"
	"residential-energy-intelligence-agent-platform/pkg/apperror"
)

type LLMConfigService struct {
	db               *gorm.DB
	llmRepo          repository.LLMConfigRepository
	systemConfigRepo repository.SystemConfigRepository
	logger           *zap.Logger
}

func NewLLMConfigService(
	db *gorm.DB,
	llmRepo repository.LLMConfigRepository,
	systemConfigRepo repository.SystemConfigRepository,
	logger *zap.Logger,
) *LLMConfigService {
	return &LLMConfigService{
		db:               db,
		llmRepo:          llmRepo,
		systemConfigRepo: systemConfigRepo,
		logger:           logger,
	}
}

func (s *LLMConfigService) List(ctx context.Context) ([]domain.LLMConfigSummary, *apperror.AppError) {
	if s.db == nil || s.llmRepo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持读取 LLM 配置", nil)
	}

	records, err := s.llmRepo.List(ctx)
	if err != nil {
		return nil, apperror.Internal(err)
	}

	items := make([]domain.LLMConfigSummary, 0, len(records))
	for _, record := range records {
		items = append(items, toLLMConfigSummary(record))
	}
	return items, nil
}

func (s *LLMConfigService) Create(ctx context.Context, payload domain.LLMConfigPayload) (*domain.LLMConfigSummary, *apperror.AppError) {
	if s.db == nil || s.llmRepo == nil || s.systemConfigRepo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持创建 LLM 配置", nil)
	}
	if appErr := validateLLMConfigPayload(payload); appErr != nil {
		return nil, appErr
	}

	var created domain.LLMConfigRecord
	if err := s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		llmRepo := repository.NewLLMConfigRepository(tx)
		systemConfigRepo := repository.NewSystemConfigRepository(tx)

		total, err := llmRepo.Count(ctx)
		if err != nil {
			return err
		}

		isDefault := payload.IsDefault || total == 0
		if isDefault {
			if err := llmRepo.ClearDefault(ctx); err != nil {
				return err
			}
		}

		created = domain.LLMConfigRecord{
			Name:           strings.TrimSpace(payload.Name),
			BaseURL:        strings.TrimSpace(payload.BaseURL),
			APIKey:         strings.TrimSpace(payload.APIKey),
			ModelName:      strings.TrimSpace(payload.ModelName),
			Temperature:    payload.Temperature,
			TimeoutSeconds: payload.TimeoutSeconds,
			IsDefault:      isDefault,
		}

		if err := llmRepo.Create(ctx, &created); err != nil {
			return err
		}

		if isDefault {
			if err := upsertDefaultLLMID(ctx, systemConfigRepo, created.ID); err != nil {
				return err
			}
		}

		return nil
	}); err != nil {
		return nil, apperror.Internal(err)
	}

	s.logInfo("创建 LLM 配置成功", zap.Uint64("llm_config_id", created.ID))
	summary := toLLMConfigSummary(created)
	return &summary, nil
}

func (s *LLMConfigService) Update(ctx context.Context, id uint64, payload domain.LLMConfigPayload) (*domain.LLMConfigSummary, *apperror.AppError) {
	if s.db == nil || s.llmRepo == nil || s.systemConfigRepo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持更新 LLM 配置", nil)
	}
	if appErr := validateLLMConfigPayload(payload); appErr != nil {
		return nil, appErr
	}

	var updated domain.LLMConfigRecord
	if err := s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		llmRepo := repository.NewLLMConfigRepository(tx)
		systemConfigRepo := repository.NewSystemConfigRepository(tx)

		record, err := llmRepo.GetByID(ctx, id)
		if err != nil {
			return err
		}

		if record.IsDefault && !payload.IsDefault {
			return apperror.Conflict("INVALID_REQUEST", "默认配置不能直接取消，请先切换默认项", map[string]any{
				"id": id,
			})
		}

		record.Name = strings.TrimSpace(payload.Name)
		record.BaseURL = strings.TrimSpace(payload.BaseURL)
		record.APIKey = strings.TrimSpace(payload.APIKey)
		record.ModelName = strings.TrimSpace(payload.ModelName)
		record.Temperature = payload.Temperature
		record.TimeoutSeconds = payload.TimeoutSeconds
		record.IsDefault = payload.IsDefault

		if payload.IsDefault {
			if err := llmRepo.ClearDefault(ctx); err != nil {
				return err
			}
			record.IsDefault = true
		}

		if err := llmRepo.Update(ctx, record); err != nil {
			return err
		}

		if record.IsDefault {
			if err := upsertDefaultLLMID(ctx, systemConfigRepo, record.ID); err != nil {
				return err
			}
		}

		updated = *record
		return nil
	}); err != nil {
		return nil, mapLLMConfigError(err, id)
	}

	s.logInfo("更新 LLM 配置成功", zap.Uint64("llm_config_id", id))
	summary := toLLMConfigSummary(updated)
	return &summary, nil
}

func (s *LLMConfigService) Delete(ctx context.Context, id uint64) *apperror.AppError {
	if s.db == nil || s.llmRepo == nil || s.systemConfigRepo == nil {
		return apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持删除 LLM 配置", nil)
	}

	if err := s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		llmRepo := repository.NewLLMConfigRepository(tx)

		record, err := llmRepo.GetByID(ctx, id)
		if err != nil {
			return err
		}
		if record.IsDefault {
			return apperror.Conflict("INVALID_REQUEST", "默认配置删除前请先切换默认项", map[string]any{
				"id": id,
			})
		}

		return llmRepo.Delete(ctx, id)
	}); err != nil {
		return mapLLMConfigError(err, id)
	}

	s.logInfo("删除 LLM 配置成功", zap.Uint64("llm_config_id", id))
	return nil
}

func (s *LLMConfigService) SetDefault(ctx context.Context, id uint64) (*domain.LLMConfigSummary, *apperror.AppError) {
	if s.db == nil || s.llmRepo == nil || s.systemConfigRepo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持设置默认 LLM 配置", nil)
	}

	var updated domain.LLMConfigRecord
	if err := s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		llmRepo := repository.NewLLMConfigRepository(tx)
		systemConfigRepo := repository.NewSystemConfigRepository(tx)
		record, err := llmRepo.GetByID(ctx, id)
		if err != nil {
			return err
		}

		if err := llmRepo.ClearDefault(ctx); err != nil {
			return err
		}

		record.IsDefault = true
		if err := llmRepo.Update(ctx, record); err != nil {
			return err
		}

		if err := upsertDefaultLLMID(ctx, systemConfigRepo, record.ID); err != nil {
			return err
		}

		updated = *record
		return nil
	}); err != nil {
		return nil, mapLLMConfigError(err, id)
	}

	s.logInfo("设置默认 LLM 配置成功", zap.Uint64("llm_config_id", id))
	summary := toLLMConfigSummary(updated)
	return &summary, nil
}

func validateLLMConfigPayload(payload domain.LLMConfigPayload) *apperror.AppError {
	switch {
	case strings.TrimSpace(payload.Name) == "":
		return apperror.Unprocessable("INVALID_REQUEST", "name 不能为空", nil)
	case strings.TrimSpace(payload.BaseURL) == "":
		return apperror.Unprocessable("INVALID_REQUEST", "base_url 不能为空", nil)
	case strings.TrimSpace(payload.APIKey) == "":
		return apperror.Unprocessable("INVALID_REQUEST", "api_key 不能为空", nil)
	case strings.TrimSpace(payload.ModelName) == "":
		return apperror.Unprocessable("INVALID_REQUEST", "model_name 不能为空", nil)
	case payload.Temperature < 0 || payload.Temperature > 2:
		return apperror.Unprocessable("INVALID_REQUEST", "temperature 必须在 0 到 2 之间", nil)
	case payload.TimeoutSeconds <= 0:
		return apperror.Unprocessable("INVALID_REQUEST", "timeout_seconds 必须大于 0", nil)
	default:
		return nil
	}
}

func toLLMConfigSummary(record domain.LLMConfigRecord) domain.LLMConfigSummary {
	return domain.LLMConfigSummary{
		ID:             record.ID,
		Name:           record.Name,
		BaseURL:        record.BaseURL,
		ModelName:      record.ModelName,
		Temperature:    record.Temperature,
		TimeoutSeconds: record.TimeoutSeconds,
		IsDefault:      record.IsDefault,
		CreatedAt:      record.CreatedAt,
		UpdatedAt:      record.UpdatedAt,
	}
}

func mapLLMConfigError(err error, id uint64) *apperror.AppError {
	if err == nil {
		return nil
	}

	var appErr *apperror.AppError
	if ok := errorAs(err, &appErr); ok && appErr != nil {
		return appErr
	}

	if errors.Is(err, gorm.ErrRecordNotFound) {
		return apperror.NotFound("LLM_CONFIG_NOT_FOUND", "LLM 配置不存在", map[string]any{
			"id": id,
		})
	}

	return apperror.Internal(err)
}

func upsertDefaultLLMID(ctx context.Context, repo repository.SystemConfigRepository, id uint64) error {
	return repo.Upsert(ctx, &domain.SystemConfigRecord{
		ConfigKey:   "default_llm_id",
		ConfigValue: strconv.FormatUint(id, 10),
		Description: stringPtr("默认 LLM 配置 ID"),
	})
}

func (s *LLMConfigService) logInfo(message string, fields ...zap.Field) {
	if s.logger != nil {
		s.logger.Info(message, fields...)
	}
}

func errorAs(err error, target **apperror.AppError) bool {
	appErr, ok := err.(*apperror.AppError)
	if !ok {
		return false
	}
	*target = appErr
	return true
}
