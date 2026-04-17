package service

import (
	"context"
	"testing"

	"go.uber.org/zap"

	"residential-energy-intelligence-agent-platform/internal/domain"
)

type stubSystemConfigRepository struct {
	records []domain.SystemConfigRecord
}

func (r *stubSystemConfigRepository) List(_ context.Context) ([]domain.SystemConfigRecord, error) {
	return r.records, nil
}

func (r *stubSystemConfigRepository) Upsert(_ context.Context, _ *domain.SystemConfigRecord) error {
	return nil
}

func (r *stubSystemConfigRepository) Delete(_ context.Context, _ string) error {
	return nil
}

func TestSystemConfigServiceGetNormalizesFrozenModelWindow(t *testing.T) {
	repo := &stubSystemConfigRepository{
		records: []domain.SystemConfigRecord{
			{
				ConfigKey:   "model_history_window_config",
				ConfigValue: `{"classification_days":7,"forecast_history_days":7}`,
			},
		},
	}

	service := NewSystemConfigService(nil, repo, zap.NewNop())
	config, err := service.Get(context.Background())
	if err != nil {
		t.Fatalf("Get() 返回错误: %v", err)
	}

	if config.ModelHistoryWindowConfig.ClassificationDays != 1 {
		t.Fatalf("classification_days = %d, want 1", config.ModelHistoryWindowConfig.ClassificationDays)
	}
	if config.ModelHistoryWindowConfig.ForecastHistoryDays != 7 {
		t.Fatalf("forecast_history_days = %d, want 7", config.ModelHistoryWindowConfig.ForecastHistoryDays)
	}
}
