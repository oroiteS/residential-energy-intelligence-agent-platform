package service

import (
	"testing"
	"time"

	"residential-energy-intelligence-agent-platform/internal/domain"
)

func TestSelectClassificationWindowPicksLatestCompleteDay(t *testing.T) {
	rows := make([]processedFeatureRow, 0, 192)
	start := time.Date(2026, 4, 1, 0, 0, 0, 0, time.Local)
	for day := 0; day < 2; day++ {
		for slot := 0; slot < 96; slot++ {
			rows = append(rows, processedFeatureRow{
				Timestamp:            start.AddDate(0, 0, day).Add(time.Duration(slot) * 15 * time.Minute),
				Aggregate:            1,
				ActiveApplianceCount: 1,
				BurstEventCount:      0,
			})
		}
	}

	windowRows, windowStart, _, appErr := selectClassificationWindow(rows, 1, "full_dataset", nil, nil)
	if appErr != nil {
		t.Fatalf("selectClassificationWindow() 返回错误: %v", appErr)
	}
	if len(windowRows) != 96 {
		t.Fatalf("窗口点数 = %d, want 96", len(windowRows))
	}
	if got := windowStart.Format("2006-01-02"); got != "2026-04-02" {
		t.Fatalf("windowStart date = %s, want 2026-04-02", got)
	}
}

func TestClassificationRecordDTOReturnsV1Fields(t *testing.T) {
	record := &domain.ClassificationResultRecord{
		ID:             1,
		DatasetID:      2,
		ModelType:      "xgboost",
		PredictedLabel: "day_low_night_high",
		Confidence:     0.9187,
	}

	dto := classificationRecordDTO(record)

	if modelType, _ := dto["model_type"].(string); modelType != "xgboost" {
		t.Fatalf("model_type = %v, want xgboost", dto["model_type"])
	}
	if schemaVersion, _ := dto["schema_version"].(string); schemaVersion != "v1" {
		t.Fatalf("schema_version = %v, want v1", dto["schema_version"])
	}
	if labelDisplayName, _ := dto["label_display_name"].(string); labelDisplayName != "白天低晚上高型" {
		t.Fatalf("label_display_name = %v, want 白天低晚上高型", dto["label_display_name"])
	}
}

func TestNormalizedClassificationModelTypeDefaultsToXGBoost(t *testing.T) {
	if got := normalizedClassificationModelType(""); got != "xgboost" {
		t.Fatalf("normalizedClassificationModelType(\"\") = %s, want xgboost", got)
	}
}
