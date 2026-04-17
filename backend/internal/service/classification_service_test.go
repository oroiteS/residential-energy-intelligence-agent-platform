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

func TestSelectClassificationWindowsReturnsAllCompleteDays(t *testing.T) {
	rows := make([]processedFeatureRow, 0, 288)
	start := time.Date(2026, 4, 1, 0, 0, 0, 0, time.Local)
	for day := 0; day < 3; day++ {
		for slot := 0; slot < 96; slot++ {
			rows = append(rows, processedFeatureRow{
				Timestamp:            start.AddDate(0, 0, day).Add(time.Duration(slot) * 15 * time.Minute),
				Aggregate:            1,
				ActiveApplianceCount: 1,
				BurstEventCount:      0,
			})
		}
	}

	windows, appErr := selectClassificationWindows(rows, 1, "full_dataset", nil, nil)
	if appErr != nil {
		t.Fatalf("selectClassificationWindows() 返回错误: %v", appErr)
	}
	if len(windows) != 3 {
		t.Fatalf("窗口数量 = %d, want 3", len(windows))
	}
	if got := windows[0].Start.Format("2006-01-02"); got != "2026-04-01" {
		t.Fatalf("首个窗口日期 = %s, want 2026-04-01", got)
	}
	if got := windows[2].Start.Format("2006-01-02"); got != "2026-04-03" {
		t.Fatalf("最后窗口日期 = %s, want 2026-04-03", got)
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
	if labelDisplayName, _ := dto["label_display_name"].(string); labelDisplayName != "晚上高峰型" {
		t.Fatalf("label_display_name = %v, want 晚上高峰型", dto["label_display_name"])
	}
}

func TestNormalizeClassificationLabelSupportsLegacyValues(t *testing.T) {
	if got := normalizeClassificationLabel("daytime_active"); got != "afternoon_peak" {
		t.Fatalf("normalizeClassificationLabel(daytime_active) = %s, want afternoon_peak", got)
	}
	if got := normalizeClassificationLabel("all_day_high"); got != "morning_peak" {
		t.Fatalf("normalizeClassificationLabel(all_day_high) = %s, want morning_peak", got)
	}
}

func TestNormalizedClassificationModelTypeDefaultsToXGBoost(t *testing.T) {
	if got := normalizedClassificationModelType(""); got != "xgboost" {
		t.Fatalf("normalizedClassificationModelType(\"\") = %s, want xgboost", got)
	}
}

func TestLatestClassificationRecordsByWindowDeduplicatesSameDay(t *testing.T) {
	dayStart := time.Date(2026, 4, 2, 0, 0, 0, 0, time.Local)
	dayEnd := dayStart.Add(95 * 15 * time.Minute)
	createdAtOld := dayStart.Add(24 * time.Hour)
	createdAtNew := createdAtOld.Add(time.Hour)

	records := []domain.ClassificationResultRecord{
		{
			ID:          1,
			WindowStart: &dayStart,
			WindowEnd:   &dayEnd,
			CreatedAt:   &createdAtNew,
		},
		{
			ID:          2,
			WindowStart: &dayStart,
			WindowEnd:   &dayEnd,
			CreatedAt:   &createdAtOld,
		},
	}

	unique := latestClassificationRecordsByWindow(records)
	if len(unique) != 1 {
		t.Fatalf("去重后数量 = %d, want 1", len(unique))
	}
	if unique[0].ID != 1 {
		t.Fatalf("保留记录 ID = %d, want 1", unique[0].ID)
	}
}
