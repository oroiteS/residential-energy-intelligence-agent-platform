package service

import (
	"testing"
	"time"

	"residential-energy-intelligence-agent-platform/internal/domain"
)

func TestMinuteInPeriodSupportsCrossDayRange(t *testing.T) {
	if !minuteInPeriod(23*60+30, "23:00-07:00") {
		t.Fatal("23:30 应命中跨天谷时段")
	}
	if !minuteInPeriod(6*60+30, "23:00-07:00") {
		t.Fatal("06:30 应命中跨天谷时段")
	}
	if minuteInPeriod(12*60, "23:00-07:00") {
		t.Fatal("12:00 不应命中跨天谷时段")
	}
}

func TestComputeAnalysisBuildsExpectedSummary(t *testing.T) {
	service := &AnalysisService{}
	rows := []analysisRow{
		{Timestamp: time.Date(2026, 4, 1, 8, 0, 0, 0, time.Local), Aggregate: 4000.0},
		{Timestamp: time.Date(2026, 4, 1, 23, 15, 0, 0, time.Local), Aggregate: 8000.0},
	}

	detail, record, err := service.computeAnalysis(1, rows, domain.PeakValleyConfig{
		Peak:   []string{"08:00-18:00"},
		Valley: []string{"23:00-07:00"},
	})
	if err != nil {
		t.Fatalf("computeAnalysis() 返回错误: %v", err)
	}

	if detail.Summary.TotalKWH != 3 {
		t.Fatalf("total_kwh = %v, want 3", detail.Summary.TotalKWH)
	}
	if detail.Summary.PeakKWH != 1 {
		t.Fatalf("peak_kwh = %v, want 1", detail.Summary.PeakKWH)
	}
	if detail.Summary.ValleyKWH != 2 {
		t.Fatalf("valley_kwh = %v, want 2", detail.Summary.ValleyKWH)
	}
	if detail.Summary.FlatKWH != 0 {
		t.Fatalf("flat_kwh = %v, want 0", detail.Summary.FlatKWH)
	}
	if record.DatasetID != 1 {
		t.Fatalf("record.DatasetID = %d, want 1", record.DatasetID)
	}
}
