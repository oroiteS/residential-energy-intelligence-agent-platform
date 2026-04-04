package service

import (
	"testing"
	"time"
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
