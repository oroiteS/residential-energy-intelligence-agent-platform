package service

import (
	"testing"

	"residential-energy-intelligence-agent-platform/internal/domain"
)

func TestBuildRuleAdviceItemsReturnsAtLeastThreeItems(t *testing.T) {
	items := buildRuleAdviceItems(&domain.AnalysisResultRecord{
		PeakRatio:   0.45,
		ValleyRatio: 0.10,
		DailyAvgKWH: 12.5,
		MaxLoadW:    4200,
	})

	if len(items) < 3 {
		t.Fatalf("len(items) = %d, want >= 3", len(items))
	}
}
