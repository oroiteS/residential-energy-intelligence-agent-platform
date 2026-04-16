package service

import (
	"testing"
	"time"

	"residential-energy-intelligence-agent-platform/internal/domain"
)

func TestBuildForecastSummaryUsesSMAPECompatibleMetricsShape(t *testing.T) {
	start := time.Date(2026, 4, 2, 0, 0, 0, 0, time.Local)
	series := make([]domain.ForecastSeriesPoint, 0, 96)
	for index := 0; index < 96; index++ {
		series = append(series, domain.ForecastSeriesPoint{
			Timestamp: start.Add(time.Duration(index) * 15 * time.Minute),
			Predicted: 1 + float64(index)/100,
		})
	}

	summary := buildForecastSummary(start, start.Add(95*15*time.Minute), "15min", series, domain.PeakValleyConfig{
		Peak:   []string{"07:00-11:00", "18:00-23:00"},
		Valley: []string{"23:00-07:00"},
	})

	if summary.Granularity != "15min" {
		t.Fatalf("Granularity = %s, want 15min", summary.Granularity)
	}
	if summary.PredictedAvgLoadW <= 0 {
		t.Fatalf("PredictedAvgLoadW = %f, want > 0", summary.PredictedAvgLoadW)
	}
	if summary.PredictedPeakLoadW <= 0 {
		t.Fatalf("PredictedPeakLoadW = %f, want > 0", summary.PredictedPeakLoadW)
	}
	if len(summary.ForecastPeakPeriods) == 0 {
		t.Fatalf("ForecastPeakPeriods 为空，want 至少一个高负荷时段")
	}
}

func TestExpected15MinutePointsForSingleDay(t *testing.T) {
	start := time.Date(2026, 4, 2, 0, 0, 0, 0, time.Local)
	end := start.Add(95 * 15 * time.Minute)
	if got := expected15MinutePoints(start, end); got != 96 {
		t.Fatalf("expected15MinutePoints() = %d, want 96", got)
	}
}

func TestNormalizeForecastRequestSupportsTFT(t *testing.T) {
	modelType, granularity, appErr := normalizeForecastRequest("tft", "15min")
	if appErr != nil {
		t.Fatalf("normalizeForecastRequest() returned unexpected error: %v", appErr)
	}
	if modelType != "tft" {
		t.Fatalf("modelType = %s, want tft", modelType)
	}
	if granularity != "15min" {
		t.Fatalf("granularity = %s, want 15min", granularity)
	}
}

func TestNormalizeForecastRequestMapsLegacyModelsToTFT(t *testing.T) {
	for _, legacyModel := range []string{"", "lstm", "transformer", "transformer_encoder_direct"} {
		modelType, _, appErr := normalizeForecastRequest(legacyModel, "15min")
		if appErr != nil {
			t.Fatalf("normalizeForecastRequest(%q) returned unexpected error: %v", legacyModel, appErr)
		}
		if modelType != "tft" {
			t.Fatalf("normalizeForecastRequest(%q) modelType = %s, want tft", legacyModel, modelType)
		}
	}
}

func TestNormalizeForecastRequestRejectsUnknownModel(t *testing.T) {
	_, _, appErr := normalizeForecastRequest("gru", "15min")
	if appErr == nil {
		t.Fatalf("normalizeForecastRequest() error = nil, want invalid model error")
	}
}

func TestValidateFutureForecastWindowSupportsThreeFutureDays(t *testing.T) {
	datasetEnd := time.Date(2026, 4, 7, 23, 45, 0, 0, time.Local)
	start := time.Date(2026, 4, 10, 0, 0, 0, 0, time.Local)
	end := start.Add(95 * 15 * time.Minute)

	dayOffset, appErr := validateFutureForecastWindow(datasetEnd, start, end)
	if appErr != nil {
		t.Fatalf("validateFutureForecastWindow() returned unexpected error: %v", appErr)
	}
	if dayOffset != 3 {
		t.Fatalf("dayOffset = %d, want 3", dayOffset)
	}
}

func TestValidateFutureForecastWindowRejectsOutOfRangeDay(t *testing.T) {
	datasetEnd := time.Date(2026, 4, 7, 23, 45, 0, 0, time.Local)
	start := time.Date(2026, 4, 11, 0, 0, 0, 0, time.Local)
	end := start.Add(95 * 15 * time.Minute)

	_, appErr := validateFutureForecastWindow(datasetEnd, start, end)
	if appErr == nil {
		t.Fatal("validateFutureForecastWindow() error = nil, want out of range error")
	}
}

func TestBuildForecastSeriesClampsNegativeValues(t *testing.T) {
	start := time.Date(2026, 4, 8, 0, 0, 0, 0, time.Local)
	series := buildForecastSeries(start, []float64{-12.5, 18.0})

	if got := series[0].Predicted; got != 0 {
		t.Fatalf("series[0].Predicted = %f, want 0", got)
	}
	if got := series[1].Predicted; got != 18.0 {
		t.Fatalf("series[1].Predicted = %f, want 18.0", got)
	}
}
