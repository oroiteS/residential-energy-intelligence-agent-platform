package service

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"go.uber.org/zap"
	"gorm.io/gorm"

	"residential-energy-intelligence-agent-platform/internal/config"
	"residential-energy-intelligence-agent-platform/internal/domain"
	"residential-energy-intelligence-agent-platform/internal/integration/modelclient"
	"residential-energy-intelligence-agent-platform/internal/repository"
	"residential-energy-intelligence-agent-platform/pkg/apperror"
)

type ForecastService struct {
	cfg                 *config.AppConfig
	datasetRepo         repository.DatasetRepository
	forecastRepo        repository.ForecastResultRepository
	systemConfigService *SystemConfigService
	modelClient         modelclient.Client
	logger              *zap.Logger
}

type ForecastPredictInput struct {
	ModelType     string    `json:"model_type"`
	Granularity   string    `json:"granularity"`
	ForecastStart time.Time `json:"forecast_start"`
	ForecastEnd   time.Time `json:"forecast_end"`
	ForceRefresh  bool      `json:"force_refresh"`
}

type ForecastListParams struct {
	Page      int
	PageSize  int
	ModelType string
}

type ForecastBacktestInput struct {
	ModelType     string    `json:"model_type"`
	Granularity   string    `json:"granularity"`
	BacktestStart time.Time `json:"backtest_start"`
	BacktestEnd   time.Time `json:"backtest_end"`
}

func NewForecastService(
	cfg *config.AppConfig,
	datasetRepo repository.DatasetRepository,
	forecastRepo repository.ForecastResultRepository,
	systemConfigService *SystemConfigService,
	modelClient modelclient.Client,
	logger *zap.Logger,
) *ForecastService {
	return &ForecastService{
		cfg:                 cfg,
		datasetRepo:         datasetRepo,
		forecastRepo:        forecastRepo,
		systemConfigService: systemConfigService,
		modelClient:         modelClient,
		logger:              logger,
	}
}

func (s *ForecastService) Predict(ctx context.Context, datasetID uint64, input ForecastPredictInput) (map[string]any, *apperror.AppError) {
	if s.datasetRepo == nil || s.forecastRepo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持时序预测", nil)
	}
	if s.modelClient == nil {
		return nil, apperror.ServiceUnavailable("MODEL_SERVICE_UNAVAILABLE", "模型服务未初始化", nil)
	}

	modelType, granularity, appErr := normalizeForecastRequest(input.ModelType, input.Granularity)
	if appErr != nil {
		return nil, appErr
	}
	if err := validateForecastRange(input.ForecastStart, input.ForecastEnd); err != nil {
		return nil, apperror.Unprocessable("INVALID_REQUEST", err.Error(), nil)
	}

	dataset, appErr := getReadyDatasetRecord(ctx, s.datasetRepo, datasetID)
	if appErr != nil {
		return nil, appErr
	}
	rows, err := readProcessedFeatureRows(*dataset.ProcessedFilePath)
	if err != nil {
		return nil, apperror.Internal(err)
	}

	runtimeConfig, err := s.systemConfigService.Get(ctx)
	if err != nil {
		return nil, apperror.Internal(err)
	}
	historyRows, appErr := selectForecastHistoryRows(rows, input.ForecastStart, runtimeConfig.ModelHistoryWindowConfig.ForecastHistoryDays)
	if appErr != nil {
		return nil, appErr
	}

	if !input.ForceRefresh {
		if existing, err := s.forecastRepo.GetLatestByRange(ctx, datasetID, modelType, input.ForecastStart, input.ForecastEnd, granularity); err == nil {
			return map[string]any{"forecast": forecastRecordDTO(existing)}, nil
		} else if err != nil && !errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, apperror.Internal(err)
		}
	}

	response, err := s.modelClient.Forecast(ctx, modelclient.ForecastRequest{
		ModelType:     modelType,
		DatasetID:     datasetID,
		ForecastStart: input.ForecastStart.Format(time.RFC3339),
		ForecastEnd:   input.ForecastEnd.Format(time.RFC3339),
		Granularity:   granularity,
		Series:        buildModelSeries(historyRows),
		Metadata:      modelclient.Metadata{Granularity: granularity, Unit: "w"},
	})
	if err != nil {
		return nil, apperror.ServiceUnavailable("MODEL_SERVICE_UNAVAILABLE", "预测模型服务调用失败", map[string]any{"error": err.Error()})
	}

	expectedPoints := expected15MinutePoints(input.ForecastStart, input.ForecastEnd)
	if len(response.Predictions) != expectedPoints {
		return nil, apperror.ServiceUnavailable("MODEL_SERVICE_INVALID_RESPONSE", "模型返回的预测点数量不正确", map[string]any{
			"expected_points": expectedPoints,
			"actual_points":   len(response.Predictions),
		})
	}

	series := buildForecastSeries(input.ForecastStart, response.Predictions)
	summary := buildForecastSummary(input.ForecastStart, input.ForecastEnd, granularity, series, runtimeConfig.PeakValleyConfig)
	summaryJSON, err := json.Marshal(summary)
	if err != nil {
		return nil, apperror.Internal(err)
	}

	now := time.Now()
	detailPath, err := s.buildForecastDetailPath(datasetID)
	if err != nil {
		return nil, apperror.Internal(err)
	}
	record := &domain.ForecastResultRecord{
		DatasetID:     datasetID,
		ModelType:     modelType,
		ForecastStart: input.ForecastStart,
		ForecastEnd:   input.ForecastEnd,
		Granularity:   granularity,
		Summary:       summaryJSON,
		DetailPath:    detailPath,
		CreatedAt:     &now,
	}
	if err := s.forecastRepo.Create(ctx, record); err != nil {
		return nil, apperror.Internal(err)
	}
	if err := s.writeForecastDetail(record, summary, series); err != nil {
		return nil, apperror.Internal(err)
	}

	s.logInfo("时序预测完成", zap.Uint64("dataset_id", datasetID), zap.String("model_type", modelType))
	return map[string]any{"forecast": forecastRecordDTO(record)}, nil
}

func (s *ForecastService) List(ctx context.Context, datasetID uint64, params ForecastListParams) (map[string]any, *apperror.AppError) {
	if s.forecastRepo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持读取预测结果", nil)
	}

	records, total, err := s.forecastRepo.ListByDatasetID(ctx, datasetID, strings.ToLower(strings.TrimSpace(params.ModelType)), params.Page, params.PageSize)
	if err != nil {
		return nil, apperror.Internal(err)
	}

	items := make([]map[string]any, 0, len(records))
	for _, record := range records {
		recordCopy := record
		items = append(items, forecastRecordDTO(&recordCopy))
	}

	page := params.Page
	if page <= 0 {
		page = 1
	}
	pageSize := params.PageSize
	if pageSize <= 0 {
		pageSize = 20
	}
	if pageSize > 100 {
		pageSize = 100
	}

	return map[string]any{
		"items": items,
		"pagination": map[string]any{
			"page":      page,
			"page_size": pageSize,
			"total":     total,
		},
	}, nil
}

func (s *ForecastService) Get(ctx context.Context, forecastID uint64) (map[string]any, *apperror.AppError) {
	if s.forecastRepo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持读取预测结果", nil)
	}

	record, err := s.forecastRepo.GetByID(ctx, forecastID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, apperror.NotFound("FORECAST_NOT_FOUND", "预测结果不存在", map[string]any{"forecast_id": forecastID})
		}
		return nil, apperror.Internal(err)
	}

	content, err := os.ReadFile(record.DetailPath)
	if err != nil {
		return nil, apperror.Internal(err)
	}

	var detail domain.ForecastDetail
	if err := json.Unmarshal(content, &detail); err != nil {
		return nil, apperror.Internal(err)
	}
	return map[string]any{
		"forecast": detail.Forecast,
		"series":   detail.Series,
	}, nil
}

func (s *ForecastService) Backtest(ctx context.Context, datasetID uint64, input ForecastBacktestInput) (map[string]any, *apperror.AppError) {
	if s.datasetRepo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持回测", nil)
	}
	if s.modelClient == nil {
		return nil, apperror.ServiceUnavailable("MODEL_SERVICE_UNAVAILABLE", "模型服务未初始化", nil)
	}

	modelType, granularity, appErr := normalizeForecastRequest(input.ModelType, input.Granularity)
	if appErr != nil {
		return nil, appErr
	}
	if err := validateForecastRange(input.BacktestStart, input.BacktestEnd); err != nil {
		return nil, apperror.Unprocessable("INVALID_REQUEST", err.Error(), nil)
	}

	dataset, appErr := getReadyDatasetRecord(ctx, s.datasetRepo, datasetID)
	if appErr != nil {
		return nil, appErr
	}
	rows, err := readProcessedFeatureRows(*dataset.ProcessedFilePath)
	if err != nil {
		return nil, apperror.Internal(err)
	}

	runtimeConfig, err := s.systemConfigService.Get(ctx)
	if err != nil {
		return nil, apperror.Internal(err)
	}
	historyRows, appErr := selectForecastHistoryRows(rows, input.BacktestStart, runtimeConfig.ModelHistoryWindowConfig.ForecastHistoryDays)
	if appErr != nil {
		return nil, appErr
	}
	actualRows := filterRowsInRange(rows, input.BacktestStart, input.BacktestEnd)
	if len(actualRows) != expected15MinutePoints(input.BacktestStart, input.BacktestEnd) {
		return nil, apperror.Unprocessable("INSUFFICIENT_HISTORY", "回测区间缺少完整实际值", map[string]any{"actual_points": len(actualRows)})
	}

	series := append(buildModelSeries(historyRows), buildModelSeries(actualRows)...)
	response, err := s.modelClient.Backtest(ctx, modelclient.BacktestRequest{
		ModelType:     modelType,
		DatasetID:     datasetID,
		BacktestStart: input.BacktestStart.Format(time.RFC3339),
		BacktestEnd:   input.BacktestEnd.Format(time.RFC3339),
		Granularity:   granularity,
		Series:        series,
		Metadata:      modelclient.Metadata{Granularity: granularity, Unit: "w"},
	})
	if err != nil {
		return nil, apperror.ServiceUnavailable("MODEL_SERVICE_UNAVAILABLE", "回测模型服务调用失败", map[string]any{"error": err.Error()})
	}

	points := make([]domain.BacktestPredictionPoint, 0, len(response.Predictions))
	for _, point := range response.Predictions {
		ts, parseErr := time.Parse(time.RFC3339, point.Timestamp)
		if parseErr != nil {
			return nil, apperror.ServiceUnavailable("MODEL_SERVICE_INVALID_RESPONSE", "模型返回的回测时间格式非法", map[string]any{"timestamp": point.Timestamp})
		}
		points = append(points, domain.BacktestPredictionPoint{
			Timestamp: ts,
			Actual:    roundFloat(point.Actual, 6),
			Predicted: roundFloat(point.Predicted, 6),
		})
	}

	return map[string]any{
		"backtest": map[string]any{
			"dataset_id":     datasetID,
			"model_type":     modelType,
			"backtest_start": input.BacktestStart,
			"backtest_end":   input.BacktestEnd,
			"granularity":    granularity,
			"metrics": map[string]any{
				"mae":   roundFloat(response.Metrics.MAE, 4),
				"rmse":  roundFloat(response.Metrics.RMSE, 4),
				"smape": roundFloat(response.Metrics.SMAPE, 4),
				"wape":  roundFloat(response.Metrics.WAPE, 4),
			},
		},
		"predictions": points,
	}, nil
}

func normalizeForecastRequest(modelType, granularity string) (string, string, *apperror.AppError) {
	normalizedModelType := strings.TrimSpace(strings.ToLower(modelType))
	if normalizedModelType == "" {
		normalizedModelType = "lstm"
	}
	if normalizedModelType != "lstm" && normalizedModelType != "transformer" {
		return "", "", apperror.Unprocessable("INVALID_REQUEST", "当前仅支持 lstm 或 transformer 预测模型", map[string]any{"model_type": modelType})
	}

	normalizedGranularity := strings.TrimSpace(strings.ToLower(granularity))
	if normalizedGranularity == "" {
		normalizedGranularity = "15min"
	}
	if normalizedGranularity != "15min" {
		return "", "", apperror.Unprocessable("INVALID_REQUEST", "当前仅支持 15min 粒度预测", map[string]any{"granularity": granularity})
	}
	return normalizedModelType, normalizedGranularity, nil
}

func validateForecastRange(start, end time.Time) error {
	if end.Before(start) {
		return errors.New("预测时间范围非法")
	}
	if expected15MinutePoints(start, end) != 96 {
		return errors.New("当前模型固定预测下一天 96 个 15 分钟点，请传入完整单日区间")
	}
	return nil
}

func expected15MinutePoints(start, end time.Time) int {
	return int(end.Sub(start)/(15*time.Minute)) + 1
}

func selectForecastHistoryRows(rows []processedFeatureRow, forecastStart time.Time, historyDays int) ([]processedFeatureRow, *apperror.AppError) {
	expectedPoints := historyDays * 96
	if expectedPoints <= 0 {
		expectedPoints = 288
	}
	historyStart := forecastStart.Add(-time.Duration(expectedPoints) * 15 * time.Minute)
	filtered := filterRowsByExclusiveEnd(rows, historyStart, forecastStart)
	if len(filtered) != expectedPoints {
		return nil, apperror.Unprocessable("INSUFFICIENT_HISTORY", "历史数据不足，至少需要最近 3 天完整 15 分钟序列", map[string]any{
			"expected_points": expectedPoints,
			"actual_points":   len(filtered),
		})
	}
	return filtered, nil
}

func buildForecastSeries(start time.Time, predictions []float64) []domain.ForecastSeriesPoint {
	series := make([]domain.ForecastSeriesPoint, 0, len(predictions))
	for index, value := range predictions {
		series = append(series, domain.ForecastSeriesPoint{
			Timestamp: start.Add(time.Duration(index) * 15 * time.Minute),
			Predicted: roundFloat(value, 6),
		})
	}
	return series
}

func buildForecastSummary(start, end time.Time, granularity string, series []domain.ForecastSeriesPoint, peakValleyConfig domain.PeakValleyConfig) domain.ForecastSummary {
	totalLoad := 0.0
	peakLoadW := 0.0
	peakTotal := 0.0
	valleyTotal := 0.0
	flatTotal := 0.0
	values := make([]float64, 0, len(series))
	for _, point := range series {
		totalLoad += point.Predicted
		if point.Predicted > peakLoadW {
			peakLoadW = point.Predicted
		}
		values = append(values, point.Predicted)
		switch classifyPeakValley(point.Timestamp, peakValleyConfig) {
		case "peak":
			peakTotal += point.Predicted
		case "valley":
			valleyTotal += point.Predicted
		default:
			flatTotal += point.Predicted
		}
	}

	threshold := percentile(values, 0.75)
	periods := extractHighLoadPeriods(series, threshold)

	peakRatio := 0.0
	valleyRatio := 0.0
	flatRatio := 0.0
	if totalLoad > 1e-6 {
		peakRatio = peakTotal / totalLoad
		valleyRatio = valleyTotal / totalLoad
		flatRatio = flatTotal / totalLoad
	}
	avgLoadW := 0.0
	if len(series) > 0 {
		avgLoadW = totalLoad / float64(len(series))
	}

	riskFlags := make([]string, 0, 3)
	if hasEveningPeak(series, threshold) {
		riskFlags = append(riskFlags, "evening_peak_risk")
	}
	if valleyRatio >= 0.35 {
		riskFlags = append(riskFlags, "night_load_risk")
	}
	if peakRatio >= 0.45 {
		riskFlags = append(riskFlags, "peak_usage_risk")
	}

	return domain.ForecastSummary{
		ForecastStart:        start,
		ForecastEnd:          end,
		Granularity:          granularity,
		PredictedAvgLoadW:    roundFloat(avgLoadW, 4),
		PredictedPeakLoadW:   roundFloat(peakLoadW, 4),
		ForecastPeakPeriods:  periods,
		PredictedPeakRatio:   roundFloat(peakRatio, 4),
		PredictedValleyRatio: roundFloat(valleyRatio, 4),
		PredictedFlatRatio:   roundFloat(flatRatio, 4),
		RiskFlags:            riskFlags,
	}
}

func percentile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sortedValues := append([]float64(nil), values...)
	sort.Float64s(sortedValues)
	index := int(math.Ceil(float64(len(sortedValues))*p)) - 1
	if index < 0 {
		index = 0
	}
	if index >= len(sortedValues) {
		index = len(sortedValues) - 1
	}
	return sortedValues[index]
}

func extractHighLoadPeriods(series []domain.ForecastSeriesPoint, threshold float64) []string {
	periods := make([]string, 0)
	var start *time.Time
	var previous *time.Time
	for _, point := range series {
		if point.Predicted >= threshold {
			if start == nil {
				ts := point.Timestamp
				start = &ts
			}
			ts := point.Timestamp
			previous = &ts
			continue
		}
		if start != nil && previous != nil {
			periods = append(periods, formatPeriod(*start, previous.Add(15*time.Minute)))
			start = nil
			previous = nil
		}
	}
	if start != nil && previous != nil {
		periods = append(periods, formatPeriod(*start, previous.Add(15*time.Minute)))
	}
	return periods
}

func formatPeriod(start, end time.Time) string {
	return start.Format(time.RFC3339) + "/" + end.Format(time.RFC3339)
}

func hasEveningPeak(series []domain.ForecastSeriesPoint, threshold float64) bool {
	for _, point := range series {
		hour := point.Timestamp.Hour()
		if hour >= 18 && hour < 23 && point.Predicted >= threshold {
			return true
		}
	}
	return false
}

func (s *ForecastService) buildForecastDetailPath(datasetID uint64) (string, error) {
	forecastDir := filepath.Join(s.cfg.OutputRootDir, "forecasts")
	if err := os.MkdirAll(forecastDir, 0o755); err != nil {
		return "", err
	}
	return filepath.ToSlash(filepath.Join(forecastDir, fmt.Sprintf("dataset_%d_forecast_%d.json", datasetID, time.Now().UnixNano()))), nil
}

func (s *ForecastService) writeForecastDetail(record *domain.ForecastResultRecord, summary domain.ForecastSummary, series []domain.ForecastSeriesPoint) error {
	if record == nil {
		return errors.New("forecast record 不能为空")
	}
	forecastPayload := map[string]any{
		"id":             record.ID,
		"dataset_id":     record.DatasetID,
		"model_type":     record.ModelType,
		"forecast_start": summary.ForecastStart,
		"forecast_end":   summary.ForecastEnd,
		"granularity":    summary.Granularity,
		"summary":        summary,
		"detail_path":    record.DetailPath,
		"created_at":     record.CreatedAt,
	}
	content, err := json.MarshalIndent(domain.ForecastDetail{
		Forecast: forecastPayload,
		Series:   series,
	}, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(record.DetailPath, content, 0o644); err != nil {
		return err
	}
	return nil
}

func forecastRecordDTO(record *domain.ForecastResultRecord) map[string]any {
	var summary any
	if len(record.Summary) > 0 {
		_ = json.Unmarshal(record.Summary, &summary)
	}
	return map[string]any{
		"id":             record.ID,
		"dataset_id":     record.DatasetID,
		"model_type":     record.ModelType,
		"forecast_start": record.ForecastStart,
		"forecast_end":   record.ForecastEnd,
		"granularity":    record.Granularity,
		"summary":        summary,
		"detail_path":    record.DetailPath,
		"created_at":     record.CreatedAt,
	}
}

func (s *ForecastService) logInfo(message string, fields ...zap.Field) {
	if s.logger != nil {
		s.logger.Info(message, fields...)
	}
}
