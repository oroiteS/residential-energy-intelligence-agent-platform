package service

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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

type ClassificationService struct {
	cfg                 *config.AppConfig
	datasetRepo         repository.DatasetRepository
	classificationRepo  repository.ClassificationResultRepository
	systemConfigService *SystemConfigService
	modelClient         modelclient.Client
	logger              *zap.Logger
}

type ClassificationPredictInput struct {
	ModelType    string     `json:"model_type"`
	WindowMode   string     `json:"window_mode"`
	WindowStart  *time.Time `json:"window_start"`
	WindowEnd    *time.Time `json:"window_end"`
	ForceRefresh bool       `json:"force_refresh"`
}

type ClassificationListParams struct {
	Page      int
	PageSize  int
	ModelType string
}

func NewClassificationService(
	cfg *config.AppConfig,
	datasetRepo repository.DatasetRepository,
	classificationRepo repository.ClassificationResultRepository,
	systemConfigService *SystemConfigService,
	modelClient modelclient.Client,
	logger *zap.Logger,
) *ClassificationService {
	return &ClassificationService{
		cfg:                 cfg,
		datasetRepo:         datasetRepo,
		classificationRepo:  classificationRepo,
		systemConfigService: systemConfigService,
		modelClient:         modelClient,
		logger:              logger,
	}
}

func (s *ClassificationService) Predict(ctx context.Context, datasetID uint64, input ClassificationPredictInput) (map[string]any, *apperror.AppError) {
	if s.datasetRepo == nil || s.classificationRepo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持分类推理", nil)
	}
	if s.modelClient == nil {
		return nil, apperror.ServiceUnavailable("MODEL_SERVICE_UNAVAILABLE", "模型服务未初始化", nil)
	}

	modelType := strings.TrimSpace(strings.ToLower(input.ModelType))
	if modelType == "" {
		modelType = "xgboost"
	}
	if modelType != "xgboost" {
		return nil, apperror.Unprocessable("INVALID_REQUEST", "当前仅支持 xgboost 分类模型", map[string]any{"model_type": input.ModelType})
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

	windowRows, windowStart, windowEnd, appErr := selectClassificationWindow(rows, runtimeConfig.ModelHistoryWindowConfig.ClassificationDays, input.WindowMode, input.WindowStart, input.WindowEnd)
	if appErr != nil {
		return nil, appErr
	}

	if !input.ForceRefresh {
		if existing, err := s.classificationRepo.GetLatestByWindow(ctx, datasetID, modelType, &windowStart, &windowEnd); err == nil {
			return map[string]any{"classification": classificationRecordDTO(existing)}, nil
		} else if err != nil && !errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, apperror.Internal(err)
		}
	}

	request := modelclient.PredictClassificationRequest{
		ModelType: modelType,
		DatasetID: datasetID,
		Window: modelclient.Window{
			Start: windowStart.Format(time.RFC3339),
			End:   windowEnd.Format(time.RFC3339),
		},
		Series:   buildModelSeries(windowRows),
		Metadata: modelclient.Metadata{Granularity: "15min", Unit: "w"},
	}
	response, err := s.modelClient.PredictClassification(ctx, request)
	if err != nil {
		return nil, apperror.ServiceUnavailable("MODEL_SERVICE_UNAVAILABLE", "分类模型服务调用失败", map[string]any{"error": err.Error()})
	}

	probabilities := map[string]float64{
		"day_high_night_low": roundFloat(response.ProbDayHighNightLow, 6),
		"day_low_night_high": roundFloat(response.ProbDayLowNightHigh, 6),
		"all_day_high":       roundFloat(response.ProbAllDayHigh, 6),
		"all_day_low":        roundFloat(response.ProbAllDayLow, 6),
	}
	probabilitiesJSON, err := json.Marshal(probabilities)
	if err != nil {
		return nil, apperror.Internal(err)
	}

	explanationText := buildClassificationExplanation(windowRows, response.PredictedLabel)
	now := time.Now()
	record := &domain.ClassificationResultRecord{
		DatasetID:      datasetID,
		ModelType:      modelType,
		PredictedLabel: response.PredictedLabel,
		Confidence:     roundFloat(response.Confidence, 4),
		Probabilities:  probabilitiesJSON,
		Explanation:    &explanationText,
		WindowStart:    &windowStart,
		WindowEnd:      &windowEnd,
		CreatedAt:      &now,
	}
	if err := s.classificationRepo.Create(ctx, record); err != nil {
		return nil, apperror.Internal(err)
	}

	s.logInfo("分类推理完成", zap.Uint64("dataset_id", datasetID), zap.String("label", response.PredictedLabel))
	return map[string]any{"classification": classificationRecordDTO(record)}, nil
}

func (s *ClassificationService) GetLatest(ctx context.Context, datasetID uint64, modelType string) (map[string]any, *apperror.AppError) {
	if s.classificationRepo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持读取分类结果", nil)
	}

	record, err := s.classificationRepo.GetLatest(ctx, datasetID, normalizedClassificationModelType(modelType))
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, apperror.NotFound("CLASSIFICATION_NOT_FOUND", "分类结果不存在", map[string]any{"dataset_id": datasetID})
		}
		return nil, apperror.Internal(err)
	}
	return map[string]any{"classification": classificationRecordDTO(record)}, nil
}

func (s *ClassificationService) List(ctx context.Context, datasetID uint64, params ClassificationListParams) (map[string]any, *apperror.AppError) {
	if s.classificationRepo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持读取分类结果", nil)
	}

	records, total, err := s.classificationRepo.ListByDatasetID(ctx, datasetID, normalizedClassificationModelType(params.ModelType), params.Page, params.PageSize)
	if err != nil {
		return nil, apperror.Internal(err)
	}

	items := make([]map[string]any, 0, len(records))
	for _, record := range records {
		recordCopy := record
		items = append(items, classificationRecordDTO(&recordCopy))
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

func selectClassificationWindow(rows []processedFeatureRow, historyDays int, windowMode string, requestedStart, requestedEnd *time.Time) ([]processedFeatureRow, time.Time, time.Time, *apperror.AppError) {
	expectedPoints := historyDays * 96
	if expectedPoints <= 0 {
		expectedPoints = 96
	}

	mode := strings.TrimSpace(strings.ToLower(windowMode))
	if mode == "" {
		mode = "full_dataset"
	}

	if mode == "time_window" {
		if requestedStart == nil || requestedEnd == nil {
			return nil, time.Time{}, time.Time{}, apperror.Unprocessable("INVALID_REQUEST", "time_window 模式必须传 window_start 和 window_end", nil)
		}
		filtered := filterRowsInRange(rows, *requestedStart, *requestedEnd)
		if len(filtered) != expectedPoints {
			return nil, time.Time{}, time.Time{}, apperror.Unprocessable("INVALID_REQUEST", "分类窗口数据点数量不足，当前模型固定需要 96 个 15 分钟点", map[string]any{"actual_points": len(filtered)})
		}
		return filtered, filtered[0].Timestamp, filtered[len(filtered)-1].Timestamp, nil
	}

	grouped := make(map[string][]processedFeatureRow)
	order := make([]string, 0)
	for _, row := range rows {
		dateKey := row.Timestamp.Format("2006-01-02")
		if _, exists := grouped[dateKey]; !exists {
			order = append(order, dateKey)
		}
		grouped[dateKey] = append(grouped[dateKey], row)
	}
	sort.Strings(order)
	for index := len(order) - 1; index >= 0; index-- {
		dayRows := grouped[order[index]]
		if len(dayRows) != expectedPoints {
			continue
		}
		sort.Slice(dayRows, func(i, j int) bool {
			return dayRows[i].Timestamp.Before(dayRows[j].Timestamp)
		})
		return dayRows, dayRows[0].Timestamp, dayRows[len(dayRows)-1].Timestamp, nil
	}
	return nil, time.Time{}, time.Time{}, apperror.NotFound("INSUFFICIENT_HISTORY", "数据集中不存在完整的单日分类窗口", nil)
}

func buildModelSeries(rows []processedFeatureRow) []modelclient.TimeSeriesPoint {
	series := make([]modelclient.TimeSeriesPoint, 0, len(rows))
	for _, row := range rows {
		series = append(series, modelclient.TimeSeriesPoint{
			Timestamp:            row.Timestamp.Format(time.RFC3339),
			Aggregate:            row.Aggregate,
			ActiveApplianceCount: row.ActiveApplianceCount,
			BurstEventCount:      row.BurstEventCount,
		})
	}
	return series
}

func buildClassificationExplanation(rows []processedFeatureRow, predictedLabel string) string {
	if len(rows) == 0 {
		return "分类窗口为空，无法生成解释。"
	}
	dayTotal := 0.0
	nightTotal := 0.0
	dayCount := 0
	nightCount := 0
	fullTotal := 0.0
	for _, row := range rows {
		fullTotal += row.Aggregate
		hour := row.Timestamp.Hour()
		if hour >= 8 && hour < 18 {
			dayTotal += row.Aggregate
			dayCount++
		} else {
			nightTotal += row.Aggregate
			nightCount++
		}
	}
	dayMean := safeMean(dayTotal, dayCount)
	nightMean := safeMean(nightTotal, nightCount)
	fullMean := safeMean(fullTotal, len(rows))
	ratio := 0.0
	if nightMean > 1e-6 {
		ratio = dayMean / nightMean
	}
	reverseRatio := 0.0
	if dayMean > 1e-6 {
		reverseRatio = nightMean / dayMean
	}

	switch predictedLabel {
	case "day_high_night_low":
		return fmt.Sprintf("白天均值高于夜间均值，day_mean/night_mean = %.2f，符合白天高晚上低型。", roundFloat(ratio, 2))
	case "day_low_night_high":
		return fmt.Sprintf("夜间均值高于白天均值，night_mean/day_mean = %.2f，符合白天低晚上高型。", roundFloat(reverseRatio, 2))
	case "all_day_high":
		return fmt.Sprintf("全天平均负荷较高，full_mean = %.4f，整体更接近全天高负载型。", roundFloat(fullMean, 4))
	default:
		return fmt.Sprintf("全天平均负荷较低，full_mean = %.4f，整体更接近全天低负载型。", roundFloat(fullMean, 4))
	}
}

func safeMean(total float64, count int) float64 {
	if count <= 0 {
		return 0
	}
	return total / float64(count)
}

func classificationRecordDTO(record *domain.ClassificationResultRecord) map[string]any {
	var probabilities map[string]float64
	if len(record.Probabilities) > 0 {
		_ = json.Unmarshal(record.Probabilities, &probabilities)
	}
	return map[string]any{
		"id":                 record.ID,
		"dataset_id":         record.DatasetID,
		"schema_version":     "v1",
		"model_type":         normalizedClassificationModelType(record.ModelType),
		"predicted_label":    record.PredictedLabel,
		"confidence":         roundFloat(record.Confidence, 4),
		"label_display_name": classificationLabelText(record.PredictedLabel),
		"probabilities":      probabilities,
		"explanation":        nullableString(record.Explanation),
		"window_start":       record.WindowStart,
		"window_end":         record.WindowEnd,
		"created_at":         record.CreatedAt,
	}
}

func normalizedClassificationModelType(raw string) string {
	if strings.TrimSpace(raw) == "" {
		return "xgboost"
	}
	return strings.ToLower(strings.TrimSpace(raw))
}

func (s *ClassificationService) logInfo(message string, fields ...zap.Field) {
	if s.logger != nil {
		s.logger.Info(message, fields...)
	}
}
