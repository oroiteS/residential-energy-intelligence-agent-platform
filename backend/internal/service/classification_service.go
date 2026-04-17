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

type classificationWindow struct {
	Rows   []processedFeatureRow
	Start  time.Time
	End    time.Time
	DayKey string
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

	windows, appErr := selectClassificationWindows(rows, runtimeConfig.ModelHistoryWindowConfig.ClassificationDays, input.WindowMode, input.WindowStart, input.WindowEnd)
	if appErr != nil {
		return nil, appErr
	}

	items := make([]map[string]any, 0, len(windows))
	var latestRecord *domain.ClassificationResultRecord

	for _, window := range windows {
		record, err := s.getOrCreateClassificationRecord(ctx, datasetID, modelType, window, input.ForceRefresh)
		if err != nil {
			return nil, err
		}
		recordCopy := *record
		items = append(items, classificationRecordDTO(&recordCopy))
		latestRecord = &recordCopy
	}

	if latestRecord == nil {
		return nil, apperror.NotFound("CLASSIFICATION_NOT_FOUND", "分类结果不存在", map[string]any{"dataset_id": datasetID})
	}

	s.logInfo("分类推理完成", zap.Uint64("dataset_id", datasetID), zap.Int("days", len(items)), zap.String("latest_label", latestRecord.PredictedLabel))
	return map[string]any{
		"classification": classificationRecordDTO(latestRecord),
		"items":          reverseClassificationItems(items),
	}, nil
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

	records, err := s.classificationRepo.ListAllByDatasetID(ctx, datasetID, normalizedClassificationModelType(params.ModelType))
	if err != nil {
		return nil, apperror.Internal(err)
	}

	uniqueRecords := latestClassificationRecordsByWindow(records)
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

	startIndex := (page - 1) * pageSize
	if startIndex > len(uniqueRecords) {
		startIndex = len(uniqueRecords)
	}
	endIndex := startIndex + pageSize
	if endIndex > len(uniqueRecords) {
		endIndex = len(uniqueRecords)
	}

	items := make([]map[string]any, 0, endIndex-startIndex)
	for _, record := range uniqueRecords[startIndex:endIndex] {
		recordCopy := record
		items = append(items, classificationRecordDTO(&recordCopy))
	}

	return map[string]any{
		"items": items,
		"pagination": map[string]any{
			"page":      page,
			"page_size": pageSize,
			"total":     len(uniqueRecords),
		},
	}, nil
}

func selectClassificationWindow(rows []processedFeatureRow, historyDays int, windowMode string, requestedStart, requestedEnd *time.Time) ([]processedFeatureRow, time.Time, time.Time, *apperror.AppError) {
	windows, appErr := selectClassificationWindows(rows, historyDays, windowMode, requestedStart, requestedEnd)
	if appErr != nil {
		return nil, time.Time{}, time.Time{}, appErr
	}
	lastWindow := windows[len(windows)-1]
	return lastWindow.Rows, lastWindow.Start, lastWindow.End, nil
}

func selectClassificationWindows(rows []processedFeatureRow, historyDays int, windowMode string, requestedStart, requestedEnd *time.Time) ([]classificationWindow, *apperror.AppError) {
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
			return nil, apperror.Unprocessable("INVALID_REQUEST", "time_window 模式必须传 window_start 和 window_end", nil)
		}
		filtered := filterRowsInRange(rows, *requestedStart, *requestedEnd)
		if len(filtered) != expectedPoints {
			return nil, apperror.Unprocessable("INVALID_REQUEST", "分类窗口数据点数量不足，当前模型固定需要 96 个 15 分钟点", map[string]any{"actual_points": len(filtered)})
		}
		return []classificationWindow{{
			Rows:   filtered,
			Start:  filtered[0].Timestamp,
			End:    filtered[len(filtered)-1].Timestamp,
			DayKey: filtered[0].Timestamp.Format("2006-01-02"),
		}}, nil
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
	windows := make([]classificationWindow, 0, len(order))
	for _, dayKey := range order {
		dayRows := grouped[dayKey]
		if len(dayRows) != expectedPoints {
			continue
		}
		sort.Slice(dayRows, func(i, j int) bool {
			return dayRows[i].Timestamp.Before(dayRows[j].Timestamp)
		})
		windows = append(windows, classificationWindow{
			Rows:   dayRows,
			Start:  dayRows[0].Timestamp,
			End:    dayRows[len(dayRows)-1].Timestamp,
			DayKey: dayKey,
		})
	}
	if len(windows) == 0 {
		return nil, apperror.NotFound("INSUFFICIENT_HISTORY", "数据集中不存在完整的单日分类窗口", nil)
	}
	return windows, nil
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
	case "afternoon_peak":
		return fmt.Sprintf("白天尤其下午时段负荷更高，day_mean/night_mean = %.2f，更接近下午高峰型。", roundFloat(ratio, 2))
	case "day_low_night_high":
		return fmt.Sprintf("夜间均值高于白天均值，night_mean/day_mean = %.2f，更接近晚上高峰型。", roundFloat(reverseRatio, 2))
	case "morning_peak":
		return fmt.Sprintf("全天峰值更偏向上午窗口，full_mean = %.4f，整体更接近上午高峰型。", roundFloat(fullMean, 4))
	default:
		return fmt.Sprintf("全天平均负荷较平稳，full_mean = %.4f，整体更接近全天平稳型。", roundFloat(fullMean, 4))
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
		"predicted_label":    normalizeClassificationLabel(record.PredictedLabel),
		"confidence":         roundFloat(record.Confidence, 4),
		"label_display_name": classificationLabelText(record.PredictedLabel),
		"probabilities":      normalizeClassificationProbabilities(probabilities),
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

func normalizeClassificationLabel(raw string) string {
	switch strings.TrimSpace(raw) {
	case "daytime_active", "day_high_night_low":
		return "afternoon_peak"
	case "daytime_peak_strong", "all_day_high":
		return "morning_peak"
	case "flat_stable":
		return "all_day_low"
	case "night_dominant":
		return "day_low_night_high"
	default:
		return strings.TrimSpace(raw)
	}
}

func normalizeClassificationProbabilities(raw map[string]float64) map[string]float64 {
	if len(raw) == 0 {
		return map[string]float64{}
	}
	normalized := make(map[string]float64, len(raw))
	for label, probability := range raw {
		normalized[normalizeClassificationLabel(label)] = roundFloat(probability, 6)
	}
	return normalized
}

func (s *ClassificationService) getOrCreateClassificationRecord(
	ctx context.Context,
	datasetID uint64,
	modelType string,
	window classificationWindow,
	forceRefresh bool,
) (*domain.ClassificationResultRecord, *apperror.AppError) {
	existing, err := s.classificationRepo.GetLatestByWindow(ctx, datasetID, modelType, &window.Start, &window.End)
	if err == nil && !forceRefresh {
		return existing, nil
	}
	if err != nil && !errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, apperror.Internal(err)
	}

	request := modelclient.PredictClassificationRequest{
		ModelType: modelType,
		DatasetID: datasetID,
		Window: modelclient.Window{
			Start: window.Start.Format(time.RFC3339),
			End:   window.End.Format(time.RFC3339),
		},
		Series:   buildModelSeries(window.Rows),
		Metadata: modelclient.Metadata{Granularity: "15min", Unit: "w"},
	}
	response, predictErr := s.modelClient.PredictClassification(ctx, request)
	if predictErr != nil {
		return nil, apperror.ServiceUnavailable("MODEL_SERVICE_UNAVAILABLE", "分类模型服务调用失败", map[string]any{"error": predictErr.Error()})
	}

	predictedLabel := normalizeClassificationLabel(response.PredictedLabel)
	probabilities := normalizeClassificationProbabilities(response.Probabilities)
	if len(probabilities) == 0 {
		probabilities = map[string]float64{
			"afternoon_peak":     roundFloat(response.ProbAfternoonPeak, 6),
			"day_low_night_high": roundFloat(response.ProbDayLowNightHigh, 6),
			"all_day_low":        roundFloat(response.ProbAllDayLow, 6),
			"morning_peak":       roundFloat(response.ProbMorningPeak, 6),
		}
	}
	probabilitiesJSON, marshalErr := json.Marshal(probabilities)
	if marshalErr != nil {
		return nil, apperror.Internal(marshalErr)
	}

	explanationText := buildClassificationExplanation(window.Rows, predictedLabel)
	now := time.Now()
	record := &domain.ClassificationResultRecord{
		DatasetID:      datasetID,
		ModelType:      modelType,
		PredictedLabel: predictedLabel,
		Confidence:     roundFloat(response.Confidence, 4),
		Probabilities:  probabilitiesJSON,
		Explanation:    &explanationText,
		WindowStart:    &window.Start,
		WindowEnd:      &window.End,
		CreatedAt:      &now,
	}
	if existing != nil {
		record.ID = existing.ID
		if saveErr := s.classificationRepo.Save(ctx, record); saveErr != nil {
			return nil, apperror.Internal(saveErr)
		}
		return record, nil
	}
	if createErr := s.classificationRepo.Create(ctx, record); createErr != nil {
		return nil, apperror.Internal(createErr)
	}
	return record, nil
}

func latestClassificationRecordsByWindow(records []domain.ClassificationResultRecord) []domain.ClassificationResultRecord {
	unique := make(map[string]domain.ClassificationResultRecord, len(records))
	order := make([]string, 0, len(records))
	for _, record := range records {
		key := classificationWindowKey(record.WindowStart, record.WindowEnd)
		if _, exists := unique[key]; exists {
			continue
		}
		unique[key] = record
		order = append(order, key)
	}

	result := make([]domain.ClassificationResultRecord, 0, len(order))
	for _, key := range order {
		result = append(result, unique[key])
	}
	sort.Slice(result, func(i, j int) bool {
		left := result[i]
		right := result[j]
		if left.WindowStart != nil && right.WindowStart != nil && !left.WindowStart.Equal(*right.WindowStart) {
			return left.WindowStart.After(*right.WindowStart)
		}
		if left.CreatedAt != nil && right.CreatedAt != nil && !left.CreatedAt.Equal(*right.CreatedAt) {
			return left.CreatedAt.After(*right.CreatedAt)
		}
		return left.ID > right.ID
	})
	return result
}

func classificationWindowKey(start, end *time.Time) string {
	startKey := "nil"
	endKey := "nil"
	if start != nil {
		startKey = start.UTC().Format(time.RFC3339)
	}
	if end != nil {
		endKey = end.UTC().Format(time.RFC3339)
	}
	return startKey + "|" + endKey
}

func reverseClassificationItems(items []map[string]any) []map[string]any {
	reversed := make([]map[string]any, 0, len(items))
	for index := len(items) - 1; index >= 0; index-- {
		reversed = append(reversed, items[index])
	}
	return reversed
}

func (s *ClassificationService) logInfo(message string, fields ...zap.Field) {
	if s.logger != nil {
		s.logger.Info(message, fields...)
	}
}
