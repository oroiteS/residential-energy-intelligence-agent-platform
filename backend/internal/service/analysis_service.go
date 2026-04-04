package service

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"go.uber.org/zap"
	"gorm.io/gorm"

	"residential-energy-intelligence-agent-platform/internal/config"
	"residential-energy-intelligence-agent-platform/internal/domain"
	"residential-energy-intelligence-agent-platform/internal/repository"
	"residential-energy-intelligence-agent-platform/pkg/apperror"
)

type AnalysisService struct {
	cfg                 *config.AppConfig
	datasetRepo         repository.DatasetRepository
	analysisRepo        repository.AnalysisResultRepository
	systemConfigService *SystemConfigService
	logger              *zap.Logger
}

type analysisRow struct {
	Timestamp time.Time
	Aggregate float64
}

func NewAnalysisService(
	cfg *config.AppConfig,
	datasetRepo repository.DatasetRepository,
	analysisRepo repository.AnalysisResultRepository,
	systemConfigService *SystemConfigService,
	logger *zap.Logger,
) *AnalysisService {
	return &AnalysisService{
		cfg:                 cfg,
		datasetRepo:         datasetRepo,
		analysisRepo:        analysisRepo,
		systemConfigService: systemConfigService,
		logger:              logger,
	}
}

func (s *AnalysisService) Get(ctx context.Context, datasetID uint64) (*domain.AnalysisDetail, *apperror.AppError) {
	if s.analysisRepo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持统计分析", nil)
	}

	record, err := s.analysisRepo.GetByDatasetID(ctx, datasetID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, apperror.NotFound("ANALYSIS_NOT_FOUND", "统计分析结果不存在", map[string]any{"dataset_id": datasetID})
		}
		return nil, apperror.Internal(err)
	}

	if record.DetailPath == nil || strings.TrimSpace(*record.DetailPath) == "" {
		return nil, apperror.NotFound("ANALYSIS_NOT_FOUND", "统计分析详情不存在", map[string]any{"dataset_id": datasetID})
	}

	content, err := os.ReadFile(*record.DetailPath)
	if err != nil {
		return nil, apperror.Internal(err)
	}

	var detail domain.AnalysisDetail
	if err := json.Unmarshal(content, &detail); err != nil {
		return nil, apperror.Internal(err)
	}
	return &detail, nil
}

func (s *AnalysisService) Recompute(ctx context.Context, datasetID uint64) (*domain.AnalysisDetail, *apperror.AppError) {
	detail, appErr := s.Generate(ctx, datasetID, true)
	if appErr != nil {
		return nil, appErr
	}
	return detail, nil
}

func (s *AnalysisService) Generate(ctx context.Context, datasetID uint64, requireReady bool) (*domain.AnalysisDetail, *apperror.AppError) {
	if s.datasetRepo == nil || s.analysisRepo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持统计分析", nil)
	}

	dataset, err := s.datasetRepo.GetByID(ctx, datasetID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, apperror.NotFound("DATASET_NOT_FOUND", "数据集不存在", map[string]any{"id": datasetID})
		}
		return nil, apperror.Internal(err)
	}
	if requireReady && dataset.Status != "ready" {
		return nil, apperror.Conflict("DATASET_NOT_READY", "数据集尚未处理完成", map[string]any{"id": datasetID})
	}
	if dataset.ProcessedFilePath == nil || strings.TrimSpace(*dataset.ProcessedFilePath) == "" {
		return nil, apperror.NotFound("ANALYSIS_NOT_FOUND", "数据集尚未生成清洗文件", map[string]any{"dataset_id": datasetID})
	}

	rows, err := readProcessedRows(*dataset.ProcessedFilePath)
	if err != nil {
		return nil, apperror.Internal(err)
	}
	if len(rows) == 0 {
		return nil, apperror.NotFound("ANALYSIS_NOT_FOUND", "清洗后的数据为空，无法生成统计分析", map[string]any{"dataset_id": datasetID})
	}

	runtimeConfig, err := s.systemConfigService.Get(ctx)
	if err != nil {
		return nil, apperror.Internal(err)
	}

	detail, record, err := s.computeAnalysis(datasetID, rows, runtimeConfig.PeakValleyConfig)
	if err != nil {
		return nil, apperror.Internal(err)
	}

	detailPath, err := s.writeAnalysisDetail(datasetID, detail)
	if err != nil {
		return nil, apperror.Internal(err)
	}
	detail.DetailPath = detailPath
	now := time.Now()
	detail.UpdatedAt = &now

	record.DetailPath = &detailPath
	record.CreatedAt = &now
	if err := s.analysisRepo.Upsert(ctx, record); err != nil {
		return nil, apperror.Internal(err)
	}

	detailContent, err := json.MarshalIndent(detail, "", "  ")
	if err != nil {
		return nil, apperror.Internal(err)
	}
	if err := os.WriteFile(detailPath, detailContent, 0o644); err != nil {
		return nil, apperror.Internal(err)
	}

	s.logInfo("统计分析生成完成", zap.Uint64("dataset_id", datasetID))
	return detail, nil
}

func (s *AnalysisService) computeAnalysis(datasetID uint64, rows []analysisRow, peakValleyConfig domain.PeakValleyConfig) (*domain.AnalysisDetail, *domain.AnalysisResultRecord, error) {
	totalKWH := 0.0
	peakKWH := 0.0
	valleyKWH := 0.0
	flatKWH := 0.0

	maxLoadW := -1.0
	minLoadW := -1.0
	var maxLoadTime *time.Time
	var minLoadTime *time.Time

	dailyKWH := make(map[string]float64)
	weeklyKWH := make(map[string]float64)
	weeklyEnd := make(map[string]string)
	slotLoadSum := make(map[int]float64)
	slotCount := make(map[int]int)

	for _, row := range rows {
		loadW := row.Aggregate
		energyKWH := loadW * 0.25 / 1000
		totalKWH += energyKWH

		if maxLoadTime == nil || loadW > maxLoadW {
			ts := row.Timestamp
			maxLoadW = loadW
			maxLoadTime = &ts
		}
		if minLoadTime == nil || loadW < minLoadW {
			ts := row.Timestamp
			minLoadW = loadW
			minLoadTime = &ts
		}

		dateKey := row.Timestamp.Format("2006-01-02")
		dailyKWH[dateKey] += energyKWH

		weekStart := startOfWeek(row.Timestamp)
		weekStartKey := weekStart.Format("2006-01-02")
		weeklyKWH[weekStartKey] += energyKWH
		weeklyEnd[weekStartKey] = weekStart.AddDate(0, 0, 6).Format("2006-01-02")

		hour := row.Timestamp.Hour()
		slotLoadSum[hour] += loadW
		slotCount[hour]++

		switch classifyPeakValley(row.Timestamp, peakValleyConfig) {
		case "peak":
			peakKWH += energyKWH
		case "valley":
			valleyKWH += energyKWH
		default:
			flatKWH += energyKWH
		}
	}

	dailyTrend := make([]map[string]any, 0, len(dailyKWH))
	dailyKeys := make([]string, 0, len(dailyKWH))
	for dateKey := range dailyKWH {
		dailyKeys = append(dailyKeys, dateKey)
	}
	sort.Strings(dailyKeys)
	for _, dateKey := range dailyKeys {
		dailyTrend = append(dailyTrend, map[string]any{
			"date": dateKey,
			"kwh":  roundFloat(dailyKWH[dateKey], 4),
		})
	}

	weeklyTrend := make([]map[string]any, 0, len(weeklyKWH))
	weekKeys := make([]string, 0, len(weeklyKWH))
	for weekKey := range weeklyKWH {
		weekKeys = append(weekKeys, weekKey)
	}
	sort.Strings(weekKeys)
	for _, weekKey := range weekKeys {
		weeklyTrend = append(weeklyTrend, map[string]any{
			"week_start": weekKey,
			"week_end":   weeklyEnd[weekKey],
			"kwh":        roundFloat(weeklyKWH[weekKey], 4),
		})
	}

	typicalDayCurve := make([]map[string]any, 0, 24)
	for hour := 0; hour < 24; hour++ {
		avgLoadW := 0.0
		if slotCount[hour] > 0 {
			avgLoadW = slotLoadSum[hour] / float64(slotCount[hour])
		}
		typicalDayCurve = append(typicalDayCurve, map[string]any{
			"hour":       hour,
			"avg_load_w": roundFloat(avgLoadW, 2),
		})
	}

	peakRatio := 0.0
	valleyRatio := 0.0
	flatRatio := 0.0
	if totalKWH > 0 {
		peakRatio = peakKWH / totalKWH
		valleyRatio = valleyKWH / totalKWH
		flatRatio = flatKWH / totalKWH
	}

	peakValleyPie := []map[string]any{
		{"name": "峰时", "ratio": roundFloat(peakRatio, 4), "kwh": roundFloat(peakKWH, 4)},
		{"name": "谷时", "ratio": roundFloat(valleyRatio, 4), "kwh": roundFloat(valleyKWH, 4)},
		{"name": "平时", "ratio": roundFloat(flatRatio, 4), "kwh": roundFloat(flatKWH, 4)},
	}

	dailyAvgKWH := 0.0
	if len(dailyKWH) > 0 {
		dailyAvgKWH = totalKWH / float64(len(dailyKWH))
	}

	summary := domain.AnalysisSummary{
		TotalKWH:    roundFloat(totalKWH, 4),
		DailyAvgKWH: roundFloat(dailyAvgKWH, 4),
		MaxLoadW:    roundFloat(maxLoadW, 2),
		MaxLoadTime: maxLoadTime,
		MinLoadW:    roundFloat(minLoadW, 2),
		MinLoadTime: minLoadTime,
		PeakKWH:     roundFloat(peakKWH, 4),
		ValleyKWH:   roundFloat(valleyKWH, 4),
		FlatKWH:     roundFloat(flatKWH, 4),
		PeakRatio:   roundFloat(peakRatio, 4),
		ValleyRatio: roundFloat(valleyRatio, 4),
		FlatRatio:   roundFloat(flatRatio, 4),
	}

	record := &domain.AnalysisResultRecord{
		DatasetID:   datasetID,
		TotalKWH:    summary.TotalKWH,
		DailyAvgKWH: summary.DailyAvgKWH,
		MaxLoadW:    summary.MaxLoadW,
		MaxLoadTime: summary.MaxLoadTime,
		MinLoadW:    summary.MinLoadW,
		MinLoadTime: summary.MinLoadTime,
		PeakKWH:     summary.PeakKWH,
		ValleyKWH:   summary.ValleyKWH,
		FlatKWH:     summary.FlatKWH,
		PeakRatio:   summary.PeakRatio,
		ValleyRatio: summary.ValleyRatio,
		FlatRatio:   summary.FlatRatio,
	}

	return &domain.AnalysisDetail{
		Summary:          summary,
		PeakValleyConfig: peakValleyConfig,
		Charts: domain.AnalysisCharts{
			DailyTrend:      dailyTrend,
			WeeklyTrend:     weeklyTrend,
			TypicalDayCurve: typicalDayCurve,
			PeakValleyPie:   peakValleyPie,
		},
	}, record, nil
}

func (s *AnalysisService) writeAnalysisDetail(datasetID uint64, detail *domain.AnalysisDetail) (string, error) {
	analysisDir := filepath.Join(s.cfg.OutputRootDir, "analysis")
	if err := os.MkdirAll(analysisDir, 0o755); err != nil {
		return "", err
	}
	return filepath.ToSlash(filepath.Join(analysisDir, fmt.Sprintf("dataset_%d.json", datasetID))), nil
}

func readProcessedRows(path string) ([]analysisRow, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.FieldsPerRecord = -1
	reader.TrimLeadingSpace = true
	rows, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(rows) <= 1 {
		return nil, nil
	}

	headerIndex := make(map[string]int, len(rows[0]))
	for index, header := range rows[0] {
		headerIndex[strings.TrimSpace(header)] = index
	}
	timestampIndex, ok := headerIndex["timestamp"]
	if !ok {
		return nil, errors.New("processed csv 缺少 timestamp 列")
	}
	aggregateIndex, ok := headerIndex["aggregate"]
	if !ok {
		return nil, errors.New("processed csv 缺少 aggregate 列")
	}

	result := make([]analysisRow, 0, len(rows)-1)
	for _, row := range rows[1:] {
		if isEmptyRow(row) {
			continue
		}
		timestamp, err := parseFlexibleTime(cellValue(row, timestampIndex))
		if err != nil {
			return nil, err
		}
		aggregate, err := parseFloat(cellValue(row, aggregateIndex))
		if err != nil {
			return nil, err
		}
		result = append(result, analysisRow{
			Timestamp: timestamp,
			Aggregate: aggregate,
		})
	}
	return result, nil
}

func startOfWeek(ts time.Time) time.Time {
	weekday := int(ts.Weekday())
	if weekday == 0 {
		weekday = 7
	}
	start := time.Date(ts.Year(), ts.Month(), ts.Day(), 0, 0, 0, 0, ts.Location())
	return start.AddDate(0, 0, -(weekday - 1))
}

func classifyPeakValley(ts time.Time, config domain.PeakValleyConfig) string {
	minute := ts.Hour()*60 + ts.Minute()
	for _, period := range config.Peak {
		if minuteInPeriod(minute, period) {
			return "peak"
		}
	}
	for _, period := range config.Valley {
		if minuteInPeriod(minute, period) {
			return "valley"
		}
	}
	return "flat"
}

func minuteInPeriod(minute int, period string) bool {
	parts := strings.Split(strings.TrimSpace(period), "-")
	if len(parts) != 2 {
		return false
	}

	startMinute, ok := parseClockMinute(parts[0])
	if !ok {
		return false
	}
	endMinute, ok := parseClockMinute(parts[1])
	if !ok {
		return false
	}

	if startMinute == endMinute {
		return true
	}
	if startMinute < endMinute {
		return minute >= startMinute && minute < endMinute
	}
	return minute >= startMinute || minute < endMinute
}

func parseClockMinute(raw string) (int, bool) {
	parts := strings.Split(strings.TrimSpace(raw), ":")
	if len(parts) != 2 {
		return 0, false
	}
	hour := 0
	minute := 0
	var err error
	if hour, err = parseInt(parts[0]); err != nil {
		return 0, false
	}
	if minute, err = parseInt(parts[1]); err != nil {
		return 0, false
	}
	if hour < 0 || hour > 23 || minute < 0 || minute > 59 {
		return 0, false
	}
	return hour*60 + minute, true
}

func parseInt(raw string) (int, error) {
	value := strings.TrimSpace(raw)
	if value == "" {
		return 0, errors.New("空整数")
	}
	return strconv.Atoi(value)
}

func (s *AnalysisService) logInfo(message string, fields ...zap.Field) {
	if s.logger != nil {
		s.logger.Info(message, fields...)
	}
}
