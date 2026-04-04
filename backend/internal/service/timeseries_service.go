package service

import (
	"context"
	"encoding/csv"
	"errors"
	"os"
	"sort"
	"strings"
	"time"

	"gorm.io/gorm"

	"residential-energy-intelligence-agent-platform/internal/domain"
	"residential-energy-intelligence-agent-platform/internal/repository"
	"residential-energy-intelligence-agent-platform/pkg/apperror"
)

type processedFeatureRow struct {
	Timestamp            time.Time
	Aggregate            float64
	ActiveApplianceCount int
	BurstEventCount      int
}

func readProcessedFeatureRows(path string) ([]processedFeatureRow, error) {
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

	requiredColumns := []string{"timestamp", "aggregate", "active_appliance_count", "burst_event_count"}
	for _, column := range requiredColumns {
		if _, exists := headerIndex[column]; !exists {
			return nil, errors.New("processed csv 缺少必要列: " + column)
		}
	}

	featureRows := make([]processedFeatureRow, 0, len(rows)-1)
	for _, row := range rows[1:] {
		if isEmptyRow(row) {
			continue
		}
		timestamp, err := parseFlexibleTime(cellValue(row, headerIndex["timestamp"]))
		if err != nil {
			return nil, err
		}
		aggregate, err := parseFloat(cellValue(row, headerIndex["aggregate"]))
		if err != nil {
			return nil, err
		}
		active, err := strconvAtoiSafe(cellValue(row, headerIndex["active_appliance_count"]))
		if err != nil {
			return nil, err
		}
		burst, err := strconvAtoiSafe(cellValue(row, headerIndex["burst_event_count"]))
		if err != nil {
			return nil, err
		}
		featureRows = append(featureRows, processedFeatureRow{
			Timestamp:            timestamp,
			Aggregate:            aggregate,
			ActiveApplianceCount: active,
			BurstEventCount:      burst,
		})
	}

	sort.Slice(featureRows, func(i, j int) bool {
		return featureRows[i].Timestamp.Before(featureRows[j].Timestamp)
	})
	return featureRows, nil
}

func filterRowsInRange(rows []processedFeatureRow, start, end time.Time) []processedFeatureRow {
	filtered := make([]processedFeatureRow, 0)
	for _, row := range rows {
		if row.Timestamp.Before(start) || row.Timestamp.After(end) {
			continue
		}
		filtered = append(filtered, row)
	}
	return filtered
}

func filterRowsByExclusiveEnd(rows []processedFeatureRow, start, endExclusive time.Time) []processedFeatureRow {
	filtered := make([]processedFeatureRow, 0)
	for _, row := range rows {
		if row.Timestamp.Before(start) || !row.Timestamp.Before(endExclusive) {
			continue
		}
		filtered = append(filtered, row)
	}
	return filtered
}

func strconvAtoiSafe(raw string) (int, error) {
	value, err := parseFloat(raw)
	if err != nil {
		return 0, err
	}
	return int(value), nil
}

func getReadyDatasetRecord(ctx context.Context, datasetRepo repository.DatasetRepository, datasetID uint64) (*domain.DatasetRecord, *apperror.AppError) {
	if datasetRepo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持读取数据集", nil)
	}

	dataset, err := datasetRepo.GetByID(ctx, datasetID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, apperror.NotFound("DATASET_NOT_FOUND", "数据集不存在", map[string]any{"id": datasetID})
		}
		return nil, apperror.Internal(err)
	}
	if dataset.Status != "ready" {
		return nil, apperror.Conflict("DATASET_NOT_READY", "数据集尚未处理完成", map[string]any{"id": datasetID})
	}
	if dataset.ProcessedFilePath == nil || strings.TrimSpace(*dataset.ProcessedFilePath) == "" {
		return nil, apperror.NotFound("DATASET_NOT_READY", "数据集缺少清洗后的时序文件", map[string]any{"id": datasetID})
	}
	return dataset, nil
}
