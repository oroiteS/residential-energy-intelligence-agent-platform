package service

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"mime/multipart"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/xuri/excelize/v2"
	"go.uber.org/zap"
	"gorm.io/gorm"

	"residential-energy-intelligence-agent-platform/internal/config"
	"residential-energy-intelligence-agent-platform/internal/domain"
	"residential-energy-intelligence-agent-platform/internal/job"
	"residential-energy-intelligence-agent-platform/internal/repository"
	"residential-energy-intelligence-agent-platform/pkg/apperror"
)

type DatasetService struct {
	cfg             *config.AppConfig
	repo            repository.DatasetRepository
	analysisRepo    repository.AnalysisResultRepository
	forecastRepo    repository.ForecastResultRepository
	adviceRepo      repository.EnergyAdviceRepository
	reportRepo      repository.ReportRepository
	executor        job.Executor
	analysisService *AnalysisService
	logger          *zap.Logger
}

type ImportDatasetInput struct {
	Name          string
	Description   string
	HouseholdID   string
	Unit          string
	ColumnMapping string
	File          *multipart.FileHeader
}

type DatasetListParams struct {
	Page     int
	PageSize int
	Status   string
	Keyword  string
}

type DatasetListResult struct {
	Items      []map[string]any `json:"items"`
	Pagination map[string]any   `json:"pagination"`
}

type DatasetDetailResult struct {
	Dataset        map[string]any                `json:"dataset"`
	QualitySummary *domain.DatasetQualitySummary `json:"quality_summary"`
}

type datasetParsedTable struct {
	Headers []string
	Rows    [][]string
}

type datasetProcessResult struct {
	ProcessedRows    []processedDatasetRow
	FeatureCols      []string
	ResolvedMapping  map[string]string
	ApplianceColumns []string
	QualityReport    domain.DatasetQualityReport
	TimeStart        time.Time
	TimeEnd          time.Time
}

type processedDatasetRow struct {
	Timestamp            time.Time
	Aggregate            float64
	ActiveApplianceCount int
	BurstEventCount      int
}

type bucketAggregate struct {
	Timestamp     time.Time
	Count         int
	AggregateSum  float64
	ApplianceSums []float64
}

func NewDatasetService(
	cfg *config.AppConfig,
	repo repository.DatasetRepository,
	analysisRepo repository.AnalysisResultRepository,
	forecastRepo repository.ForecastResultRepository,
	adviceRepo repository.EnergyAdviceRepository,
	reportRepo repository.ReportRepository,
	executor job.Executor,
	analysisService *AnalysisService,
	logger *zap.Logger,
) *DatasetService {
	return &DatasetService{
		cfg:             cfg,
		repo:            repo,
		analysisRepo:    analysisRepo,
		forecastRepo:    forecastRepo,
		adviceRepo:      adviceRepo,
		reportRepo:      reportRepo,
		executor:        executor,
		analysisService: analysisService,
		logger:          logger,
	}
}

func (s *DatasetService) Import(ctx context.Context, input ImportDatasetInput) (map[string]any, *apperror.AppError) {
	if s.repo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持导入数据集", nil)
	}
	if s.executor == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "任务执行器未初始化，暂不支持导入数据集", nil)
	}

	unit, appErr := normalizeImportUnit(input.Unit)
	if appErr != nil {
		return nil, appErr
	}
	if strings.TrimSpace(input.Name) == "" {
		return nil, apperror.Unprocessable("INVALID_REQUEST", "name 不能为空", nil)
	}
	if input.File == nil {
		return nil, apperror.Unprocessable("INVALID_REQUEST", "file 不能为空", nil)
	}

	rawMapping, appErr := parseInputColumnMapping(input.ColumnMapping)
	if appErr != nil {
		return nil, appErr
	}

	rawPath, ext, appErr := s.saveUploadedFile(input.File)
	if appErr != nil {
		return nil, appErr
	}

	table, err := readDatasetTable(rawPath, ext)
	if err != nil {
		_ = os.Remove(rawPath)
		return nil, apperror.Unprocessable("UNSUPPORTED_FILE_TYPE", err.Error(), nil)
	}

	resolvedMapping, applianceColumns, appErr := resolveColumnMapping(table.Headers, rawMapping)
	if appErr != nil {
		_ = os.Remove(rawPath)
		return nil, appErr
	}

	featureColsJSON, err := json.Marshal(table.Headers)
	if err != nil {
		_ = os.Remove(rawPath)
		return nil, apperror.Internal(err)
	}
	columnMappingJSON, err := json.Marshal(resolvedMapping)
	if err != nil {
		_ = os.Remove(rawPath)
		return nil, apperror.Internal(err)
	}

	record := &domain.DatasetRecord{
		Name:          strings.TrimSpace(input.Name),
		RawFilePath:   rawPath,
		FeatureCols:   featureColsJSON,
		ColumnMapping: columnMappingJSON,
		Status:        "processing",
	}
	if description := strings.TrimSpace(input.Description); description != "" {
		record.Description = &description
	}
	if householdID := strings.TrimSpace(input.HouseholdID); householdID != "" {
		record.HouseholdID = &householdID
	}

	if err := s.repo.Create(ctx, record); err != nil {
		_ = os.Remove(rawPath)
		return nil, apperror.Internal(err)
	}

	datasetID := record.ID
	s.executor.Submit(context.Background(), func(_ context.Context) {
		s.processImport(datasetID, rawPath, ext, unit, resolvedMapping, applianceColumns, table.Headers, record.HouseholdID)
	})

	s.logInfo("数据集导入任务已受理", zap.Uint64("dataset_id", datasetID), zap.String("unit", unit))
	return datasetSummaryDTO(record), nil
}

func (s *DatasetService) List(ctx context.Context, params DatasetListParams) (*DatasetListResult, *apperror.AppError) {
	if s.repo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持读取数据集", nil)
	}

	items, total, err := s.repo.List(ctx, domain.DatasetListFilter{
		Page:     params.Page,
		PageSize: params.PageSize,
		Status:   params.Status,
		Keyword:  params.Keyword,
	})
	if err != nil {
		return nil, apperror.Internal(err)
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

	resultItems := make([]map[string]any, 0, len(items))
	for _, item := range items {
		resultItems = append(resultItems, datasetSummaryDTO(&item))
	}

	return &DatasetListResult{
		Items: resultItems,
		Pagination: map[string]any{
			"page":      page,
			"page_size": pageSize,
			"total":     total,
		},
	}, nil
}

func (s *DatasetService) GetDetail(ctx context.Context, id uint64) (*DatasetDetailResult, *apperror.AppError) {
	if s.repo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持读取数据集", nil)
	}

	record, err := s.repo.GetByID(ctx, id)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, apperror.NotFound("DATASET_NOT_FOUND", "数据集不存在", map[string]any{"id": id})
		}
		return nil, apperror.Internal(err)
	}

	detail := datasetDetailDTO(record)
	qualitySummary, err := loadQualitySummary(record.QualityReportPath)
	if err != nil {
		s.logWarn("读取质量报告失败", zap.Uint64("dataset_id", id), zap.Error(err))
	}

	return &DatasetDetailResult{
		Dataset:        detail,
		QualitySummary: qualitySummary,
	}, nil
}

func (s *DatasetService) Delete(ctx context.Context, id uint64) *apperror.AppError {
	if s.repo == nil {
		return apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持删除数据集", nil)
	}

	record, err := s.repo.GetByID(ctx, id)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return apperror.NotFound("DATASET_NOT_FOUND", "数据集不存在", map[string]any{"id": id})
		}
		return apperror.Internal(err)
	}

	filePaths := make([]string, 0, 8)
	filePaths = appendIfNotEmpty(filePaths, record.RawFilePath)
	filePaths = appendIfNotEmptyPtr(filePaths, record.ProcessedFilePath)
	filePaths = appendIfNotEmptyPtr(filePaths, record.QualityReportPath)

	if s.analysisRepo != nil {
		if analysis, getErr := s.analysisRepo.GetByDatasetID(ctx, id); getErr == nil && analysis.DetailPath != nil {
			filePaths = appendIfNotEmptyPtr(filePaths, analysis.DetailPath)
		}
	}
	if s.adviceRepo != nil {
		if advices, listErr := s.adviceRepo.ListByDatasetID(ctx, id, ""); listErr == nil {
			for _, advice := range advices {
				filePaths = appendIfNotEmpty(filePaths, advice.ContentPath)
			}
		}
	}
	if s.forecastRepo != nil {
		if forecasts, listErr := s.forecastRepo.ListAllByDatasetID(ctx, id); listErr == nil {
			for _, forecast := range forecasts {
				filePaths = appendIfNotEmpty(filePaths, forecast.DetailPath)
			}
		}
	}
	if s.reportRepo != nil {
		if reports, listErr := s.reportRepo.ListByDatasetID(ctx, id); listErr == nil {
			for _, report := range reports {
				filePaths = appendIfNotEmpty(filePaths, report.FilePath)
			}
		}
	}

	for _, path := range uniqueStrings(filePaths) {
		if err := deleteFileIfExists(path); err != nil {
			return apperror.Conflict("INVALID_REQUEST", "文件删除失败，已阻止数据集删除", map[string]any{
				"id":        id,
				"file_path": path,
				"error":     err.Error(),
			})
		}
	}

	if err := s.repo.Delete(ctx, id); err != nil {
		return apperror.Internal(err)
	}

	s.logInfo("数据集删除完成", zap.Uint64("dataset_id", id))
	return nil
}

func (s *DatasetService) processImport(
	datasetID uint64,
	rawPath string,
	ext string,
	unit string,
	resolvedMapping map[string]string,
	applianceColumns []string,
	headers []string,
	householdID *string,
) {
	record, err := s.repo.GetByID(context.Background(), datasetID)
	if err != nil {
		s.logError("导入任务读取数据集失败", zap.Uint64("dataset_id", datasetID), zap.Error(err))
		return
	}

	result, err := runDatasetPreprocess(rawPath, ext, unit, resolvedMapping, applianceColumns, headers)
	if err != nil {
		s.markDatasetError(record, err)
		return
	}

	processedPath, err := s.writeProcessedCSV(datasetID, result.ProcessedRows, householdID)
	if err != nil {
		s.markDatasetError(record, err)
		return
	}

	qualityPath, err := s.writeQualityReport(datasetID, result.QualityReport)
	if err != nil {
		s.markDatasetError(record, err)
		return
	}

	featureColsJSON, err := json.Marshal(result.FeatureCols)
	if err != nil {
		s.markDatasetError(record, err)
		return
	}
	columnMappingJSON, err := json.Marshal(result.ResolvedMapping)
	if err != nil {
		s.markDatasetError(record, err)
		return
	}

	record.Status = "ready"
	record.RowCount = uint32(len(result.ProcessedRows))
	record.TimeStart = &result.TimeStart
	record.TimeEnd = &result.TimeEnd
	record.FeatureCols = featureColsJSON
	record.ColumnMapping = columnMappingJSON
	record.ProcessedFilePath = &processedPath
	record.QualityReportPath = &qualityPath
	record.ErrorMessage = nil

	if err := s.repo.Update(context.Background(), record); err != nil {
		s.logError("更新导入结果失败", zap.Uint64("dataset_id", datasetID), zap.Error(err))
		return
	}

	if s.analysisService != nil {
		if _, appErr := s.analysisService.Generate(context.Background(), datasetID, false); appErr != nil {
			s.markDatasetError(record, appErr)
			return
		}
	}

	s.logInfo("数据集导入完成", zap.Uint64("dataset_id", datasetID), zap.Int("processed_rows", len(result.ProcessedRows)))
}

func (s *DatasetService) markDatasetError(record *domain.DatasetRecord, processErr error) {
	if record == nil {
		return
	}
	message := strings.TrimSpace(processErr.Error())
	record.Status = "error"
	record.ErrorMessage = &message
	if err := s.repo.Update(context.Background(), record); err != nil {
		s.logError("更新数据集错误状态失败", zap.Uint64("dataset_id", record.ID), zap.Error(err))
		return
	}
	s.logError("数据集导入失败", zap.Uint64("dataset_id", record.ID), zap.Error(processErr))
}

func (s *DatasetService) saveUploadedFile(fileHeader *multipart.FileHeader) (string, string, *apperror.AppError) {
	ext := strings.ToLower(filepath.Ext(fileHeader.Filename))
	if ext != ".csv" && ext != ".xlsx" {
		return "", "", apperror.Unprocessable("UNSUPPORTED_FILE_TYPE", "仅支持 csv 或 xlsx 文件", nil)
	}

	if err := os.MkdirAll(s.cfg.DataUploadDir, 0o755); err != nil {
		return "", "", apperror.Internal(err)
	}

	filename := fmt.Sprintf("dataset_%d%s", time.Now().UnixNano(), ext)
	relativePath := filepath.ToSlash(filepath.Join(s.cfg.DataUploadDir, filename))

	src, err := fileHeader.Open()
	if err != nil {
		return "", "", apperror.Internal(err)
	}
	defer src.Close()

	dst, err := os.Create(relativePath)
	if err != nil {
		return "", "", apperror.Internal(err)
	}
	defer dst.Close()

	if _, err := dst.ReadFrom(src); err != nil {
		return "", "", apperror.Internal(err)
	}

	return relativePath, ext, nil
}

func (s *DatasetService) writeProcessedCSV(datasetID uint64, rows []processedDatasetRow, householdID *string) (string, error) {
	processedDir := filepath.Join(s.cfg.DataUploadDir, "processed")
	if err := os.MkdirAll(processedDir, 0o755); err != nil {
		return "", err
	}

	path := filepath.ToSlash(filepath.Join(processedDir, fmt.Sprintf("dataset_%d_15min.csv", datasetID)))
	file, err := os.Create(path)
	if err != nil {
		return "", err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	if err := writer.Write([]string{
		"house_id",
		"timestamp",
		"date",
		"slot_index",
		"aggregate",
		"active_appliance_count",
		"burst_event_count",
		"is_weekend",
	}); err != nil {
		return "", err
	}

	houseID := ""
	if householdID != nil {
		houseID = *householdID
	}

	for _, row := range rows {
		slotIndex := row.Timestamp.Hour()*4 + row.Timestamp.Minute()/15
		record := []string{
			houseID,
			row.Timestamp.Format(time.RFC3339),
			row.Timestamp.Format("2006-01-02"),
			strconv.Itoa(slotIndex),
			strconv.FormatFloat(row.Aggregate, 'f', 6, 64),
			strconv.Itoa(row.ActiveApplianceCount),
			strconv.Itoa(row.BurstEventCount),
			strconv.FormatBool(isWeekend(row.Timestamp)),
		}
		if err := writer.Write(record); err != nil {
			return "", err
		}
	}

	return path, writer.Error()
}

func (s *DatasetService) writeQualityReport(datasetID uint64, report domain.DatasetQualityReport) (string, error) {
	qualityDir := filepath.Join(s.cfg.OutputRootDir, "quality")
	if err := os.MkdirAll(qualityDir, 0o755); err != nil {
		return "", err
	}

	path := filepath.ToSlash(filepath.Join(qualityDir, fmt.Sprintf("dataset_%d.json", datasetID)))
	report.DatasetID = datasetID
	content, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(path, content, 0o644); err != nil {
		return "", err
	}
	return path, nil
}

func runDatasetPreprocess(
	rawPath string,
	ext string,
	unit string,
	resolvedMapping map[string]string,
	applianceColumns []string,
	headers []string,
) (*datasetProcessResult, error) {
	table, err := readDatasetTable(rawPath, ext)
	if err != nil {
		return nil, err
	}

	columnIndex := make(map[string]int, len(table.Headers))
	for idx, header := range table.Headers {
		columnIndex[header] = idx
	}

	timestampColumn, err := singleMappedColumn(resolvedMapping, "timestamp")
	if err != nil {
		return nil, err
	}
	aggregateColumn, err := singleMappedColumn(resolvedMapping, "aggregate")
	if err != nil {
		return nil, err
	}
	issuesColumn := optionalMappedColumn(resolvedMapping, "issues")

	duplicateCount := 0
	seenTimestamps := make(map[int64]struct{})
	buckets := make(map[int64]*bucketAggregate)
	inputRowCount := 0

	for _, row := range table.Rows {
		if isEmptyRow(row) {
			continue
		}
		inputRowCount++

		timestampValue := cellValue(row, columnIndex[timestampColumn])
		timestamp, err := parseFlexibleTime(timestampValue)
		if err != nil {
			continue
		}

		if issuesColumn != "" {
			issuesValue := strings.TrimSpace(cellValue(row, columnIndex[issuesColumn]))
			if issuesValue == "1" {
				continue
			}
		}

		aggregateValue, err := parseFloat(cellValue(row, columnIndex[aggregateColumn]))
		if err != nil {
			continue
		}

		applianceValues := make([]float64, len(applianceColumns))
		for i, column := range applianceColumns {
			value, parseErr := parseOptionalFloat(cellValue(row, columnIndex[column]))
			if parseErr == nil {
				applianceValues[i] = value
			}
		}

		if _, exists := seenTimestamps[timestamp.Unix()]; exists {
			duplicateCount++
		}
		seenTimestamps[timestamp.Unix()] = struct{}{}

		bucketTime := floorTo15Min(timestamp)
		key := bucketTime.Unix()
		bucket, exists := buckets[key]
		if !exists {
			bucket = &bucketAggregate{
				Timestamp:     bucketTime,
				ApplianceSums: make([]float64, len(applianceColumns)),
			}
			buckets[key] = bucket
		}
		bucket.Count++
		bucket.AggregateSum += aggregateValue
		for i := range applianceValues {
			bucket.ApplianceSums[i] += applianceValues[i]
		}
	}

	if len(buckets) == 0 {
		return nil, errors.New("导入文件中未解析到有效数据行")
	}

	keys := make([]int64, 0, len(buckets))
	for key := range buckets {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })

	start := time.Unix(keys[0], 0).In(time.Local)
	end := time.Unix(keys[len(keys)-1], 0).In(time.Local)
	expectedRows := int(end.Sub(start)/(15*time.Minute)) + 1
	if expectedRows < 0 {
		expectedRows = len(keys)
	}

	rows := make([]processedDatasetRow, 0, expectedRows)
	applianceSeries := make([][]float64, len(applianceColumns))
	for i := range applianceSeries {
		applianceSeries[i] = make([]float64, 0, expectedRows)
	}

	current := start
	missingRows := 0
	for !current.After(end) {
		key := current.Unix()
		row := processedDatasetRow{Timestamp: current}
		if bucket, exists := buckets[key]; exists {
			aggregateKWH, appliancePowerW := convertBucket(unit, bucket)
			row.Aggregate = aggregateKWH
			row.ActiveApplianceCount = countActiveAppliances(appliancePowerW)
			for i := range appliancePowerW {
				applianceSeries[i] = append(applianceSeries[i], appliancePowerW[i])
			}
		} else {
			missingRows++
			for i := range applianceSeries {
				applianceSeries[i] = append(applianceSeries[i], 0)
			}
		}
		rows = append(rows, row)
		current = current.Add(15 * time.Minute)
	}

	fillBurstEventCount(rows, applianceSeries)

	missingRate := 0.0
	if len(rows) > 0 {
		missingRate = float64(missingRows) / float64(len(rows))
	}

	cleaningStrategies := []string{"15 分钟粒度重采样", "重复时间戳合并"}
	if missingRows > 0 {
		cleaningStrategies = append(cleaningStrategies, "缺失窗口补零")
	}
	if issuesColumn != "" {
		cleaningStrategies = append(cleaningStrategies, "Issues=1 记录剔除")
	}
	if len(applianceColumns) > 0 {
		cleaningStrategies = append(cleaningStrategies, "按识别电器列构造辅助特征")
	}

	qualitySummary := domain.DatasetQualitySummary{
		MissingRate:        roundFloat(missingRate, 4),
		DuplicateCount:     duplicateCount,
		SamplingInterval:   "15min",
		CleaningStrategies: cleaningStrategies,
	}

	return &datasetProcessResult{
		ProcessedRows:    rows,
		FeatureCols:      headers,
		ResolvedMapping:  resolvedMapping,
		ApplianceColumns: applianceColumns,
		QualityReport: domain.DatasetQualityReport{
			InputRowCount:         inputRowCount,
			ProcessedRowCount:     len(rows),
			MissingRate:           qualitySummary.MissingRate,
			DuplicateCount:        duplicateCount,
			SamplingInterval:      "15min",
			CleaningStrategies:    cleaningStrategies,
			FeatureCols:           headers,
			ResolvedColumnMapping: resolvedMapping,
			DetectedApplianceCols: applianceColumns,
			QualitySummary:        qualitySummary,
		},
		TimeStart: start,
		TimeEnd:   end,
	}, nil
}

func readDatasetTable(path string, ext string) (*datasetParsedTable, error) {
	switch ext {
	case ".csv":
		return readCSVTable(path)
	case ".xlsx":
		return readXLSXTable(path)
	default:
		return nil, errors.New("不支持的文件类型")
	}
}

func readCSVTable(path string) (*datasetParsedTable, error) {
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
	if len(rows) == 0 {
		return nil, errors.New("CSV 文件为空")
	}

	headers := trimCells(rows[0])
	return &datasetParsedTable{
		Headers: headers,
		Rows:    rows[1:],
	}, nil
}

func readXLSXTable(path string) (*datasetParsedTable, error) {
	file, err := excelize.OpenFile(path)
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = file.Close()
	}()

	sheets := file.GetSheetList()
	if len(sheets) == 0 {
		return nil, errors.New("xlsx 文件不包含工作表")
	}

	rows, err := file.GetRows(sheets[0])
	if err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, errors.New("xlsx 文件为空")
	}

	headers := trimCells(rows[0])
	return &datasetParsedTable{
		Headers: headers,
		Rows:    rows[1:],
	}, nil
}

func parseInputColumnMapping(raw string) (map[string]string, *apperror.AppError) {
	if strings.TrimSpace(raw) == "" {
		return map[string]string{}, nil
	}

	var mapping map[string]string
	if err := json.Unmarshal([]byte(raw), &mapping); err != nil {
		return nil, apperror.InvalidRequest("column_mapping 不是合法 JSON", map[string]any{"error": err.Error()})
	}

	normalized := make(map[string]string, len(mapping))
	for source, semantic := range mapping {
		source = strings.TrimSpace(source)
		semantic = normalizeSemantic(semantic)
		if source == "" || semantic == "" {
			continue
		}
		switch semantic {
		case "timestamp", "aggregate", "issues", "appliance":
			normalized[source] = semantic
		default:
			return nil, apperror.InvalidRequest("column_mapping 存在不支持的语义值", map[string]any{
				"source_column": source,
				"semantic":      semantic,
			})
		}
	}
	return normalized, nil
}

func resolveColumnMapping(headers []string, provided map[string]string) (map[string]string, []string, *apperror.AppError) {
	resolved := make(map[string]string, len(headers))
	headerSet := make(map[string]struct{}, len(headers))
	for _, header := range headers {
		headerSet[header] = struct{}{}
	}
	for source, semantic := range provided {
		if _, exists := headerSet[source]; !exists {
			return nil, nil, apperror.InvalidRequest("column_mapping 包含不存在的原始列名", map[string]any{
				"source_column": source,
			})
		}
		resolved[source] = normalizeSemantic(semantic)
	}

	for _, header := range headers {
		if _, exists := resolved[header]; exists {
			continue
		}
		semantic := detectSemantic(header)
		if semantic != "" {
			resolved[header] = semantic
		}
	}

	hasTimestamp := false
	hasAggregate := false
	applianceColumns := make([]string, 0)
	for _, header := range headers {
		switch resolved[header] {
		case "timestamp":
			hasTimestamp = true
		case "aggregate":
			hasAggregate = true
		case "appliance":
			applianceColumns = append(applianceColumns, header)
		}
	}

	if !hasTimestamp || !hasAggregate {
		return nil, nil, apperror.Unprocessable("COLUMN_MAPPING_REQUIRED", "未识别到时间戳列或总用电量列，请补充 column_mapping", nil)
	}

	return resolved, applianceColumns, nil
}

func datasetSummaryDTO(record *domain.DatasetRecord) map[string]any {
	return map[string]any{
		"id":           record.ID,
		"name":         record.Name,
		"description":  nullableString(record.Description),
		"household_id": nullableString(record.HouseholdID),
		"row_count":    record.RowCount,
		"time_start":   record.TimeStart,
		"time_end":     record.TimeEnd,
		"status":       record.Status,
		"created_at":   record.CreatedAt,
		"updated_at":   record.UpdatedAt,
	}
}

func datasetDetailDTO(record *domain.DatasetRecord) map[string]any {
	data := datasetSummaryDTO(record)
	data["raw_file_path"] = record.RawFilePath
	data["processed_file_path"] = nullableString(record.ProcessedFilePath)
	data["feature_cols"] = decodeJSONField(record.FeatureCols)
	data["column_mapping"] = decodeJSONField(record.ColumnMapping)
	data["quality_report_path"] = nullableString(record.QualityReportPath)
	data["error_message"] = nullableString(record.ErrorMessage)
	return data
}

func loadQualitySummary(path *string) (*domain.DatasetQualitySummary, error) {
	if path == nil || strings.TrimSpace(*path) == "" {
		return nil, nil
	}
	content, err := os.ReadFile(*path)
	if err != nil {
		return nil, err
	}

	var report domain.DatasetQualityReport
	if err := json.Unmarshal(content, &report); err != nil {
		return nil, err
	}
	return &report.QualitySummary, nil
}

func normalizeImportUnit(unit string) (string, *apperror.AppError) {
	normalized := strings.ToLower(strings.TrimSpace(unit))
	if normalized == "" {
		return "w", nil
	}
	switch normalized {
	case "kwh", "wh", "w":
		return normalized, nil
	default:
		return "", apperror.Unprocessable("INVALID_REQUEST", "unit 仅支持 kwh、wh 或 w", nil)
	}
}

func normalizeSemantic(value string) string {
	normalized := strings.ToLower(strings.TrimSpace(value))
	switch normalized {
	case "value":
		return "aggregate"
	default:
		return normalized
	}
}

func detectSemantic(header string) string {
	normalized := strings.ToLower(strings.TrimSpace(header))
	switch normalized {
	case "timestamp", "time", "datetime", "date_time", "日期时间", "时间":
		return "timestamp"
	case "aggregate", "kwh", "usage", "value", "电量", "总用电量", "load":
		return "aggregate"
	case "issues", "issue":
		return "issues"
	}

	if strings.HasPrefix(normalized, "appliance") ||
		strings.HasPrefix(normalized, "submeter") ||
		strings.HasPrefix(normalized, "device") {
		return "appliance"
	}
	return ""
}

func singleMappedColumn(mapping map[string]string, semantic string) (string, error) {
	for source, value := range mapping {
		if value == semantic {
			return source, nil
		}
	}
	return "", fmt.Errorf("未找到 %s 对应列", semantic)
}

func optionalMappedColumn(mapping map[string]string, semantic string) string {
	for source, value := range mapping {
		if value == semantic {
			return source
		}
	}
	return ""
}

func trimCells(cells []string) []string {
	result := make([]string, len(cells))
	for i, cell := range cells {
		result[i] = strings.TrimSpace(cell)
	}
	return result
}

func isEmptyRow(row []string) bool {
	for _, cell := range row {
		if strings.TrimSpace(cell) != "" {
			return false
		}
	}
	return true
}

func cellValue(row []string, index int) string {
	if index < 0 || index >= len(row) {
		return ""
	}
	return strings.TrimSpace(row[index])
}

func parseFlexibleTime(raw string) (time.Time, error) {
	value := strings.TrimSpace(raw)
	if value == "" {
		return time.Time{}, errors.New("空时间")
	}

	layouts := []string{
		time.RFC3339,
		"2006-01-02 15:04:05",
		"2006/01/02 15:04:05",
		"2006-01-02 15:04",
		"2006/01/02 15:04",
		"2006-01-02",
		"2006/01/02",
	}
	for _, layout := range layouts {
		if ts, err := time.ParseInLocation(layout, value, time.Local); err == nil {
			return ts, nil
		}
	}

	excelSerial, err := strconv.ParseFloat(value, 64)
	if err == nil && excelSerial > 0 {
		base := time.Date(1899, 12, 30, 0, 0, 0, 0, time.Local)
		days := time.Duration(excelSerial * 24 * float64(time.Hour))
		return base.Add(days), nil
	}

	return time.Time{}, errors.New("无法解析时间")
}

func parseFloat(raw string) (float64, error) {
	value := strings.TrimSpace(strings.ReplaceAll(raw, ",", ""))
	if value == "" {
		return 0, errors.New("空数值")
	}
	return strconv.ParseFloat(value, 64)
}

func parseOptionalFloat(raw string) (float64, error) {
	if strings.TrimSpace(raw) == "" {
		return 0, nil
	}
	return parseFloat(raw)
}

func floorTo15Min(ts time.Time) time.Time {
	return ts.Truncate(15 * time.Minute)
}

func convertBucket(unit string, bucket *bucketAggregate) (float64, []float64) {
	appliancePowerW := make([]float64, len(bucket.ApplianceSums))
	switch unit {
	case "w":
		meanPowerW := bucket.AggregateSum / float64(bucket.Count)
		for i := range bucket.ApplianceSums {
			appliancePowerW[i] = bucket.ApplianceSums[i] / float64(bucket.Count)
		}
		return meanPowerW, appliancePowerW
	case "wh":
		for i := range bucket.ApplianceSums {
			appliancePowerW[i] = bucket.ApplianceSums[i] * 4
		}
		return bucket.AggregateSum * 4, appliancePowerW
	default:
		for i := range bucket.ApplianceSums {
			appliancePowerW[i] = bucket.ApplianceSums[i] * 4000
		}
		return bucket.AggregateSum * 4000, appliancePowerW
	}
}

func countActiveAppliances(appliancePowerW []float64) int {
	count := 0
	for _, value := range appliancePowerW {
		if value > 10 {
			count++
		}
	}
	return count
}

func fillBurstEventCount(rows []processedDatasetRow, applianceSeries [][]float64) {
	for applianceIndex := range applianceSeries {
		series := applianceSeries[applianceIndex]
		for i := 1; i < len(series); i++ {
			if series[i]-series[i-1] <= 300 {
				continue
			}
			runLength := 1
			for j := i + 1; j < len(series) && series[j] > 10; j++ {
				runLength++
				if runLength > 2 {
					break
				}
			}
			if runLength <= 2 {
				rows[i].BurstEventCount++
			}
		}
	}
}

func roundFloat(value float64, precision int) float64 {
	format := "%." + strconv.Itoa(precision) + "f"
	text := fmt.Sprintf(format, value)
	rounded, err := strconv.ParseFloat(text, 64)
	if err != nil {
		return value
	}
	return rounded
}

func nullableString(value *string) any {
	if value == nil {
		return nil
	}
	return *value
}

func decodeJSONField(raw []byte) any {
	if len(raw) == 0 {
		return nil
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil
	}
	return value
}

func isWeekend(ts time.Time) bool {
	return ts.Weekday() == time.Saturday || ts.Weekday() == time.Sunday
}

func appendIfNotEmpty(items []string, value string) []string {
	if strings.TrimSpace(value) == "" {
		return items
	}
	return append(items, value)
}

func appendIfNotEmptyPtr(items []string, value *string) []string {
	if value == nil {
		return items
	}
	return appendIfNotEmpty(items, *value)
}

func uniqueStrings(items []string) []string {
	seen := make(map[string]struct{}, len(items))
	result := make([]string, 0, len(items))
	for _, item := range items {
		if _, exists := seen[item]; exists {
			continue
		}
		seen[item] = struct{}{}
		result = append(result, item)
	}
	return result
}

func deleteFileIfExists(path string) error {
	if strings.TrimSpace(path) == "" {
		return nil
	}
	err := os.Remove(path)
	if err == nil || errors.Is(err, os.ErrNotExist) {
		return nil
	}
	return err
}

func (s *DatasetService) logInfo(message string, fields ...zap.Field) {
	if s.logger != nil {
		s.logger.Info(message, fields...)
	}
}

func (s *DatasetService) logWarn(message string, fields ...zap.Field) {
	if s.logger != nil {
		s.logger.Warn(message, fields...)
	}
}

func (s *DatasetService) logError(message string, fields ...zap.Field) {
	if s.logger != nil {
		s.logger.Error(message, fields...)
	}
}
