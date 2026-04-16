package service

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	"go.uber.org/zap"
	"gorm.io/gorm"

	"residential-energy-intelligence-agent-platform/internal/config"
	"residential-energy-intelligence-agent-platform/internal/domain"
	"residential-energy-intelligence-agent-platform/internal/integration/agentclient"
	"residential-energy-intelligence-agent-platform/internal/repository"
	"residential-energy-intelligence-agent-platform/pkg/apperror"
)

type AgentService struct {
	cfg                 *config.AppConfig
	datasetRepo         repository.DatasetRepository
	sessionRepo         repository.ChatSessionRepository
	messageRepo         repository.ChatMessageRepository
	analysisRepo        repository.AnalysisResultRepository
	classificationRepo  repository.ClassificationResultRepository
	forecastRepo        repository.ForecastResultRepository
	adviceRepo          repository.EnergyAdviceRepository
	systemConfigService *SystemConfigService
	agentClient         agentclient.Client
	logger              *zap.Logger
}

type AgentAskInput struct {
	DatasetID uint64                    `json:"dataset_id"`
	SessionID uint64                    `json:"session_id"`
	Question  string                    `json:"question"`
	History   []agentclient.HistoryItem `json:"history"`
}

type ReportSummary struct {
	Title           string
	Overview        string
	Sections        []agentclient.ReportSection
	Recommendations []string
	Degraded        bool
	ErrorReason     *string
}

func NewAgentService(
	cfg *config.AppConfig,
	datasetRepo repository.DatasetRepository,
	sessionRepo repository.ChatSessionRepository,
	messageRepo repository.ChatMessageRepository,
	analysisRepo repository.AnalysisResultRepository,
	classificationRepo repository.ClassificationResultRepository,
	forecastRepo repository.ForecastResultRepository,
	adviceRepo repository.EnergyAdviceRepository,
	systemConfigService *SystemConfigService,
	agentClient agentclient.Client,
	logger *zap.Logger,
) *AgentService {
	return &AgentService{
		cfg:                 cfg,
		datasetRepo:         datasetRepo,
		sessionRepo:         sessionRepo,
		messageRepo:         messageRepo,
		analysisRepo:        analysisRepo,
		classificationRepo:  classificationRepo,
		forecastRepo:        forecastRepo,
		adviceRepo:          adviceRepo,
		systemConfigService: systemConfigService,
		agentClient:         agentClient,
		logger:              logger,
	}
}

func (s *AgentService) Ask(ctx context.Context, input AgentAskInput) (map[string]any, *apperror.AppError) {
	if s.datasetRepo == nil || s.analysisRepo == nil || s.sessionRepo == nil || s.messageRepo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持智能问答", nil)
	}

	input.Question = strings.TrimSpace(input.Question)
	if input.DatasetID == 0 || input.SessionID == 0 || input.Question == "" {
		return nil, apperror.InvalidRequest("dataset_id、session_id、question 不能为空", map[string]any{
			"dataset_id": input.DatasetID,
			"session_id": input.SessionID,
		})
	}
	if appErr := validateAgentHistory(input.History); appErr != nil {
		return nil, appErr
	}

	sessionRecord, appErr := s.getValidatedSession(ctx, input.DatasetID, input.SessionID)
	if appErr != nil {
		return nil, appErr
	}

	dataset, appErr := getReadyDatasetRecord(ctx, s.datasetRepo, input.DatasetID)
	if appErr != nil {
		return nil, appErr
	}

	analysisRecord, err := s.analysisRepo.GetByDatasetID(ctx, input.DatasetID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, apperror.NotFound("ANALYSIS_NOT_FOUND", "统计分析结果不存在", map[string]any{"dataset_id": input.DatasetID})
		}
		return nil, apperror.Internal(err)
	}

	runtimeConfig, err := s.systemConfigService.Get(ctx)
	if err != nil {
		return nil, apperror.Internal(err)
	}

	contextPayload, appErr := s.buildAgentContext(ctx, dataset, analysisRecord, runtimeConfig)
	if appErr != nil {
		return nil, appErr
	}

	historyItems, appErr := s.loadAskHistory(ctx, input.SessionID, input.History)
	if appErr != nil {
		return nil, appErr
	}
	input.History = historyItems

	userMessage, appErr := s.createUserMessage(ctx, input.SessionID, input.Question)
	if appErr != nil {
		return nil, appErr
	}

	now := time.Now()
	fallbackIntent := inferAgentIntent(input.Question)
	clientResponse, err := s.askAgent(ctx, input, contextPayload)
	if err != nil {
		s.logWarn("智能体调用失败，已回退本地降级回答", zap.Error(err), zap.Uint64("dataset_id", input.DatasetID))
		fallback := buildLocalAgentFallback(input.SessionID, input.Question, contextPayload, "AGENT_SERVICE_UNAVAILABLE", &now)
		if persistErr := s.createAssistantMessage(ctx, input.SessionID, fallback); persistErr != nil {
			return nil, persistErr
		}
		_ = s.touchSession(ctx, sessionRecord.ID)
		_ = userMessage
		return fallback, nil
	}

	result := buildAgentResponsePayload(input.SessionID, clientResponse, fallbackIntent, now)
	if result["answer"] == "" {
		fallback := buildLocalAgentFallback(input.SessionID, input.Question, contextPayload, "EMPTY_AGENT_ANSWER", &now)
		if persistErr := s.createAssistantMessage(ctx, input.SessionID, fallback); persistErr != nil {
			return nil, persistErr
		}
		_ = s.touchSession(ctx, sessionRecord.ID)
		_ = userMessage
		return fallback, nil
	}
	if appErr := s.createAssistantMessage(ctx, input.SessionID, result); appErr != nil {
		return nil, appErr
	}
	if appErr := s.touchSession(ctx, sessionRecord.ID); appErr != nil {
		return nil, appErr
	}
	return result, nil
}

func (s *AgentService) askAgent(ctx context.Context, input AgentAskInput, contextPayload map[string]any) (*agentclient.AskResponse, error) {
	if s.agentClient == nil {
		return nil, errors.New("智能体客户端未初始化")
	}
	return s.agentClient.Ask(ctx, agentclient.AskRequest{
		DatasetID: input.DatasetID,
		SessionID: input.SessionID,
		Question:  input.Question,
		History:   input.History,
		Context:   contextPayload,
	})
}

func (s *AgentService) GenerateReportSummary(
	ctx context.Context,
	dataset *domain.DatasetRecord,
	analysisRecord *domain.AnalysisResultRecord,
) (*ReportSummary, *apperror.AppError) {
	if dataset == nil {
		return nil, apperror.InvalidRequest("dataset 不能为空", nil)
	}
	if analysisRecord == nil {
		return nil, apperror.InvalidRequest("analysisRecord 不能为空", nil)
	}

	runtimeConfig, err := s.systemConfigService.Get(ctx)
	if err != nil {
		return nil, apperror.Internal(err)
	}

	contextPayload, appErr := s.buildAgentContext(ctx, dataset, analysisRecord, runtimeConfig)
	if appErr != nil {
		return nil, appErr
	}

	result, err := s.summarizeReport(ctx, dataset.ID, contextPayload)
	if err != nil {
		s.logWarn("报告摘要智能体调用失败，已回退本地生成", zap.Error(err), zap.Uint64("dataset_id", dataset.ID))
		fallback := buildLocalReportSummary(dataset, contextPayload, "AGENT_SERVICE_UNAVAILABLE")
		return &fallback, nil
	}

	normalized := normalizeReportSummaryResponse(dataset, result)
	if normalized.Overview == "" {
		fallback := buildLocalReportSummary(dataset, contextPayload, "EMPTY_REPORT_SUMMARY")
		return &fallback, nil
	}
	return &normalized, nil
}

func (s *AgentService) summarizeReport(
	ctx context.Context,
	datasetID uint64,
	contextPayload map[string]any,
) (*agentclient.ReportSummaryResponse, error) {
	if s.agentClient == nil {
		return nil, errors.New("智能体客户端未初始化")
	}
	return s.agentClient.SummarizeReport(ctx, agentclient.ReportSummaryRequest{
		DatasetID: datasetID,
		Context:   contextPayload,
	})
}

func (s *AgentService) buildAgentContext(
	ctx context.Context,
	dataset *domain.DatasetRecord,
	analysisRecord *domain.AnalysisResultRecord,
	runtimeConfig *domain.SystemRuntimeConfig,
) (map[string]any, *apperror.AppError) {
	rows, err := readProcessedFeatureRows(*dataset.ProcessedFilePath)
	if err != nil {
		return nil, apperror.Internal(err)
	}

	contextPayload := map[string]any{
		"dataset": map[string]any{
			"id":                  dataset.ID,
			"name":                dataset.Name,
			"description":         nullableString(dataset.Description),
			"household_id":        nullableString(dataset.HouseholdID),
			"row_count":           dataset.RowCount,
			"time_start":          dataset.TimeStart,
			"time_end":            dataset.TimeEnd,
			"processed_file_path": nullableString(dataset.ProcessedFilePath),
		},
		"analysis_summary":       buildAnalysisSummaryContext(analysisRecord),
		"recent_history_summary": buildRecentHistorySummary(rows, runtimeConfig.ModelHistoryWindowConfig.ForecastHistoryDays, runtimeConfig.PeakValleyConfig),
		"prompt_template":        runtimeConfig.EnergyAdvicePromptTemplate,
		"history_days":           runtimeConfig.ModelHistoryWindowConfig.ForecastHistoryDays,
		"classification_days":    runtimeConfig.ModelHistoryWindowConfig.ClassificationDays,
		"peak_valley_config":     runtimeConfig.PeakValleyConfig,
	}

	contextPayload["classification_result"] = s.loadLatestClassification(ctx, dataset.ID)
	contextPayload["forecast_summary"] = s.loadLatestForecastSummary(ctx, dataset.ID)
	contextPayload["rule_advices"] = s.loadRuleAdvices(ctx, dataset.ID)
	return contextPayload, nil
}

func buildAnalysisSummaryContext(record *domain.AnalysisResultRecord) map[string]any {
	if record == nil {
		return map[string]any{}
	}
	return map[string]any{
		"total_kwh":     roundFloat(record.TotalKWH, 4),
		"daily_avg_kwh": roundFloat(record.DailyAvgKWH, 4),
		"max_load_w":    roundFloat(record.MaxLoadW, 2),
		"max_load_time": record.MaxLoadTime,
		"min_load_w":    roundFloat(record.MinLoadW, 2),
		"min_load_time": record.MinLoadTime,
		"peak_kwh":      roundFloat(record.PeakKWH, 4),
		"valley_kwh":    roundFloat(record.ValleyKWH, 4),
		"flat_kwh":      roundFloat(record.FlatKWH, 4),
		"peak_ratio":    roundFloat(record.PeakRatio, 4),
		"valley_ratio":  roundFloat(record.ValleyRatio, 4),
		"flat_ratio":    roundFloat(record.FlatRatio, 4),
	}
}

func buildRecentHistorySummary(rows []processedFeatureRow, historyDays int, peakValleyConfig domain.PeakValleyConfig) map[string]any {
	if len(rows) == 0 {
		return map[string]any{}
	}

	windowSize := historyDays * 96
	if windowSize <= 0 {
		windowSize = 288
	}
	if len(rows) > windowSize {
		rows = rows[len(rows)-windowSize:]
	}

	total := 0.0
	peakTotal := 0.0
	valleyTotal := 0.0
	flatTotal := 0.0
	maxAggregate := 0.0
	activeTotal := 0
	burstTotal := 0
	dayBuckets := make(map[string]float64)

	for _, row := range rows {
		total += row.Aggregate
		activeTotal += row.ActiveApplianceCount
		burstTotal += row.BurstEventCount
		if row.Aggregate > maxAggregate {
			maxAggregate = row.Aggregate
		}
		dayBuckets[row.Timestamp.Format("2006-01-02")] += row.Aggregate

		switch classifyPeakValley(row.Timestamp, peakValleyConfig) {
		case "peak":
			peakTotal += row.Aggregate
		case "valley":
			valleyTotal += row.Aggregate
		default:
			flatTotal += row.Aggregate
		}
	}

	peakRatio := 0.0
	valleyRatio := 0.0
	flatRatio := 0.0
	if total > 1e-6 {
		peakRatio = peakTotal / total
		valleyRatio = valleyTotal / total
		flatRatio = flatTotal / total
	}

	dailyTotals := make([]map[string]any, 0, len(dayBuckets))
	keys := make([]string, 0, len(dayBuckets))
	for key := range dayBuckets {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		dailyTotals = append(dailyTotals, map[string]any{
			"date": key,
			"kwh":  roundFloat(dayBuckets[key], 4),
		})
	}

	dayCount := len(dayBuckets)
	if dayCount <= 0 {
		dayCount = 1
	}

	return map[string]any{
		"window_start":                  rows[0].Timestamp,
		"window_end":                    rows[len(rows)-1].Timestamp,
		"point_count":                   len(rows),
		"total_kwh":                     roundFloat(total, 4),
		"daily_avg_kwh":                 roundFloat(total/float64(dayCount), 4),
		"peak_ratio":                    roundFloat(peakRatio, 4),
		"valley_ratio":                  roundFloat(valleyRatio, 4),
		"flat_ratio":                    roundFloat(flatRatio, 4),
		"max_load_w":                    roundFloat(maxAggregate*4000, 2),
		"avg_active_appliance_count":    roundFloat(float64(activeTotal)/float64(len(rows)), 4),
		"avg_burst_event_count":         roundFloat(float64(burstTotal)/float64(len(rows)), 4),
		"daily_totals":                  dailyTotals,
		"history_days_with_observation": dayCount,
	}
}

func (s *AgentService) loadLatestClassification(ctx context.Context, datasetID uint64) map[string]any {
	if s.classificationRepo == nil {
		return map[string]any{}
	}
	record, err := s.classificationRepo.GetLatest(ctx, datasetID, normalizedClassificationModelType(""))
	if err != nil {
		return map[string]any{}
	}
	var probabilities map[string]float64
	if len(record.Probabilities) > 0 {
		_ = json.Unmarshal(record.Probabilities, &probabilities)
	}
	return map[string]any{
		"id":                 record.ID,
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

func (s *AgentService) loadLatestForecastSummary(ctx context.Context, datasetID uint64) map[string]any {
	if s.forecastRepo == nil {
		return map[string]any{}
	}
	records, _, err := s.forecastRepo.ListByDatasetID(ctx, datasetID, "", 1, 1)
	if err != nil || len(records) == 0 {
		return map[string]any{}
	}

	record := records[0]
	summary := make(map[string]any)
	if len(record.Summary) > 0 {
		_ = json.Unmarshal(record.Summary, &summary)
	}
	if len(summary) == 0 {
		return map[string]any{}
	}
	return normalizeForecastSummaryContext(summary, record)
}

func (s *AgentService) loadRuleAdvices(ctx context.Context, datasetID uint64) []map[string]any {
	if s.adviceRepo == nil {
		return nil
	}
	records, err := s.adviceRepo.ListByDatasetID(ctx, datasetID, "rule")
	if err != nil {
		return nil
	}

	items := make([]map[string]any, 0)
	for _, record := range records {
		contentItems := readAdviceContentItems(record.ContentPath)
		if len(contentItems) == 0 {
			items = append(items, map[string]any{
				"id":         record.ID,
				"summary":    nullableString(record.Summary),
				"created_at": record.CreatedAt,
			})
			continue
		}
		for _, item := range contentItems {
			items = append(items, map[string]any{
				"id":         record.ID,
				"reason":     item.Reason,
				"action":     item.Action,
				"summary":    nullableString(record.Summary),
				"created_at": record.CreatedAt,
			})
		}
	}
	return items
}

func readAdviceContentItems(path string) []domain.AdviceItem {
	if strings.TrimSpace(path) == "" {
		return nil
	}
	content, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var adviceContent domain.AdviceContent
	if err := json.Unmarshal(content, &adviceContent); err != nil {
		return nil
	}
	return adviceContent.Items
}

func validateAgentHistory(history []agentclient.HistoryItem) *apperror.AppError {
	for index, item := range history {
		role := strings.TrimSpace(item.Role)
		content := strings.TrimSpace(item.Content)
		if role != "system" && role != "user" && role != "assistant" {
			return apperror.InvalidRequest("history.role 非法", map[string]any{
				"index": index,
				"role":  item.Role,
			})
		}
		if content == "" {
			return apperror.InvalidRequest("history.content 不能为空", map[string]any{
				"index": index,
			})
		}
	}
	return nil
}

func (s *AgentService) getValidatedSession(ctx context.Context, datasetID, sessionID uint64) (*domain.ChatSessionRecord, *apperror.AppError) {
	record, err := s.sessionRepo.GetByID(ctx, sessionID)
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, apperror.NotFound("CHAT_SESSION_NOT_FOUND", "聊天会话不存在", map[string]any{"session_id": sessionID})
		}
		return nil, apperror.Internal(err)
	}
	if record.DatasetID == nil || *record.DatasetID != datasetID {
		return nil, apperror.Conflict("CHAT_SESSION_DATASET_MISMATCH", "会话与数据集不匹配", map[string]any{
			"session_id": sessionID,
			"dataset_id": datasetID,
		})
	}
	return record, nil
}

func (s *AgentService) loadAskHistory(ctx context.Context, sessionID uint64, requestHistory []agentclient.HistoryItem) ([]agentclient.HistoryItem, *apperror.AppError) {
	records, err := s.messageRepo.ListAllBySessionID(ctx, sessionID)
	if err != nil {
		return nil, apperror.Internal(err)
	}
	if len(records) == 0 {
		return requestHistory, nil
	}

	history := make([]agentclient.HistoryItem, 0, len(records))
	for _, record := range records {
		content := ""
		if record.Content != nil {
			content = strings.TrimSpace(*record.Content)
		}
		if content == "" {
			continue
		}
		history = append(history, agentclient.HistoryItem{
			Role:    record.Role,
			Content: content,
		})
	}
	return history, nil
}

func (s *AgentService) createUserMessage(ctx context.Context, sessionID uint64, question string) (*domain.ChatMessageRecord, *apperror.AppError) {
	content := strings.TrimSpace(question)
	now := time.Now()
	record := &domain.ChatMessageRecord{
		SessionID: sessionID,
		Role:      "user",
		Content:   &content,
		CreatedAt: &now,
	}
	if err := s.messageRepo.Create(ctx, record); err != nil {
		return nil, apperror.Internal(err)
	}
	return record, nil
}

func (s *AgentService) createAssistantMessage(ctx context.Context, sessionID uint64, payload map[string]any) *apperror.AppError {
	answer := strings.TrimSpace(stringValue(payload["answer"]))
	modelName := "models_agent"
	now := time.Now()
	record := &domain.ChatMessageRecord{
		SessionID: sessionID,
		Role:      "assistant",
		Content:   &answer,
		ModelName: &modelName,
		CreatedAt: &now,
	}
	if err := s.messageRepo.Create(ctx, record); err != nil {
		return apperror.Internal(err)
	}

	contentPath, err := writeAssistantMessagePayload(s.cfg.OutputRootDir, sessionID, record.ID, payload)
	if err != nil {
		return apperror.Internal(err)
	}
	record.ContentPath = &contentPath
	if err := s.messageRepo.Update(ctx, record); err != nil {
		return apperror.Internal(err)
	}
	return nil
}

func (s *AgentService) touchSession(ctx context.Context, sessionID uint64) *apperror.AppError {
	if err := s.sessionRepo.Touch(ctx, sessionID, time.Now()); err != nil {
		return apperror.Internal(err)
	}
	return nil
}

func buildLocalAgentFallback(sessionID uint64, question string, contextPayload map[string]any, errorReason string, createdAt *time.Time) map[string]any {
	intent := inferAgentIntent(question)
	missingInformation := buildLocalMissingInformation(contextPayload)
	confidenceLevel := deriveLocalFallbackConfidenceLevel(contextPayload, missingInformation)
	return map[string]any{
		"session_id":          sessionID,
		"answer":              buildLocalFallbackAnswer(strings.TrimSpace(question), contextPayload),
		"citations":           buildLocalFallbackCitations(contextPayload),
		"actions":             buildLocalFallbackActions(contextPayload),
		"missing_information": missingInformation,
		"confidence_level":    confidenceLevel,
		"intent":              intent,
		"degraded":            true,
		"error_reason":        errorReason,
		"created_at":          createdAt,
	}
}

func buildLocalFallbackAnswer(question string, contextPayload map[string]any) string {
	sentences := []string{"智能问答暂时不可用，以下为基于现有分析结果的建议。"}

	analysisSummary, _ := contextPayload["analysis_summary"].(map[string]any)
	classificationResult, _ := contextPayload["classification_result"].(map[string]any)
	forecastSummary, _ := contextPayload["forecast_summary"].(map[string]any)

	if peakRatio, ok := floatValue(analysisSummary["peak_ratio"]); ok {
		sentences = append(sentences, fmt.Sprintf("峰时占比约为 %.2f。", peakRatio))
	}

	if predictedLabel := stringValue(classificationResult["predicted_label"]); predictedLabel != "" {
		sentences = append(sentences, fmt.Sprintf("当前行为类型判断为%s。", classificationLabelText(predictedLabel)))
	}

	if peakPeriod := stringValue(forecastSummary["peak_period"]); peakPeriod != "" {
		sentences = append(sentences, fmt.Sprintf("最近一次预测显示高负荷时段集中在 %s。", peakPeriod))
	}

	if strings.Contains(question, "夜间") && stringValue(classificationResult["predicted_label"]) == "day_low_night_high" {
		sentences = append(sentences, "建议优先排查夜间持续运行设备，并检查热水器、取暖和充电类负载。")
	} else if strings.Contains(question, "明天") && stringValue(forecastSummary["peak_period"]) != "" {
		sentences = append(sentences, "建议提前避开预测高负荷时段安排洗衣、热水或充电任务。")
	} else {
		sentences = append(sentences, "建议优先处理峰时段高负荷设备，并结合规则建议逐项执行。")
	}

	return strings.Join(sentences, "")
}

func buildLocalFallbackCitations(contextPayload map[string]any) []agentclient.CitationItem {
	citations := make([]agentclient.CitationItem, 0, 6)
	analysisSummary, _ := contextPayload["analysis_summary"].(map[string]any)
	classificationResult, _ := contextPayload["classification_result"].(map[string]any)
	forecastSummary, _ := contextPayload["forecast_summary"].(map[string]any)

	if value, exists := analysisSummary["peak_ratio"]; exists {
		citations = append(citations, agentclient.CitationItem{Key: "peak_ratio", Label: "峰时占比", Value: value})
	}
	if value, exists := analysisSummary["daily_avg_kwh"]; exists {
		citations = append(citations, agentclient.CitationItem{Key: "daily_avg_kwh", Label: "日均用电量", Value: value})
	}
	if value := stringValue(classificationResult["predicted_label"]); value != "" {
		citations = append(citations, agentclient.CitationItem{Key: "predicted_label", Label: "行为类型", Value: value})
	}
	if value, exists := classificationResult["confidence"]; exists {
		citations = append(citations, agentclient.CitationItem{Key: "confidence", Label: "分类置信度", Value: value})
	}
	if value := stringValue(forecastSummary["peak_period"]); value != "" {
		citations = append(citations, agentclient.CitationItem{Key: "forecast_peak_period", Label: "预测高负荷时段", Value: value})
	}
	if value, exists := forecastSummary["risk_flags"]; exists {
		citations = append(citations, agentclient.CitationItem{Key: "risk_flags", Label: "预测风险标签", Value: value})
	}
	return citations
}

func buildLocalFallbackActions(contextPayload map[string]any) []string {
	actions := make([]string, 0, 5)

	if rawItems, ok := contextPayload["rule_advices"].([]map[string]any); ok {
		for _, item := range rawItems {
			action := stringValue(item["action"])
			if action == "" {
				action = stringValue(item["summary"])
			}
			if action != "" && !containsString(actions, action) {
				actions = append(actions, action)
			}
		}
	}

	if len(actions) == 0 {
		classificationResult, _ := contextPayload["classification_result"].(map[string]any)
		switch stringValue(classificationResult["predicted_label"]) {
		case "day_low_night_high":
			actions = append(actions, "优先检查晚间持续运行设备", "将热水器改为定时运行")
		case "day_high_night_low":
			actions = append(actions, "将白天高耗电任务错峰执行", "重点排查白天集中启停设备")
		default:
			actions = append(actions, "复查峰时段高耗电设备", "优先调整可延后负荷到低负荷时段")
		}
	}

	if len(actions) > 5 {
		return actions[:5]
	}
	return actions
}

func buildAgentResponsePayload(sessionID uint64, response *agentclient.AskResponse, fallbackIntent string, createdAt time.Time) map[string]any {
	citations := response.Citations
	if citations == nil {
		citations = []agentclient.CitationItem{}
	}
	actions := response.Actions
	if actions == nil {
		actions = []string{}
	}
	missingInformation := response.MissingInformation
	if missingInformation == nil {
		missingInformation = []agentclient.MissingInformationItem{}
	}

	intent := strings.TrimSpace(response.Intent)
	if intent == "" {
		intent = fallbackIntent
	}

	confidenceLevel := normalizeConfidenceLevelPointer(response.ConfidenceLevel)
	if confidenceLevel == nil {
		derived := deriveAgentConfidenceLevel(response.Degraded, len(citations), len(missingInformation))
		confidenceLevel = &derived
	}

	return map[string]any{
		"session_id":          sessionID,
		"answer":              strings.TrimSpace(response.Answer),
		"citations":           citations,
		"actions":             actions,
		"missing_information": missingInformation,
		"confidence_level":    *confidenceLevel,
		"intent":              intent,
		"degraded":            response.Degraded,
		"error_reason":        response.ErrorReason,
		"created_at":          createdAt,
	}
}

func normalizeForecastSummaryContext(summary map[string]any, record domain.ForecastResultRecord) map[string]any {
	normalized := map[string]any{
		"id":               record.ID,
		"schema_version":   "v1",
		"model_type":       normalizeForecastModelType(stringValue(summary["model_type"]), record.ModelType),
		"forecast_horizon": "1d",
		"forecast_start":   record.ForecastStart,
		"forecast_end":     record.ForecastEnd,
		"granularity":      record.Granularity,
		"created_at":       record.CreatedAt,
		"risk_flags":       normalizeForecastRiskFlags(summary["risk_flags"]),
	}

	if value, ok := floatValue(summary["predicted_avg_load_w"]); ok {
		normalized["predicted_avg_load_w"] = roundFloat(value, 2)
	}
	if value, ok := floatValue(summary["predicted_peak_load_w"]); ok {
		normalized["predicted_peak_load_w"] = roundFloat(value, 2)
	}
	if value, ok := floatValue(summary["predicted_total_kwh"]); ok {
		normalized["predicted_total_kwh"] = roundFloat(value, 4)
	}

	peakPeriod := stringValue(summary["peak_period"])
	if peakPeriod == "" {
		periods := stringListValue(summary["forecast_peak_periods"])
		if len(periods) > 0 {
			peakPeriod = periods[0]
		}
	}
	if peakPeriod != "" {
		normalized["peak_period"] = peakPeriod
	}

	if confidenceHint := normalizeConfidenceLevelValue(stringValue(summary["confidence_hint"])); confidenceHint != "" {
		normalized["confidence_hint"] = confidenceHint
	}

	return normalized
}

func normalizeForecastModelType(primary string, fallback string) string {
	for _, candidate := range []string{primary, fallback} {
		if normalized := canonicalForecastModelType(candidate); normalized != "" {
			return normalized
		}
	}
	return "tft"
}

func normalizeForecastRiskFlags(value any) []string {
	mapping := map[string]string{
		"evening_peak_risk":  "evening_peak",
		"night_load_risk":    "high_baseload",
		"peak_usage_risk":    "daytime_peak",
		"morning_spike_risk": "abnormal_rise",
	}

	normalized := make([]string, 0)
	for _, item := range stringListValue(value) {
		flag := strings.ToLower(strings.TrimSpace(item))
		if mapped, exists := mapping[flag]; exists {
			flag = mapped
		}
		switch flag {
		case "evening_peak", "daytime_peak", "high_baseload", "abnormal_rise", "peak_overlap_risk":
			if !containsString(normalized, flag) {
				normalized = append(normalized, flag)
			}
		}
	}
	return normalized
}

func normalizeConfidenceLevelPointer(value *string) *string {
	if value == nil {
		return nil
	}
	normalized := normalizeConfidenceLevelValue(*value)
	if normalized == "" {
		return nil
	}
	return &normalized
}

func normalizeConfidenceLevelValue(value string) string {
	normalized := strings.ToLower(strings.TrimSpace(value))
	switch normalized {
	case "high", "medium", "low":
		return normalized
	default:
		return ""
	}
}

func deriveAgentConfidenceLevel(degraded bool, citationCount int, missingCount int) string {
	if degraded {
		if missingCount == 0 && citationCount >= 3 {
			return "medium"
		}
		return "low"
	}
	if missingCount == 0 && citationCount >= 2 {
		return "high"
	}
	if citationCount > 0 {
		return "medium"
	}
	return "low"
}

func stringListValue(value any) []string {
	switch typed := value.(type) {
	case []string:
		items := make([]string, 0, len(typed))
		for _, item := range typed {
			normalized := stringValue(item)
			if normalized != "" {
				items = append(items, normalized)
			}
		}
		return items
	case []any:
		items := make([]string, 0, len(typed))
		for _, item := range typed {
			normalized := stringValue(item)
			if normalized != "" {
				items = append(items, normalized)
			}
		}
		return items
	default:
		return nil
	}
}

func buildLocalMissingInformation(contextPayload map[string]any) []agentclient.MissingInformationItem {
	items := make([]agentclient.MissingInformationItem, 0, 3)

	classificationResult, _ := contextPayload["classification_result"].(map[string]any)
	if stringValue(classificationResult["predicted_label"]) == "" {
		items = append(items, agentclient.MissingInformationItem{
			Key:      "classification_result",
			Question: "请先补充最近一天的分类结果，或重新运行日用电分类分析。",
			Reason:   "当前缺少用户日用电行为分类，无法稳定判断属于哪一类用电模式。",
		})
	}

	forecastSummary, _ := contextPayload["forecast_summary"].(map[string]any)
	if stringValue(forecastSummary["peak_period"]) == "" {
		items = append(items, agentclient.MissingInformationItem{
			Key:      "forecast_summary",
			Question: "请先补充下一天负荷预测结果，尤其是峰值时段与总电量摘要。",
			Reason:   "当前缺少未来一天的预测摘要，无法给出更具体的错峰建议。",
		})
	}

	analysisSummary, _ := contextPayload["analysis_summary"].(map[string]any)
	if _, ok := floatValue(analysisSummary["daily_avg_kwh"]); !ok {
		items = append(items, agentclient.MissingInformationItem{
			Key:      "analysis_summary",
			Question: "请先补充统计分析摘要，例如日均用电量与峰时占比。",
			Reason:   "基础统计摘要不足，回答只能依赖有限上下文，解释力度会明显下降。",
		})
	}

	return items
}

func deriveLocalFallbackConfidenceLevel(contextPayload map[string]any, missingInformation []agentclient.MissingInformationItem) string {
	if len(missingInformation) > 0 {
		return "low"
	}

	classificationResult, _ := contextPayload["classification_result"].(map[string]any)
	forecastSummary, _ := contextPayload["forecast_summary"].(map[string]any)
	if stringValue(classificationResult["predicted_label"]) != "" && stringValue(forecastSummary["peak_period"]) != "" {
		return "medium"
	}
	return "low"
}

func inferAgentIntent(question string) string {
	normalized := strings.TrimSpace(question)
	switch {
	case strings.Contains(normalized, "分类"), strings.Contains(normalized, "类型"), strings.Contains(normalized, "模式"):
		return "classification"
	case strings.Contains(normalized, "预测"), strings.Contains(normalized, "明天"), strings.Contains(normalized, "未来"):
		return "forecast"
	case strings.Contains(normalized, "风险"), strings.Contains(normalized, "异常"), strings.Contains(normalized, "峰值"):
		return "risk"
	case strings.Contains(normalized, "概况"), strings.Contains(normalized, "整体"), strings.Contains(normalized, "总览"):
		return "overview"
	case strings.Contains(normalized, "建议"), strings.Contains(normalized, "怎么"), strings.Contains(normalized, "如何"), strings.Contains(normalized, "应该"):
		return "advice"
	default:
		return "follow_up"
	}
}

func buildLocalReportSummary(
	dataset *domain.DatasetRecord,
	contextPayload map[string]any,
	reason string,
) ReportSummary {
	analysisSummary, _ := contextPayload["analysis_summary"].(map[string]any)
	classificationResult, _ := contextPayload["classification_result"].(map[string]any)
	forecastSummary, _ := contextPayload["forecast_summary"].(map[string]any)
	historySummary, _ := contextPayload["recent_history_summary"].(map[string]any)

	title := "居民用电分析报告"
	if name := strings.TrimSpace(dataset.Name); name != "" {
		title = title + " - " + name
	}

	overviewParts := make([]string, 0, 4)
	if value, ok := floatValue(analysisSummary["daily_avg_kwh"]); ok {
		overviewParts = append(overviewParts, fmt.Sprintf("日均用电量约 %.2f kWh", value))
	}
	if value, ok := floatValue(analysisSummary["peak_ratio"]); ok {
		overviewParts = append(overviewParts, fmt.Sprintf("峰时占比 %.2f%%", value*100))
	}
	if label := classificationLabelText(stringValue(classificationResult["predicted_label"])); label != "" {
		overviewParts = append(overviewParts, "行为类型判断为"+label)
	}
	if peakPeriod := stringValue(forecastSummary["peak_period"]); peakPeriod != "" {
		overviewParts = append(overviewParts, "预测高负荷时段集中在"+peakPeriod)
	}
	if len(overviewParts) == 0 {
		overviewParts = append(overviewParts, "当前智能体已降级，报告摘要基于已有分析结果自动整理。")
	}
	overviewText := strings.Join(overviewParts, "，") + "。"

	behaviorParts := make([]string, 0, 4)
	if label := classificationLabelText(stringValue(classificationResult["predicted_label"])); label != "" {
		behaviorParts = append(behaviorParts, "当前家庭用电行为被识别为"+label)
	}
	if value, ok := floatValue(classificationResult["confidence"]); ok {
		behaviorParts = append(behaviorParts, fmt.Sprintf("分类置信度约 %.2f%%", value*100))
	}
	if value, ok := floatValue(historySummary["avg_active_appliance_count"]); ok {
		behaviorParts = append(behaviorParts, fmt.Sprintf("最近窗口内平均活跃电器数量约 %.2f 个", value))
	}
	if value, ok := floatValue(historySummary["avg_burst_event_count"]); ok {
		behaviorParts = append(behaviorParts, fmt.Sprintf("平均突发事件数约 %.2f", value))
	}
	behaviorText := "当前缺少明确的行为分类结果，建议结合历史曲线进一步确认用电模式。"
	if len(behaviorParts) > 0 {
		behaviorText = strings.Join(behaviorParts, "，") + "。"
	}

	riskParts := make([]string, 0, 5)
	if peakPeriod := stringValue(forecastSummary["peak_period"]); peakPeriod != "" {
		riskParts = append(riskParts, "预测高负荷时段集中在"+peakPeriod)
	}
	if value, ok := floatValue(forecastSummary["predicted_avg_load_w"]); ok {
		riskParts = append(riskParts, fmt.Sprintf("预测平均负荷约 %.2f W", value))
	}
	if value, ok := floatValue(forecastSummary["predicted_peak_load_w"]); ok {
		riskParts = append(riskParts, fmt.Sprintf("预测峰值负荷约 %.2f W", value))
	}
	if riskFlags := stringValue(forecastSummary["risk_flags"]); riskFlags != "" {
		riskParts = append(riskParts, "风险标签："+riskFlags)
	}
	if value, ok := floatValue(historySummary["max_load_w"]); ok {
		riskParts = append(riskParts, fmt.Sprintf("历史窗口内最高负荷约 %.2f W", value))
	}
	riskText := "当前缺少预测结果，暂无法给出稳定的风险判断。"
	if len(riskParts) > 0 {
		riskText = strings.Join(riskParts, "，") + "。"
	}

	recommendations := buildLocalFallbackActions(contextPayload)
	if len(recommendations) == 0 {
		recommendations = []string{"建议优先复核峰时段负荷和夜间持续运行设备。"}
	}
	noteParts := []string{
		"本报告依据统计分析、行为分类、负荷预测与规则建议自动生成",
		"适合作为阶段性分析与归档材料",
	}
	if degradedNote := reportSummaryDegradedNote(reason); degradedNote != "" {
		noteParts = append(noteParts, degradedNote)
	}
	noteText := strings.Join(noteParts, "，") + "。"

	return ReportSummary{
		Title:    title,
		Overview: overviewText,
		Sections: orderedReportSections(map[string]string{
			"总体概览": overviewText,
			"行为判断": behaviorText,
			"预测风险": riskText,
			"附注":   noteText,
		}),
		Recommendations: recommendations,
		Degraded:        true,
		ErrorReason:     stringPtr(reason),
	}
}

func normalizeReportSummaryResponse(dataset *domain.DatasetRecord, response *agentclient.ReportSummaryResponse) ReportSummary {
	if response == nil {
		return buildLocalReportSummary(dataset, map[string]any{}, "EMPTY_REPORT_SUMMARY")
	}

	title := strings.TrimSpace(response.Title)
	if title == "" {
		title = "居民用电分析报告"
		if dataset != nil && strings.TrimSpace(dataset.Name) != "" {
			title = title + " - " + strings.TrimSpace(dataset.Name)
		}
	}

	fallback := buildLocalReportSummary(dataset, map[string]any{}, "EMPTY_REPORT_SUMMARY")
	overview := strings.TrimSpace(response.Overview)
	if overview == "" {
		overview = fallback.Overview
	}

	sectionValues := make(map[string]string, len(fallback.Sections))
	for _, section := range fallback.Sections {
		sectionValues[section.Title] = section.Body
	}
	for _, section := range response.Sections {
		titleValue := strings.TrimSpace(section.Title)
		bodyValue := strings.TrimSpace(section.Body)
		if titleValue == "" || bodyValue == "" {
			continue
		}
		switch titleValue {
		case "总体概览", "行为判断", "预测风险", "附注":
			sectionValues[titleValue] = bodyValue
		}
	}
	sectionValues["总体概览"] = overview
	if response.Degraded {
		sectionValues["附注"] = reportSummaryDegradedSection(response.ErrorReason)
	}
	sections := orderedReportSections(sectionValues)

	recommendations := make([]string, 0, max(len(response.Recommendations), len(fallback.Recommendations)))
	for _, item := range response.Recommendations {
		value := strings.TrimSpace(item)
		if value != "" && !containsString(recommendations, value) {
			recommendations = append(recommendations, value)
		}
	}
	if len(recommendations) == 0 {
		recommendations = append(recommendations, fallback.Recommendations...)
	}

	return ReportSummary{
		Title:           title,
		Overview:        overview,
		Sections:        sections,
		Recommendations: recommendations,
		Degraded:        response.Degraded,
		ErrorReason:     response.ErrorReason,
	}
}

func reportSummaryDegradedNote(reason string) string {
	if strings.TrimSpace(reason) == "" {
		return ""
	}
	return "摘要由系统基于现有分析结果自动整理"
}

func reportSummaryDegradedSection(reason *string) string {
	if reason == nil || strings.TrimSpace(*reason) == "" {
		return "本报告摘要由系统基于现有分析结果自动整理，可正常用于导出与归档。"
	}
	return "本报告摘要由系统基于现有分析结果自动整理，当前未启用增强总结能力，可正常用于导出与归档。"
}

func orderedReportSections(values map[string]string) []agentclient.ReportSection {
	order := []string{"总体概览", "行为判断", "预测风险", "附注"}
	sections := make([]agentclient.ReportSection, 0, len(order))
	for _, title := range order {
		body := strings.TrimSpace(values[title])
		if body == "" {
			continue
		}
		sections = append(sections, agentclient.ReportSection{
			Title: title,
			Body:  body,
		})
	}
	return sections
}

func classificationLabelText(label string) string {
	switch label {
	case "day_high_night_low":
		return "白天高晚上低型"
	case "day_low_night_high":
		return "白天低晚上高型"
	case "all_day_high":
		return "全天高负载型"
	case "all_day_low":
		return "全天低负载型"
	default:
		return label
	}
}

func floatValue(value any) (float64, bool) {
	switch typed := value.(type) {
	case float64:
		return typed, true
	case float32:
		return float64(typed), true
	case int:
		return float64(typed), true
	case int64:
		return float64(typed), true
	case uint64:
		return float64(typed), true
	case json.Number:
		parsed, err := typed.Float64()
		return parsed, err == nil
	default:
		return 0, false
	}
}

func stringValue(value any) string {
	if value == nil {
		return ""
	}
	switch typed := value.(type) {
	case string:
		return strings.TrimSpace(typed)
	default:
		return strings.TrimSpace(fmt.Sprintf("%v", value))
	}
}

func containsString(items []string, target string) bool {
	for _, item := range items {
		if item == target {
			return true
		}
	}
	return false
}

func (s *AgentService) logWarn(message string, fields ...zap.Field) {
	if s.logger != nil {
		s.logger.Warn(message, fields...)
	}
}
