package service

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"residential-energy-intelligence-agent-platform/internal/config"
	"residential-energy-intelligence-agent-platform/internal/domain"
	"residential-energy-intelligence-agent-platform/internal/integration/agentclient"
)

func TestAgentServiceAskFallsBackWhenAgentClientFails(t *testing.T) {
	tempDir := t.TempDir()
	processedPath := filepath.Join(tempDir, "processed.csv")
	advicePath := filepath.Join(tempDir, "advice.json")
	if err := os.WriteFile(processedPath, []byte(buildAgentTestCSV()), 0o644); err != nil {
		t.Fatalf("写入 processed.csv 失败: %v", err)
	}
	if err := os.WriteFile(advicePath, []byte(`{"items":[{"reason":"峰时占比偏高","action":"将热水器改为定时运行"}]}`), 0o644); err != nil {
		t.Fatalf("写入 advice.json 失败: %v", err)
	}

	now := time.Date(2026, 4, 2, 23, 45, 0, 0, time.Local)
	probabilities, _ := json.Marshal(map[string]float64{"day_low_night_high": 0.91})
	forecastSummary, _ := json.Marshal(map[string]any{
		"forecast_peak_periods": []string{"2026-04-03T19:00:00+08:00/2026-04-03T22:00:00+08:00"},
		"risk_flags":            []string{"evening_peak_risk", "night_load_risk"},
	})
	messageRepo := newAgentChatMessageRepo()

	service := NewAgentService(
		&config.AppConfig{OutputRootDir: tempDir},
		agentDatasetRepo{
			record: &domain.DatasetRecord{
				ID:                1,
				Name:              "house_1",
				RawFilePath:       "raw.csv",
				ProcessedFilePath: stringPtr(processedPath),
				Status:            "ready",
				RowCount:          288,
				TimeEnd:           &now,
			},
		},
		agentChatSessionRepo{
			record: &domain.ChatSessionRecord{
				ID:        3,
				DatasetID: uint64Ptr(1),
				Title:     stringPtr("House 1 节能建议问答"),
				CreatedAt: &now,
				UpdatedAt: &now,
			},
		},
		messageRepo,
		agentAnalysisRepo{
			record: &domain.AnalysisResultRecord{
				DatasetID:   1,
				DailyAvgKWH: 12.34,
				PeakRatio:   0.46,
			},
		},
		agentClassificationRepo{
			record: &domain.ClassificationResultRecord{
				ID:             9,
				DatasetID:      1,
				ModelType:      "tcn",
				PredictedLabel: "day_low_night_high",
				Confidence:     0.91,
				Probabilities:  probabilities,
			},
		},
		agentForecastRepo{
			records: []domain.ForecastResultRecord{
				{
					ID:          11,
					DatasetID:   1,
					ModelType:   "lstm",
					Granularity: "15min",
					Summary:     forecastSummary,
					CreatedAt:   &now,
				},
			},
		},
		agentAdviceRepo{
			records: []domain.EnergyAdviceRecord{
				{
					ID:          5,
					DatasetID:   1,
					AdviceType:  "rule",
					ContentPath: advicePath,
					Summary:     stringPtr("将热水器改为定时运行"),
					CreatedAt:   &now,
				},
			},
		},
		&SystemConfigService{},
		agentTestClient{err: errors.New("dial tcp timeout")},
		nil,
	)

	result, appErr := service.Ask(context.Background(), AgentAskInput{
		DatasetID: 1,
		SessionID: 3,
		Question:  "为什么我家夜间用电这么高？",
		History: []agentclient.HistoryItem{
			{Role: "user", Content: "先看整体情况"},
		},
	})
	if appErr != nil {
		t.Fatalf("Ask() 返回错误: %v", appErr)
	}
	if degraded, ok := result["degraded"].(bool); !ok || !degraded {
		t.Fatalf("degraded = %v, want true", result["degraded"])
	}
	if reason, ok := result["error_reason"].(string); !ok || reason != "AGENT_SERVICE_UNAVAILABLE" {
		t.Fatalf("error_reason = %v, want AGENT_SERVICE_UNAVAILABLE", result["error_reason"])
	}
	if answer, _ := result["answer"].(string); answer == "" {
		t.Fatal("answer 为空")
	}
	actions, ok := result["actions"].([]string)
	if !ok || len(actions) == 0 {
		t.Fatalf("actions = %#v, want non-empty []string", result["actions"])
	}
	if got := len(messageRepo.records); got != 2 {
		t.Fatalf("消息条数 = %d, want 2", got)
	}
	if messageRepo.records[0].Role != "user" {
		t.Fatalf("第一条消息 role = %s, want user", messageRepo.records[0].Role)
	}
	if messageRepo.records[1].Role != "assistant" {
		t.Fatalf("第二条消息 role = %s, want assistant", messageRepo.records[1].Role)
	}
	if messageRepo.records[1].ContentPath == nil || *messageRepo.records[1].ContentPath == "" {
		t.Fatal("assistant content_path 未写入")
	}
}

func TestAgentServiceAskUsesStoredHistoryAndPersistsMessages(t *testing.T) {
	tempDir := t.TempDir()
	processedPath := filepath.Join(tempDir, "processed.csv")
	if err := os.WriteFile(processedPath, []byte(buildAgentTestCSV()), 0o644); err != nil {
		t.Fatalf("写入 processed.csv 失败: %v", err)
	}

	now := time.Date(2026, 4, 2, 23, 45, 0, 0, time.Local)
	messageRepo := newAgentChatMessageRepo()
	oldQuestion := "前一天整体负荷怎么样？"
	oldAnswer := "峰时负荷偏高。"
	messageRepo.records = append(messageRepo.records,
		domain.ChatMessageRecord{ID: 1, SessionID: 3, Role: "user", Content: &oldQuestion, CreatedAt: &now},
		domain.ChatMessageRecord{ID: 2, SessionID: 3, Role: "assistant", Content: &oldAnswer, CreatedAt: &now},
	)
	messageRepo.nextID = 3

	client := &recordingAgentClient{
		response: &agentclient.AskResponse{
			Answer:    "建议优先检查夜间设备。",
			Citations: []agentclient.CitationItem{{Key: "predicted_label", Label: "行为类型", Value: "day_low_night_high"}},
			Actions:   []string{"检查夜间持续运行设备"},
			Degraded:  false,
		},
	}

	service := NewAgentService(
		&config.AppConfig{OutputRootDir: tempDir},
		agentDatasetRepo{
			record: &domain.DatasetRecord{
				ID:                1,
				Name:              "house_1",
				RawFilePath:       "raw.csv",
				ProcessedFilePath: stringPtr(processedPath),
				Status:            "ready",
				RowCount:          288,
				TimeEnd:           &now,
			},
		},
		agentChatSessionRepo{
			record: &domain.ChatSessionRecord{
				ID:        3,
				DatasetID: uint64Ptr(1),
				Title:     stringPtr("House 1 节能建议问答"),
				CreatedAt: &now,
				UpdatedAt: &now,
			},
		},
		messageRepo,
		agentAnalysisRepo{
			record: &domain.AnalysisResultRecord{
				DatasetID:   1,
				DailyAvgKWH: 9.88,
				PeakRatio:   0.41,
			},
		},
		agentClassificationRepo{
			record: &domain.ClassificationResultRecord{
				ID:             9,
				DatasetID:      1,
				ModelType:      "tcn",
				PredictedLabel: "day_low_night_high",
				Confidence:     0.91,
			},
		},
		agentForecastRepo{},
		agentAdviceRepo{},
		&SystemConfigService{},
		client,
		nil,
	)

	result, appErr := service.Ask(context.Background(), AgentAskInput{
		DatasetID: 1,
		SessionID: 3,
		Question:  "今晚应该注意什么？",
		History: []agentclient.HistoryItem{
			{Role: "user", Content: "这个 history 不应该覆盖数据库历史"},
		},
	})
	if appErr != nil {
		t.Fatalf("Ask() 返回错误: %v", appErr)
	}
	if client.request == nil {
		t.Fatal("agent client 未收到请求")
	}
	if got := len(client.request.History); got != 2 {
		t.Fatalf("history len = %d, want 2", got)
	}
	if client.request.History[0].Content != oldQuestion {
		t.Fatalf("history[0] = %s, want %s", client.request.History[0].Content, oldQuestion)
	}
	if answer, _ := result["answer"].(string); answer == "" {
		t.Fatal("answer 为空")
	}
	if got := len(messageRepo.records); got != 4 {
		t.Fatalf("消息条数 = %d, want 4", got)
	}
	lastMessage := messageRepo.records[len(messageRepo.records)-1]
	if lastMessage.Role != "assistant" {
		t.Fatalf("最后一条消息 role = %s, want assistant", lastMessage.Role)
	}
	if lastMessage.ContentPath == nil || *lastMessage.ContentPath == "" {
		t.Fatal("assistant content_path 未写入")
	}
	if _, err := os.Stat(*lastMessage.ContentPath); err != nil {
		t.Fatalf("assistant payload 文件不存在: %v", err)
	}
}

func TestBuildRecentHistorySummaryUsesLatestWindow(t *testing.T) {
	rows := make([]processedFeatureRow, 0, 4*96)
	start := time.Date(2026, 4, 1, 0, 0, 0, 0, time.Local)
	for day := 0; day < 4; day++ {
		for slot := 0; slot < 96; slot++ {
			rows = append(rows, processedFeatureRow{
				Timestamp:            start.AddDate(0, 0, day).Add(time.Duration(slot) * 15 * time.Minute),
				Aggregate:            1 + float64(day),
				ActiveApplianceCount: 2,
				BurstEventCount:      1,
			})
		}
	}

	summary := buildRecentHistorySummary(rows, 3, domain.PeakValleyConfig{
		Peak:   []string{"07:00-11:00", "18:00-23:00"},
		Valley: []string{"23:00-07:00"},
	})

	if got, want := summary["point_count"], 288; got != want {
		t.Fatalf("point_count = %v, want %d", got, want)
	}
	startTime, ok := summary["window_start"].(time.Time)
	if !ok {
		t.Fatalf("window_start 类型错误: %#v", summary["window_start"])
	}
	if got := startTime.Format("2006-01-02"); got != "2026-04-02" {
		t.Fatalf("window_start = %s, want 2026-04-02", got)
	}
}

func TestNormalizeReportSummaryResponseHidesTechnicalReasonInNote(t *testing.T) {
	reason := "LLM_BASE_URL_MISSING"
	summary := normalizeReportSummaryResponse(
		&domain.DatasetRecord{Name: "house_1"},
		&agentclient.ReportSummaryResponse{
			Title:    "居民用电分析报告 - house_1",
			Overview: "系统已根据现有分析结果整理报告摘要。",
			Sections: []agentclient.ReportSection{
				{Title: "总体概览", Body: "系统已根据现有分析结果整理报告摘要。"},
			},
			Recommendations: []string{"优先检查峰时段负荷安排"},
			Degraded:        true,
			ErrorReason:     &reason,
		},
	)

	var note string
	for _, section := range summary.Sections {
		if section.Title == "附注" {
			note = section.Body
			break
		}
	}

	if note == "" {
		t.Fatal("附注为空")
	}
	if strings.Contains(note, reason) {
		t.Fatalf("附注暴露了技术原因码: %s", note)
	}
	if !strings.Contains(note, "未启用增强总结能力") {
		t.Fatalf("附注 = %s, want 用户可读说明", note)
	}
}

type agentTestClient struct {
	response       *agentclient.AskResponse
	reportResponse *agentclient.ReportSummaryResponse
	err            error
}

func (c agentTestClient) Health(context.Context) error {
	return nil
}

func (c agentTestClient) Ask(context.Context, agentclient.AskRequest) (*agentclient.AskResponse, error) {
	if c.err != nil {
		return nil, c.err
	}
	return c.response, nil
}

func (c agentTestClient) SummarizeReport(context.Context, agentclient.ReportSummaryRequest) (*agentclient.ReportSummaryResponse, error) {
	if c.err != nil {
		return nil, c.err
	}
	return c.reportResponse, nil
}

func (c agentTestClient) RenderPDF(context.Context, agentclient.RenderPDFRequest) ([]byte, error) {
	if c.err != nil {
		return nil, c.err
	}
	return []byte("%PDF-1.4"), nil
}

type recordingAgentClient struct {
	request       *agentclient.AskRequest
	reportRequest *agentclient.ReportSummaryRequest
	response      *agentclient.AskResponse
	reportResp    *agentclient.ReportSummaryResponse
}

func (c *recordingAgentClient) Health(context.Context) error {
	return nil
}

func (c *recordingAgentClient) Ask(_ context.Context, request agentclient.AskRequest) (*agentclient.AskResponse, error) {
	requestCopy := request
	c.request = &requestCopy
	return c.response, nil
}

func (c *recordingAgentClient) SummarizeReport(_ context.Context, request agentclient.ReportSummaryRequest) (*agentclient.ReportSummaryResponse, error) {
	requestCopy := request
	c.reportRequest = &requestCopy
	return c.reportResp, nil
}

func (c *recordingAgentClient) RenderPDF(_ context.Context, _ agentclient.RenderPDFRequest) ([]byte, error) {
	return []byte("%PDF-1.4"), nil
}

type agentDatasetRepo struct {
	record *domain.DatasetRecord
}

func (r agentDatasetRepo) Create(context.Context, *domain.DatasetRecord) error { return nil }
func (r agentDatasetRepo) Update(context.Context, *domain.DatasetRecord) error { return nil }
func (r agentDatasetRepo) GetByID(context.Context, uint64) (*domain.DatasetRecord, error) {
	if r.record == nil {
		return nil, errors.New("not found")
	}
	return r.record, nil
}
func (r agentDatasetRepo) List(context.Context, domain.DatasetListFilter) ([]domain.DatasetRecord, int64, error) {
	return nil, 0, nil
}
func (r agentDatasetRepo) Delete(context.Context, uint64) error { return nil }

type agentChatSessionRepo struct {
	record *domain.ChatSessionRecord
}

func (r agentChatSessionRepo) Create(context.Context, *domain.ChatSessionRecord) error { return nil }
func (r agentChatSessionRepo) GetByID(context.Context, uint64) (*domain.ChatSessionRecord, error) {
	if r.record == nil {
		return nil, errors.New("not found")
	}
	return r.record, nil
}
func (r agentChatSessionRepo) List(context.Context, *uint64, int, int) ([]domain.ChatSessionRecord, int64, error) {
	return nil, 0, nil
}
func (r agentChatSessionRepo) Touch(context.Context, uint64, time.Time) error { return nil }
func (r agentChatSessionRepo) Delete(context.Context, uint64) error           { return nil }

type agentChatMessageRepo struct {
	records []domain.ChatMessageRecord
	nextID  uint64
}

func newAgentChatMessageRepo() *agentChatMessageRepo {
	return &agentChatMessageRepo{nextID: 1}
}

func (r *agentChatMessageRepo) Create(_ context.Context, record *domain.ChatMessageRecord) error {
	if record.ID == 0 {
		record.ID = r.nextID
		r.nextID++
	}
	copyRecord := *record
	r.records = append(r.records, copyRecord)
	return nil
}

func (r *agentChatMessageRepo) Update(_ context.Context, record *domain.ChatMessageRecord) error {
	for index := range r.records {
		if r.records[index].ID == record.ID {
			r.records[index] = *record
			return nil
		}
	}
	return errors.New("not found")
}

func (r *agentChatMessageRepo) ListBySessionID(_ context.Context, sessionID uint64, page, pageSize int) ([]domain.ChatMessageRecord, int64, error) {
	items := make([]domain.ChatMessageRecord, 0)
	for _, record := range r.records {
		if record.SessionID == sessionID {
			items = append(items, record)
		}
	}
	return items, int64(len(items)), nil
}

func (r *agentChatMessageRepo) ListAllBySessionID(_ context.Context, sessionID uint64) ([]domain.ChatMessageRecord, error) {
	items := make([]domain.ChatMessageRecord, 0)
	for _, record := range r.records {
		if record.SessionID == sessionID {
			items = append(items, record)
		}
	}
	return items, nil
}

type agentAnalysisRepo struct {
	record *domain.AnalysisResultRecord
}

func (r agentAnalysisRepo) GetByDatasetID(context.Context, uint64) (*domain.AnalysisResultRecord, error) {
	if r.record == nil {
		return nil, errors.New("not found")
	}
	return r.record, nil
}
func (r agentAnalysisRepo) Upsert(context.Context, *domain.AnalysisResultRecord) error { return nil }
func (r agentAnalysisRepo) DeleteByDatasetID(context.Context, uint64) error            { return nil }

type agentClassificationRepo struct {
	record *domain.ClassificationResultRecord
}

func (r agentClassificationRepo) Create(context.Context, *domain.ClassificationResultRecord) error {
	return nil
}
func (r agentClassificationRepo) GetLatest(context.Context, uint64, string) (*domain.ClassificationResultRecord, error) {
	if r.record == nil {
		return nil, errors.New("not found")
	}
	return r.record, nil
}
func (r agentClassificationRepo) GetLatestByWindow(context.Context, uint64, string, *time.Time, *time.Time) (*domain.ClassificationResultRecord, error) {
	return nil, errors.New("not implemented")
}
func (r agentClassificationRepo) ListByDatasetID(context.Context, uint64, string, int, int) ([]domain.ClassificationResultRecord, int64, error) {
	return nil, 0, nil
}

type agentForecastRepo struct {
	records []domain.ForecastResultRecord
}

func (r agentForecastRepo) Create(context.Context, *domain.ForecastResultRecord) error { return nil }
func (r agentForecastRepo) GetByID(context.Context, uint64) (*domain.ForecastResultRecord, error) {
	return nil, errors.New("not implemented")
}
func (r agentForecastRepo) GetLatestByRange(context.Context, uint64, string, time.Time, time.Time, string) (*domain.ForecastResultRecord, error) {
	return nil, errors.New("not implemented")
}
func (r agentForecastRepo) ListByDatasetID(context.Context, uint64, string, int, int) ([]domain.ForecastResultRecord, int64, error) {
	return r.records, int64(len(r.records)), nil
}
func (r agentForecastRepo) ListAllByDatasetID(context.Context, uint64) ([]domain.ForecastResultRecord, error) {
	return r.records, nil
}

type agentAdviceRepo struct {
	records []domain.EnergyAdviceRecord
}

func (r agentAdviceRepo) Create(context.Context, *domain.EnergyAdviceRecord) error { return nil }
func (r agentAdviceRepo) GetByID(context.Context, uint64) (*domain.EnergyAdviceRecord, error) {
	return nil, errors.New("not implemented")
}
func (r agentAdviceRepo) ListByDatasetID(context.Context, uint64, string) ([]domain.EnergyAdviceRecord, error) {
	return r.records, nil
}
func (r agentAdviceRepo) DeleteByDatasetID(context.Context, uint64) error { return nil }

func buildAgentTestCSV() string {
	builder := strings.Builder{}
	builder.WriteString("timestamp,aggregate,active_appliance_count,burst_event_count\n")
	start := time.Date(2026, 3, 31, 0, 0, 0, 0, time.Local)
	for index := 0; index < 288; index++ {
		ts := start.Add(time.Duration(index) * 15 * time.Minute)
		builder.WriteString(ts.Format("2006-01-02 15:04:05"))
		builder.WriteString(",1.25,2,1\n")
	}
	return builder.String()
}

func uint64Ptr(value uint64) *uint64 {
	return &value
}
