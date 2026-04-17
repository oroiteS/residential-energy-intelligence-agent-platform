package service

import (
	"strings"
	"testing"
	"time"

	"residential-energy-intelligence-agent-platform/internal/domain"
	"residential-energy-intelligence-agent-platform/internal/integration/agentclient"
)

func TestRenderSimpleChinesePDF(t *testing.T) {
	now := time.Date(2026, 4, 6, 10, 30, 0, 0, time.Local)
	dataset := &domain.DatasetRecord{
		ID:          1,
		Name:        "house_1",
		HouseholdID: stringPtr("H001"),
		TimeStart:   &now,
		TimeEnd:     &now,
	}
	analysis := &domain.AnalysisResultRecord{
		TotalKWH:    123.4567,
		DailyAvgKWH: 12.3456,
		MaxLoadW:    4567.89,
		MinLoadW:    123.45,
		PeakRatio:   0.46,
		ValleyRatio: 0.21,
		FlatRatio:   0.33,
		MaxLoadTime: &now,
		MinLoadTime: &now,
	}
	summary := &ReportSummary{
		Title:    "居民用电分析报告 - house_1",
		Overview: "该家庭近期夜间负荷偏高，建议重点排查夜间持续运行设备。",
		Sections: []agentclient.ReportSection{
			{
				Title: "总体概览",
				Body:  "日均用电量约 12.35 kWh，峰时占比较高。",
			},
			{
				Title: "行为判断",
				Body:  "当前家庭表现为白天低晚上高型，夜间持续负荷较明显。",
			},
			{
				Title: "预测风险",
				Body:  "预测高负荷仍集中在晚间时段，需要重点关注夜间用电峰值。",
			},
			{
				Title: "附注",
				Body:  "本报告适合作为阶段性分析与归档材料。",
			},
		},
		Recommendations: []string{"优先检查夜间持续运行设备", "将热水器改为定时运行"},
	}

	lines := buildPDFReportLines(dataset, analysis, nil, summary)
	requiredTitles := []string{"总体概览", "行为判断", "预测风险", "节能建议", "附注"}
	for _, title := range requiredTitles {
		if !containsPDFTitle(lines, title) {
			t.Fatalf("缺少固定章节标题: %s", title)
		}
	}
	document, err := renderSimpleChinesePDF(lines)
	if err != nil {
		t.Fatalf("renderSimpleChinesePDF() 返回错误: %v", err)
	}

	content := string(document)
	if !strings.HasPrefix(content, "%PDF-1.4") {
		t.Fatalf("pdf 头不正确: %q", content[:8])
	}
	if !strings.Contains(content, "/BaseFont /STSong-Light") {
		t.Fatal("pdf 未写入中文字体定义")
	}
}

func TestRenderStyledChinesePDF(t *testing.T) {
	now := time.Date(2026, 4, 6, 10, 30, 0, 0, time.Local)
	dataset := &domain.DatasetRecord{
		ID:          1,
		Name:        "house_1",
		HouseholdID: stringPtr("H001"),
		TimeStart:   &now,
		TimeEnd:     &now,
	}
	analysis := &domain.AnalysisResultRecord{
		TotalKWH:    123.4567,
		DailyAvgKWH: 12.3456,
		MaxLoadW:    4567.89,
		MinLoadW:    123.45,
		PeakRatio:   0.46,
		ValleyRatio: 0.21,
		FlatRatio:   0.33,
		MaxLoadTime: &now,
		MinLoadTime: &now,
	}
	summary := &ReportSummary{
		Title:    "居民用电分析报告 - house_1",
		Overview: "该家庭近期夜间负荷偏高，建议重点排查夜间持续运行设备。",
		Sections: []agentclient.ReportSection{
			{Title: "总体概览", Body: "日均用电量约 12.35 kWh，峰时占比较高。"},
			{Title: "行为判断", Body: "当前家庭表现为白天低晚上高型，夜间持续负荷较明显。"},
			{Title: "预测风险", Body: "预测高负荷仍集中在晚间时段，需要重点关注夜间用电峰值。"},
			{Title: "附注", Body: "本报告适合作为阶段性分析与归档材料。"},
		},
		Recommendations: []string{"优先检查夜间持续运行设备", "将热水器改为定时运行"},
	}

	documentModel := buildPDFDocument(dataset, analysis, nil, summary)
	document, err := renderStyledChinesePDF(documentModel)
	if err != nil {
		t.Fatalf("renderStyledChinesePDF() 返回错误: %v", err)
	}

	content := string(document)
	if !strings.HasPrefix(content, "%PDF-1.4") {
		t.Fatalf("pdf 头不正确: %q", content[:8])
	}
	if !strings.Contains(content, "/BaseFont /STSong-Light") {
		t.Fatal("styled pdf 未写入中文字体定义")
	}
	if !strings.Contains(content, " re f") {
		t.Fatal("styled pdf 未绘制填充块")
	}
	if !strings.Contains(content, " rg") {
		t.Fatal("styled pdf 未写入颜色指令")
	}
}

func TestSanitizePDFTextRemovesBulletAndHumanizesTokens(t *testing.T) {
	value := sanitizePDFText("1. · 风险标签：full_mean 偏高，day_low_night_high。")
	if strings.Contains(value, "·") {
		t.Fatalf("sanitizePDFText() 未移除项目符号: %q", value)
	}
	if strings.Contains(value, "full_mean") {
		t.Fatalf("sanitizePDFText() 未转换技术字段: %q", value)
	}
	if strings.Contains(value, "day_low_night_high") {
		t.Fatalf("sanitizePDFText() 未转换分类标签: %q", value)
	}
	if !strings.Contains(value, "全天平均负荷") {
		t.Fatalf("sanitizePDFText() 未输出中文字段名: %q", value)
	}
	if !strings.Contains(value, "晚上高峰型") {
		t.Fatalf("sanitizePDFText() 未输出中文分类名: %q", value)
	}
}

func TestBuildPDFDocumentSanitizesRecommendations(t *testing.T) {
	now := time.Date(2026, 4, 6, 10, 30, 0, 0, time.Local)
	dataset := &domain.DatasetRecord{
		ID:          1,
		Name:        "house_1",
		HouseholdID: stringPtr("H001"),
		TimeStart:   &now,
		TimeEnd:     &now,
	}
	analysis := &domain.AnalysisResultRecord{
		TotalKWH:    123.4567,
		DailyAvgKWH: 12.3456,
		MaxLoadW:    4567.89,
		MinLoadW:    123.45,
		PeakRatio:   0.46,
		ValleyRatio: 0.21,
		FlatRatio:   0.33,
		MaxLoadTime: &now,
		MinLoadTime: &now,
	}
	summary := &ReportSummary{
		Overview: "· 当前总体负荷平稳，但 night_baseload_high 需要关注。",
		Sections: []agentclient.ReportSection{
			{Title: "预测风险", Body: "1. · 风险标签包含 evening_peak_risk 与 full_mean 偏高。"},
		},
		Recommendations: []string{
			"1. · 优先排查夜间持续运行设备。",
			"2. · 将热水器改为定时运行。",
		},
	}

	document := buildPDFDocument(dataset, analysis, nil, summary)
	for _, item := range document.Recommendations {
		if strings.Contains(item, "·") {
			t.Fatalf("buildPDFDocument() 保留了项目符号: %q", item)
		}
	}
	if strings.Contains(document.Overview, "night_baseload_high") {
		t.Fatalf("buildPDFDocument() 未转换风险标签: %q", document.Overview)
	}
	if !strings.Contains(document.Overview, "夜间基础负荷偏高") {
		t.Fatalf("buildPDFDocument() 未输出中文风险标签: %q", document.Overview)
	}
	if len(document.Sections) != 1 {
		t.Fatalf("buildPDFDocument() 章节数量异常: %d", len(document.Sections))
	}
	if strings.Contains(document.Sections[0].Body, "·") {
		t.Fatalf("buildPDFDocument() 未清洗章节正文: %q", document.Sections[0].Body)
	}
}

func TestBuildReportMarkdownIncludesStructuredSections(t *testing.T) {
	now := time.Date(2026, 4, 6, 10, 30, 0, 0, time.Local)
	document := pdfDocument{
		Title:       "居民用电分析报告",
		ExportedAt:  now.Format("2006-01-02 15:04:05"),
		DatasetName: "house_1",
		HouseholdID: "H001",
		TimeRange:   "2026-04-01 00:00 至 2026-04-07 23:45",
		Overview:    "该家庭近期夜间负荷偏高，建议排查持续运行设备。",
		InfoRows: []pdfInfoRow{
			{Label: "数据集名称", Value: "house_1"},
			{Label: "家庭标识", Value: "H001"},
		},
		Metrics: []pdfMetricCard{
			{Label: "总用电量", Value: "123.4567 千瓦时", Hint: "整个样本时间范围内累计用电量"},
		},
		Sections: []pdfSectionCard{
			{Title: "预测风险", Body: "预测高负荷仍集中在晚间时段。"},
		},
		Recommendations: []string{"优先检查夜间持续运行设备", "将热水器改为定时运行"},
	}

	markdown := buildReportMarkdown(document)
	requiredFragments := []string{
		"# 居民用电分析报告",
		"## 摘要概览",
		"| 项目 | 内容 |",
		"| 指标 | 数值 | 说明 |",
		"## 预测风险",
		"## 节能建议",
		"1. 优先检查夜间持续运行设备",
	}
	for _, fragment := range requiredFragments {
		if !strings.Contains(markdown, fragment) {
			t.Fatalf("buildReportMarkdown() 缺少片段: %s\n%s", fragment, markdown)
		}
	}
}

func containsPDFTitle(lines []pdfTextLine, title string) bool {
	for _, line := range lines {
		if line.Text == title {
			return true
		}
	}
	return false
}
