package service

import (
	"bytes"
	"context"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode"
	"unicode/utf16"
	"unicode/utf8"

	"residential-energy-intelligence-agent-platform/internal/domain"
	"residential-energy-intelligence-agent-platform/internal/integration/agentclient"
)

type pdfTextLine struct {
	Text string
	Size float64
}

type pdfColor struct {
	R float64
	G float64
	B float64
}

type pdfInfoRow struct {
	Label string
	Value string
}

type pdfMetricCard struct {
	Label string
	Value string
	Hint  string
}

type pdfSectionCard struct {
	Title string
	Body  string
}

type pdfDocument struct {
	Title           string
	ExportedAt      string
	DatasetName     string
	HouseholdID     string
	TimeRange       string
	Overview        string
	InfoRows        []pdfInfoRow
	Metrics         []pdfMetricCard
	Sections        []pdfSectionCard
	Recommendations []string
}

type pdfPageRenderer struct {
	streams    []string
	builder    strings.Builder
	pageNumber int
	y          float64
}

var (
	pdfColorInk        = pdfColor{R: 0.16, G: 0.20, B: 0.18}
	pdfColorMuted      = pdfColor{R: 0.40, G: 0.43, B: 0.40}
	pdfColorAccent     = pdfColor{R: 0.38, G: 0.47, B: 0.40}
	pdfColorWarm       = pdfColor{R: 0.78, G: 0.65, B: 0.44}
	pdfColorBorder     = pdfColor{R: 0.86, G: 0.81, B: 0.72}
	pdfColorPanel      = pdfColor{R: 0.97, G: 0.95, B: 0.92}
	pdfColorHero       = pdfColor{R: 0.95, G: 0.91, B: 0.84}
	pdfColorSection    = pdfColor{R: 0.91, G: 0.95, B: 0.91}
	pdfColorMetricFill = pdfColor{R: 0.99, G: 0.98, B: 0.96}
	pdfColorPaper      = pdfColor{R: 1.00, G: 1.00, B: 1.00}
	pdfTextReplacer    = strings.NewReplacer(
		"·", " ",
		"•", " ",
		"●", " ",
		"▪", " ",
		"▫", " ",
		"◦", " ",
		"○", " ",
		"daytime_active", "下午高峰型",
		"daytime_peak_strong", "上午高峰型",
		"flat_stable", "全天平稳型",
		"night_dominant", "晚上高峰型",
		"day_high_night_low", "下午高峰型",
		"all_day_high", "上午高峰型",
		"afternoon_peak", "下午高峰型",
		"day_low_night_high", "晚上高峰型",
		"all_day_low", "全天平稳型",
		"morning_peak", "上午高峰型",
		"day_mean", "白天平均负荷",
		"night_mean", "夜间平均负荷",
		"full_mean", "全天平均负荷",
		"evening_peak_risk", "晚间峰值偏高",
		"night_baseload_high", "夜间基础负荷偏高",
		"load_volatility_high", "负荷波动偏大",
		"morning_ramp_high", "早高峰爬升明显",
		"risk_flags", "风险标签",
		"kWh", "千瓦时",
	)
)

const (
	pdfPageWidth      = 595.0
	pdfPageHeight     = 842.0
	pdfMarginLeft     = 40.0
	pdfMarginRight    = 40.0
	pdfMarginTop      = 795.0
	pdfMarginBottom   = 58.0
	pdfContentWidth   = pdfPageWidth - pdfMarginLeft - pdfMarginRight
	pdfSectionGap     = 18.0
	pdfLineGap        = 6.0
	pdfParagraphSize  = 11.5
	pdfTitleStripH    = 22.0
	pdfRecommendation = "建议优先复核峰时段与夜间持续运行设备的负荷安排。"
)

func (s *ReportService) writePDFReport(
	ctx context.Context,
	dataset *domain.DatasetRecord,
	analysis *domain.AnalysisResultRecord,
	advices []domain.EnergyAdviceRecord,
	summary *ReportSummary,
) (string, uint64, error) {
	reportDir := filepath.Join(s.cfg.OutputRootDir, "reports")
	if err := os.MkdirAll(reportDir, 0o755); err != nil {
		return "", 0, err
	}

	documentModel := buildPDFDocument(dataset, analysis, advices, summary)
	baseName := fmt.Sprintf("report_%d_%d", dataset.ID, time.Now().UnixNano())
	markdownPath := filepath.ToSlash(filepath.Join(reportDir, baseName+".md"))
	pdfPath := filepath.ToSlash(filepath.Join(reportDir, baseName+".pdf"))

	markdownContent := buildReportMarkdown(documentModel)
	if err := os.WriteFile(markdownPath, []byte(markdownContent), 0o644); err != nil {
		return "", 0, err
	}

	if err := s.convertMarkdownToPDF(ctx, markdownPath, pdfPath, documentModel); err != nil {
		return "", 0, err
	}

	info, err := os.Stat(pdfPath)
	if err != nil {
		return "", 0, err
	}
	return pdfPath, uint64(info.Size()), nil
}

func buildPDFDocument(
	dataset *domain.DatasetRecord,
	analysis *domain.AnalysisResultRecord,
	advices []domain.EnergyAdviceRecord,
	summary *ReportSummary,
) pdfDocument {
	exportedAt := time.Now().Format("2006-01-02 15:04:05")
	title := "居民用电分析报告"
	overview := "当前报告基于系统内已有统计分析、分类结果、预测摘要与规则建议自动整理。"
	sections := make([]pdfSectionCard, 0, 6)
	recommendations := make([]string, 0, 8)

	if summary != nil {
		if strings.TrimSpace(summary.Title) != "" {
			title = sanitizePDFText(summary.Title)
		}
		if strings.TrimSpace(summary.Overview) != "" {
			overview = sanitizePDFText(summary.Overview)
		}
		for _, section := range summary.Sections {
			titleValue := sanitizePDFText(section.Title)
			bodyValue := sanitizePDFText(section.Body)
			if titleValue == "" || bodyValue == "" {
				continue
			}
			sections = append(sections, pdfSectionCard{
				Title: titleValue,
				Body:  bodyValue,
			})
		}
		for _, item := range summary.Recommendations {
			value := sanitizePDFText(item)
			if value != "" && !containsString(recommendations, value) {
				recommendations = append(recommendations, value)
			}
		}
	}

	if len(recommendations) == 0 {
		for _, advice := range advices {
			value := sanitizePDFText(stringValue(nullableString(advice.Summary)))
			if value != "" && !containsString(recommendations, value) {
				recommendations = append(recommendations, value)
			}
		}
	}
	if len(recommendations) == 0 {
		recommendations = append(recommendations, pdfRecommendation)
	}

	timeRange := formatTimeValue(dataset.TimeStart) + " 至 " + formatTimeValue(dataset.TimeEnd)
	peakRatioText := fmt.Sprintf(
		"峰 %.2f%% / 谷 %.2f%% / 平 %.2f%%",
		analysis.PeakRatio*100,
		analysis.ValleyRatio*100,
		analysis.FlatRatio*100,
	)

	return pdfDocument{
		Title:       sanitizePDFText(title),
		ExportedAt:  exportedAt,
		DatasetName: sanitizePDFText(dataset.Name),
		HouseholdID: sanitizePDFText(stringValue(nullableString(dataset.HouseholdID))),
		TimeRange:   sanitizePDFText(timeRange),
		Overview:    sanitizePDFText(overview),
		InfoRows: []pdfInfoRow{
			{Label: "数据集名称", Value: sanitizePDFText(dataset.Name)},
			{Label: "家庭标识", Value: sanitizePDFText(stringValue(nullableString(dataset.HouseholdID)))},
			{Label: "时间范围", Value: sanitizePDFText(timeRange)},
		},
		Metrics: []pdfMetricCard{
			{
				Label: "总用电量",
				Value: fmt.Sprintf("%.4f 千瓦时", analysis.TotalKWH),
				Hint:  "整个样本时间范围内累计用电量",
			},
			{
				Label: "日均用电量",
				Value: fmt.Sprintf("%.4f 千瓦时", analysis.DailyAvgKWH),
				Hint:  "按完整日聚合后的平均水平",
			},
			{
				Label: "最高负荷",
				Value: fmt.Sprintf("%.2f 瓦", analysis.MaxLoadW),
				Hint:  formatTimeValue(analysis.MaxLoadTime),
			},
			{
				Label: "最低负荷",
				Value: fmt.Sprintf("%.2f 瓦", analysis.MinLoadW),
				Hint:  formatTimeValue(analysis.MinLoadTime),
			},
			{
				Label: "峰谷平结构",
				Value: peakRatioText,
				Hint:  "用于判断时段负荷分布是否集中",
			},
		},
		Sections:        sections,
		Recommendations: recommendations,
	}
}

func buildReportMarkdown(document pdfDocument) string {
	lines := []string{
		"# " + firstNonEmpty(document.Title, "居民用电分析报告"),
		"",
		"> 导出时间：" + firstNonEmpty(document.ExportedAt, "-"),
		"",
		"## 摘要概览",
		"",
		firstNonEmpty(document.Overview, "当前暂无可导出的摘要内容。"),
		"",
		"## 数据集信息",
		"",
		"| 项目 | 内容 |",
		"| --- | --- |",
	}

	for _, row := range document.InfoRows {
		lines = append(lines, fmt.Sprintf("| %s | %s |", escapeMarkdownTableCell(row.Label), escapeMarkdownTableCell(firstNonEmpty(row.Value, "-"))))
	}

	lines = append(lines,
		"",
		"## 关键指标",
		"",
		"| 指标 | 数值 | 说明 |",
		"| --- | --- | --- |",
	)
	for _, metric := range document.Metrics {
		lines = append(lines, fmt.Sprintf(
			"| %s | %s | %s |",
			escapeMarkdownTableCell(metric.Label),
			escapeMarkdownTableCell(firstNonEmpty(metric.Value, "-")),
			escapeMarkdownTableCell(firstNonEmpty(metric.Hint, "-")),
		))
	}

	for _, section := range document.Sections {
		title := firstNonEmpty(section.Title, "")
		body := firstNonEmpty(section.Body, "")
		if title == "" || body == "" {
			continue
		}
		lines = append(lines, "", "## "+title, "", body)
	}

	if len(document.Recommendations) > 0 {
		lines = append(lines, "", "## 节能建议", "")
		for index, item := range document.Recommendations {
			lines = append(lines, fmt.Sprintf("%d. %s", index+1, firstNonEmpty(item, "-")))
		}
	}

	return strings.Join(lines, "\n") + "\n"
}

func escapeMarkdownTableCell(value string) string {
	text := sanitizePDFText(value)
	text = strings.ReplaceAll(text, "\n", "<br/>")
	text = strings.ReplaceAll(text, "|", "\\|")
	return text
}

func (s *ReportService) convertMarkdownToPDF(ctx context.Context, markdownPath string, pdfPath string, document pdfDocument) error {
	if s.agentService == nil || s.agentService.agentClient == nil {
		return fmt.Errorf("智能体 PDF 服务未初始化")
	}

	markdownBytes, err := os.ReadFile(markdownPath)
	if err != nil {
		return err
	}

	documentBytes, err := s.agentService.agentClient.RenderPDF(ctx, agentclient.RenderPDFRequest{
		Markdown:    string(markdownBytes),
		Title:       firstNonEmpty(document.Title, "居民用电分析报告"),
		Author:      "居民能源智能分析平台",
		Date:        firstNonEmpty(document.ExportedAt, time.Now().Format("2006-01-02")),
		Theme:       firstNonEmpty(strings.TrimSpace(s.cfg.ReportPDFTheme), "github-light"),
		Cover:       s.cfg.ReportPDFCover,
		TOC:         s.cfg.ReportPDFTOC,
		HeaderTitle: firstNonEmpty(document.Title, "居民用电分析报告"),
		FooterLeft:  "居民能源智能分析平台",
	})
	if err != nil {
		return fmt.Errorf("markdown 转 pdf 失败: %w", err)
	}

	if err := os.WriteFile(pdfPath, documentBytes, 0o644); err != nil {
		return err
	}
	return nil
}

func renderStyledChinesePDF(document pdfDocument) ([]byte, error) {
	pageStreams, err := buildStyledPDFPageStreams(document)
	if err != nil {
		return nil, err
	}
	if len(pageStreams) == 0 {
		return nil, fmt.Errorf("pdf 内容为空")
	}

	var output bytes.Buffer
	offsets := []int{0}
	writeObject := func(number int, body string) {
		offsets = append(offsets, output.Len())
		_, _ = fmt.Fprintf(&output, "%d 0 obj\n%s\nendobj\n", number, body)
	}

	output.WriteString("%PDF-1.4\n%\xE2\xE3\xCF\xD3\n")

	fontObject := 3 + len(pageStreams)*2
	descendantFontObject := fontObject + 1

	writeObject(1, "<< /Type /Catalog /Pages 2 0 R >>")

	kids := make([]string, 0, len(pageStreams))
	for index := range pageStreams {
		pageObject := 3 + index*2
		kids = append(kids, fmt.Sprintf("%d 0 R", pageObject))
	}
	writeObject(2, fmt.Sprintf("<< /Type /Pages /Kids [%s] /Count %d >>", strings.Join(kids, " "), len(pageStreams)))

	for index, stream := range pageStreams {
		pageObject := 3 + index*2
		contentObject := pageObject + 1
		writeObject(
			pageObject,
			fmt.Sprintf(
				"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.0f %.0f] /Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>",
				pdfPageWidth,
				pdfPageHeight,
				fontObject,
				contentObject,
			),
		)
		writeObject(contentObject, fmt.Sprintf("<< /Length %d >>\nstream\n%s\nendstream", len(stream), stream))
	}

	writeObject(fontObject, fmt.Sprintf("<< /Type /Font /Subtype /Type0 /BaseFont /STSong-Light /Encoding /UniGB-UCS2-H /DescendantFonts [%d 0 R] >>", descendantFontObject))
	writeObject(descendantFontObject, "<< /Type /Font /Subtype /CIDFontType0 /BaseFont /STSong-Light /CIDSystemInfo << /Registry (Adobe) /Ordering (GB1) /Supplement 4 >> /DW 1000 >>")

	xrefOffset := output.Len()
	_, _ = fmt.Fprintf(&output, "xref\n0 %d\n", len(offsets))
	output.WriteString("0000000000 65535 f \n")
	for index := 1; index < len(offsets); index++ {
		_, _ = fmt.Fprintf(&output, "%010d 00000 n \n", offsets[index])
	}
	_, _ = fmt.Fprintf(&output, "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF", len(offsets), xrefOffset)
	return output.Bytes(), nil
}

func buildStyledPDFPageStreams(document pdfDocument) ([]string, error) {
	renderer := &pdfPageRenderer{}
	renderer.startPage()

	renderer.drawHero(document)
	renderer.drawInfoTable("数据集信息", document.InfoRows)
	renderer.drawMetricCards("关键指标", document.Metrics)

	renderedOverviewSection := false
	for _, section := range document.Sections {
		title := strings.TrimSpace(section.Title)
		body := strings.TrimSpace(section.Body)
		if title == "" || body == "" {
			continue
		}
		if title == "总体概览" {
			renderedOverviewSection = true
		}
		if title == "附注" {
			renderer.drawNoteCard(title, body)
			continue
		}
		renderer.drawTextCard(title, []string{body}, pdfColorMetricFill, pdfColorAccent)
	}
	if !renderedOverviewSection && strings.TrimSpace(document.Overview) != "" {
		renderer.drawTextCard(
			"总体概览",
			[]string{document.Overview},
			pdfColorMetricFill,
			pdfColorAccent,
		)
	}

	renderer.drawRecommendationCard("节能建议", document.Recommendations)
	renderer.finishPage()

	return renderer.streams, nil
}

func (r *pdfPageRenderer) startPage() {
	r.builder.Reset()
	r.pageNumber++
	r.y = pdfMarginTop
	r.fillRect(18, 18, pdfPageWidth-36, pdfPageHeight-36, pdfColor{R: 1, G: 1, B: 1})
	r.strokeRect(18, 18, pdfPageWidth-36, pdfPageHeight-36, pdfColorBorder, 1)
	r.line(pdfMarginLeft, pdfPageHeight-34, pdfPageWidth-pdfMarginRight, pdfPageHeight-34, pdfColorBorder, 1.2)
}

func (r *pdfPageRenderer) finishPage() {
	if r.builder.Len() == 0 {
		return
	}
	r.line(pdfMarginLeft, 42, pdfPageWidth-pdfMarginRight, 42, pdfColorBorder, 0.8)
	r.drawText(pdfMarginLeft, 26, 10, pdfColorMuted, "居民用电分析报告")
	r.drawText(pdfPageWidth-pdfMarginRight-42, 26, 10, pdfColorMuted, fmt.Sprintf("第 %d 页", r.pageNumber))
	r.streams = append(r.streams, r.builder.String())
}

func (r *pdfPageRenderer) ensureSpace(height float64) {
	if r.y-height >= pdfMarginBottom {
		return
	}
	r.finishPage()
	r.startPage()
	// 后续分页页首增加轻量标题，避免内容直接顶到顶部。
	r.fillRect(pdfMarginLeft, r.y-28, pdfContentWidth, 22, pdfColorPanel)
	r.drawText(pdfMarginLeft+12, r.y-20, 10.5, pdfColorMuted, "居民用电分析报告")
	r.y -= 42
}

func (r *pdfPageRenderer) drawHero(document pdfDocument) {
	const (
		blockHeight   = 164.0
		metaCardWidth = 172.0
	)
	r.ensureSpace(blockHeight)
	top := r.y
	bottom := top - blockHeight

	r.fillRect(pdfMarginLeft, bottom, pdfContentWidth, blockHeight, pdfColorHero)
	r.fillRect(pdfMarginLeft, top-10, pdfContentWidth, 10, pdfColorAccent)
	r.strokeRect(pdfMarginLeft, bottom, pdfContentWidth, blockHeight, pdfColorBorder, 0.8)

	metaX := pdfMarginLeft + pdfContentWidth - metaCardWidth - 18
	metaTop := top - 18
	metaBottom := bottom + 18
	metaHeight := metaTop - metaBottom

	r.fillRect(metaX, metaBottom, metaCardWidth, metaHeight, pdfColorPaper)
	r.strokeRect(metaX, metaBottom, metaCardWidth, metaHeight, pdfColorBorder, 0.8)
	r.fillRect(metaX, metaTop-5, metaCardWidth, 5, pdfColorWarm)

	leftX := pdfMarginLeft + 18
	leftWidth := metaX - leftX - 18
	titleLines := limitPDFLines(wrapPDFText(document.Title, 24), 2)
	r.drawText(leftX, top-28, 10, pdfColorMuted, "居民用电分析报告")
	r.drawWrappedText(leftX, top-52, 22, pdfColorInk, titleLines, 6)

	titleHeight := estimateWrappedTextHeight(len(titleLines), 22, 6)
	summaryLabelY := top - 52 - titleHeight - 12
	r.drawText(leftX, summaryLabelY, 10.5, pdfColorMuted, "摘要提要")

	overviewLines := limitPDFLines(wrapPDFText(document.Overview, minInt(48, maxInt(28, int(leftWidth/8)))), 3)
	r.drawWrappedText(leftX, summaryLabelY-18, 11.2, pdfColorInk, overviewLines, 5.5)

	metaY := metaTop - 18
	r.drawText(metaX+12, metaY, 10, pdfColorMuted, "报告信息")
	metaY -= 18
	metaY = r.drawHeroMetaRow(metaX+12, metaY, "导出时间", document.ExportedAt)
	metaY = r.drawHeroMetaRow(metaX+12, metaY, "数据集", firstNonEmpty(document.DatasetName, "-"))
	metaY = r.drawHeroMetaRow(metaX+12, metaY, "家庭标识", firstNonEmpty(document.HouseholdID, "-"))
	r.drawHeroMetaRow(metaX+12, metaY, "时间范围", firstNonEmpty(document.TimeRange, "-"))

	r.y = bottom - pdfSectionGap
}

func (r *pdfPageRenderer) drawHeroMetaRow(x, startY float64, label string, value string) float64 {
	r.drawText(x, startY, 9.5, pdfColorMuted, label)
	valueLines := wrapPDFText(value, 20)
	r.drawWrappedText(x, startY-14, 10.5, pdfColorInk, valueLines, 4.5)
	return startY - 14 - estimateWrappedTextHeight(len(valueLines), 10.5, 4.5) - 10
}

func (r *pdfPageRenderer) drawInfoTable(title string, rows []pdfInfoRow) {
	lineCount := 0
	for _, row := range rows {
		lineCount += len(wrapPDFText(firstNonEmpty(row.Value, "-"), 56))
	}
	blockHeight := 24.0 + 20.0 + float64(len(rows))*18.0 + float64(lineCount-len(rows))*16.0 + 18.0
	r.ensureSpace(blockHeight)
	r.drawSectionTitle(title)

	cardTop := r.y
	cardHeight := blockHeight - 24.0
	cardBottom := cardTop - cardHeight
	r.fillRect(pdfMarginLeft, cardBottom, pdfContentWidth, cardHeight, pdfColorPanel)
	r.strokeRect(pdfMarginLeft, cardBottom, pdfContentWidth, cardHeight, pdfColorBorder, 0.8)

	currentY := cardTop - 24
	for index, row := range rows {
		r.drawText(pdfMarginLeft+16, currentY, 11, pdfColorMuted, row.Label)
		valueLines := wrapPDFText(firstNonEmpty(row.Value, "-"), 56)
		r.drawWrappedText(pdfMarginLeft+110, currentY, 11.5, pdfColorInk, valueLines, 5.5)
		currentY -= float64(maxInt(1, len(valueLines)))*17 + 6
		if index < len(rows)-1 {
			r.line(pdfMarginLeft+14, currentY+2, pdfMarginLeft+pdfContentWidth-14, currentY+2, pdfColorBorder, 0.6)
			currentY -= 8
		}
	}

	r.y = cardBottom - pdfSectionGap
}

func (r *pdfPageRenderer) drawMetricCards(title string, cards []pdfMetricCard) {
	if len(cards) == 0 {
		return
	}

	cardWidth := (pdfContentWidth - 14) / 2
	layoutHeight := 24.0
	for index, card := range cards {
		width := cardWidth
		if index == 4 {
			width = pdfContentWidth
		}
		layoutHeight += r.metricCardHeight(width, card)
		if index == 1 || index == 3 {
			layoutHeight += 16
		}
	}
	layoutHeight += 12
	r.ensureSpace(layoutHeight)
	r.drawSectionTitle(title)

	rowOneTop := r.y
	firstRowHeight := maxFloat64(
		r.metricCardHeight(cardWidth, cards[0]),
		conditionalMetricHeight(len(cards) > 1, r.metricCardHeight(cardWidth, cards[1])),
	)
	r.drawMetricCard(pdfMarginLeft, rowOneTop, cardWidth, firstRowHeight, cards[0])
	if len(cards) > 1 {
		r.drawMetricCard(pdfMarginLeft+cardWidth+14, rowOneTop, cardWidth, firstRowHeight, cards[1])
	}

	rowTwoTop := rowOneTop - firstRowHeight - 16
	secondRowHeight := 0.0
	if len(cards) > 2 {
		secondRowHeight = maxFloat64(secondRowHeight, r.metricCardHeight(cardWidth, cards[2]))
	}
	if len(cards) > 3 {
		secondRowHeight = maxFloat64(secondRowHeight, r.metricCardHeight(cardWidth, cards[3]))
	}
	if secondRowHeight > 0 {
		if len(cards) > 2 {
			r.drawMetricCard(pdfMarginLeft, rowTwoTop, cardWidth, secondRowHeight, cards[2])
		}
		if len(cards) > 3 {
			r.drawMetricCard(pdfMarginLeft+cardWidth+14, rowTwoTop, cardWidth, secondRowHeight, cards[3])
		}
	}

	if len(cards) > 4 {
		rowThreeTop := rowTwoTop - secondRowHeight - 16
		thirdRowHeight := r.metricCardHeight(pdfContentWidth, cards[4])
		r.drawMetricCard(pdfMarginLeft, rowThreeTop, pdfContentWidth, thirdRowHeight, cards[4])
		r.y = rowThreeTop - thirdRowHeight - 16
		return
	}
	if secondRowHeight > 0 {
		r.y = rowTwoTop - secondRowHeight - 16
		return
	}
	r.y = rowTwoTop
}

func (r *pdfPageRenderer) drawMetricCard(x, top, width, height float64, card pdfMetricCard) {
	bottom := top - height
	r.fillRect(x, bottom, width, height, pdfColorMetricFill)
	r.strokeRect(x, bottom, width, height, pdfColorBorder, 0.8)
	r.fillRect(x, top-6, width, 6, pdfColorSection)
	r.drawText(x+14, top-18, 10.5, pdfColorMuted, card.Label)
	r.drawText(x+14, top-40, 16, pdfColorInk, card.Value)

	hintLines := wrapPDFText(strings.TrimSpace(card.Hint), 24)
	if width > 300 {
		hintLines = wrapPDFText(strings.TrimSpace(card.Hint), 48)
	}
	r.line(x+14, top-50, x+width-14, top-50, pdfColorBorder, 0.6)
	r.drawWrappedText(x+14, top-66, 10, pdfColorMuted, hintLines, 4.5)
}

func (r *pdfPageRenderer) metricCardHeight(width float64, card pdfMetricCard) float64 {
	hintLines := wrapPDFText(strings.TrimSpace(card.Hint), 24)
	if width > 300 {
		hintLines = wrapPDFText(strings.TrimSpace(card.Hint), 48)
	}
	hintHeight := estimateWrappedTextHeight(len(hintLines), 10, 4.5)
	baseHeight := 78.0 + hintHeight
	if baseHeight < 86 {
		return 86
	}
	return baseHeight
}

func (r *pdfPageRenderer) drawTextCard(title string, paragraphs []string, fillColor pdfColor, accentColor pdfColor) {
	maxUnits := 56
	totalLines := 0
	for _, paragraph := range paragraphs {
		totalLines += len(wrapPDFText(strings.TrimSpace(paragraph), maxUnits))
	}
	if totalLines == 0 {
		return
	}

	blockHeight := 24.0 + 20.0 + float64(totalLines)*17.0 + 18.0
	r.ensureSpace(blockHeight)
	r.drawSectionTitle(title)

	cardTop := r.y
	cardHeight := blockHeight - 24.0
	cardBottom := cardTop - cardHeight
	r.fillRect(pdfMarginLeft, cardBottom, pdfContentWidth, cardHeight, fillColor)
	r.strokeRect(pdfMarginLeft, cardBottom, pdfContentWidth, cardHeight, pdfColorBorder, 0.8)
	r.fillRect(pdfMarginLeft, cardTop-5, pdfContentWidth, 5, accentColor)

	currentY := cardTop - 22
	for _, paragraph := range paragraphs {
		lines := wrapPDFText(strings.TrimSpace(paragraph), maxUnits)
		r.drawWrappedText(pdfMarginLeft+16, currentY, pdfParagraphSize, pdfColorInk, lines, pdfLineGap)
		currentY -= float64(len(lines))*(pdfParagraphSize+pdfLineGap) + 6
	}
	r.y = cardBottom - pdfSectionGap
}

func (r *pdfPageRenderer) drawRecommendationCard(title string, items []string) {
	if len(items) == 0 {
		return
	}

	totalLines := 0
	for _, item := range items {
		totalLines += len(wrapPDFText(strings.TrimSpace(item), 48))
	}
	blockHeight := 24.0 + 22.0 + float64(totalLines)*17.0 + float64(len(items)-1)*10.0 + 22.0
	r.ensureSpace(blockHeight)
	r.drawSectionTitle(title)

	cardTop := r.y
	cardHeight := blockHeight - 24.0
	cardBottom := cardTop - cardHeight
	r.fillRect(pdfMarginLeft, cardBottom, pdfContentWidth, cardHeight, pdfColorPanel)
	r.strokeRect(pdfMarginLeft, cardBottom, pdfContentWidth, cardHeight, pdfColorBorder, 0.8)
	r.fillRect(pdfMarginLeft, cardTop-5, pdfContentWidth, 5, pdfColorWarm)

	currentY := cardTop - 22
	for index, item := range items {
		lines := wrapPDFText(strings.TrimSpace(item), 48)
		itemHeight := estimateWrappedTextHeight(len(lines), pdfParagraphSize, pdfLineGap)
		badgeTop := currentY + 2
		badgeBottom := badgeTop - 16
		r.fillRect(pdfMarginLeft+16, badgeBottom, 16, 16, pdfColorWarm)
		r.drawText(pdfMarginLeft+21, currentY-9, 9, pdfColorPaper, fmt.Sprintf("%d", index+1))
		r.drawWrappedText(pdfMarginLeft+42, currentY, pdfParagraphSize, pdfColorInk, lines, pdfLineGap)
		currentY -= itemHeight + 10
	}
	r.y = cardBottom - pdfSectionGap
}

func (r *pdfPageRenderer) drawNoteCard(title string, body string) {
	lines := wrapPDFText(strings.TrimSpace(body), 58)
	if len(lines) == 0 {
		return
	}

	blockHeight := 24.0 + 20.0 + estimateWrappedTextHeight(len(lines), 10.8, 5.0) + 18.0
	r.ensureSpace(blockHeight)
	r.drawSectionTitle(title)

	cardTop := r.y
	cardHeight := blockHeight - 24.0
	cardBottom := cardTop - cardHeight
	r.fillRect(pdfMarginLeft, cardBottom, pdfContentWidth, cardHeight, pdfColorPanel)
	r.strokeRect(pdfMarginLeft, cardBottom, pdfContentWidth, cardHeight, pdfColorBorder, 0.8)

	r.drawText(pdfMarginLeft+16, cardTop-20, 10, pdfColorMuted, "导出说明")
	r.drawWrappedText(pdfMarginLeft+16, cardTop-40, 10.8, pdfColorMuted, lines, 5.0)
	r.y = cardBottom - pdfSectionGap
}

func (r *pdfPageRenderer) drawSectionTitle(title string) {
	r.drawText(pdfMarginLeft, r.y, 13, pdfColorAccent, title)
	r.line(pdfMarginLeft+72, r.y-4, pdfMarginLeft+pdfContentWidth, r.y-4, pdfColorBorder, 1.0)
	r.y -= 24
}

func (r *pdfPageRenderer) drawWrappedText(
	x float64,
	startY float64,
	size float64,
	color pdfColor,
	lines []string,
	lineGap float64,
) {
	currentY := startY
	for _, line := range lines {
		r.drawText(x, currentY, size, color, line)
		currentY -= size + lineGap
	}
}

func (r *pdfPageRenderer) drawText(x, y, size float64, color pdfColor, text string) {
	text = sanitizePDFText(text)
	if strings.TrimSpace(text) == "" {
		return
	}
	r.builder.WriteString("BT\n")
	r.builder.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", size))
	r.builder.WriteString(fmt.Sprintf("%.3f %.3f %.3f rg\n", color.R, color.G, color.B))
	r.builder.WriteString(fmt.Sprintf("1 0 0 1 %.2f %.2f Tm\n", x, y))
	r.builder.WriteString(fmt.Sprintf("<%s> Tj\n", encodePDFText(text)))
	r.builder.WriteString("ET\n")
}

func (r *pdfPageRenderer) fillRect(x, y, width, height float64, color pdfColor) {
	r.builder.WriteString(fmt.Sprintf("%.3f %.3f %.3f rg\n", color.R, color.G, color.B))
	r.builder.WriteString(fmt.Sprintf("%.2f %.2f %.2f %.2f re f\n", x, y, width, height))
}

func (r *pdfPageRenderer) strokeRect(x, y, width, height float64, color pdfColor, lineWidth float64) {
	r.builder.WriteString(fmt.Sprintf("%.2f w\n", lineWidth))
	r.builder.WriteString(fmt.Sprintf("%.3f %.3f %.3f RG\n", color.R, color.G, color.B))
	r.builder.WriteString(fmt.Sprintf("%.2f %.2f %.2f %.2f re S\n", x, y, width, height))
}

func (r *pdfPageRenderer) line(x1, y1, x2, y2 float64, color pdfColor, lineWidth float64) {
	r.builder.WriteString(fmt.Sprintf("%.2f w\n", lineWidth))
	r.builder.WriteString(fmt.Sprintf("%.3f %.3f %.3f RG\n", color.R, color.G, color.B))
	r.builder.WriteString(fmt.Sprintf("%.2f %.2f m %.2f %.2f l S\n", x1, y1, x2, y2))
}

func buildPDFReportLines(
	dataset *domain.DatasetRecord,
	analysis *domain.AnalysisResultRecord,
	advices []domain.EnergyAdviceRecord,
	summary *ReportSummary,
) []pdfTextLine {
	lines := make([]pdfTextLine, 0, 48)

	title := "居民用电分析报告"
	if summary != nil && strings.TrimSpace(summary.Title) != "" {
		title = sanitizePDFText(summary.Title)
	}
	lines = append(lines, pdfTextLine{Text: title, Size: 18})
	lines = append(lines, pdfTextLine{Text: "导出时间：" + time.Now().Format("2006-01-02 15:04:05"), Size: 10})
	lines = append(lines, pdfTextLine{Text: "", Size: 12})

	lines = append(lines, pdfTextLine{Text: "数据集信息", Size: 14})
	lines = append(lines, pdfTextLine{Text: "数据集名称：" + sanitizePDFText(dataset.Name), Size: 12})
	lines = append(lines, pdfTextLine{Text: "家庭标识：" + sanitizePDFText(stringValue(nullableString(dataset.HouseholdID))), Size: 12})
	lines = append(lines, pdfTextLine{Text: "时间范围：" + sanitizePDFText(formatTimeValue(dataset.TimeStart)+" 至 "+formatTimeValue(dataset.TimeEnd)), Size: 12})
	lines = append(lines, pdfTextLine{Text: "", Size: 12})

	lines = append(lines, pdfTextLine{Text: "关键指标", Size: 14})
	lines = append(lines,
		pdfTextLine{Text: fmt.Sprintf("总用电量：%.4f 千瓦时", analysis.TotalKWH), Size: 12},
		pdfTextLine{Text: fmt.Sprintf("日均用电量：%.4f 千瓦时", analysis.DailyAvgKWH), Size: 12},
		pdfTextLine{Text: fmt.Sprintf("最高负荷：%.2f 瓦，时间：%s", analysis.MaxLoadW, formatTimeValue(analysis.MaxLoadTime)), Size: 12},
		pdfTextLine{Text: fmt.Sprintf("最低负荷：%.2f 瓦，时间：%s", analysis.MinLoadW, formatTimeValue(analysis.MinLoadTime)), Size: 12},
		pdfTextLine{Text: fmt.Sprintf("峰谷平占比：峰 %.2f%%，谷 %.2f%%，平 %.2f%%", analysis.PeakRatio*100, analysis.ValleyRatio*100, analysis.FlatRatio*100), Size: 12},
	)
	lines = append(lines, pdfTextLine{Text: "", Size: 12})

	if summary != nil {
		for _, section := range summary.Sections {
			titleValue := sanitizePDFText(section.Title)
			bodyValue := sanitizePDFText(section.Body)
			if titleValue == "" || bodyValue == "" {
				continue
			}
			lines = append(lines, pdfTextLine{Text: titleValue, Size: 14})
			lines = append(lines, pdfTextLine{Text: bodyValue, Size: 12})
			lines = append(lines, pdfTextLine{Text: "", Size: 12})
		}
	}

	lines = append(lines, pdfTextLine{Text: "节能建议", Size: 14})
	recommendations := make([]string, 0, 8)
	if summary != nil {
		for _, item := range summary.Recommendations {
			value := sanitizePDFText(item)
			if value != "" && !containsString(recommendations, value) {
				recommendations = append(recommendations, value)
			}
		}
	}
	if len(recommendations) == 0 {
		for _, advice := range advices {
			value := sanitizePDFText(stringValue(nullableString(advice.Summary)))
			if value != "" && !containsString(recommendations, value) {
				recommendations = append(recommendations, value)
			}
		}
	}
	if len(recommendations) == 0 {
		recommendations = append(recommendations, "建议优先复核峰时段与夜间持续运行设备的负荷安排。")
	}
	for index, item := range recommendations {
		lines = append(lines, pdfTextLine{Text: fmt.Sprintf("%d. %s", index+1, item), Size: 12})
	}

	return lines
}

func renderSimpleChinesePDF(lines []pdfTextLine) ([]byte, error) {
	pages := paginatePDFLines(lines)
	if len(pages) == 0 {
		return nil, fmt.Errorf("pdf 内容为空")
	}

	var output bytes.Buffer
	offsets := []int{0}
	writeObject := func(number int, body string) {
		offsets = append(offsets, output.Len())
		_, _ = fmt.Fprintf(&output, "%d 0 obj\n%s\nendobj\n", number, body)
	}

	output.WriteString("%PDF-1.4\n%\xE2\xE3\xCF\xD3\n")

	fontObject := 3 + len(pages)*2
	descendantFontObject := fontObject + 1

	writeObject(1, "<< /Type /Catalog /Pages 2 0 R >>")

	kids := make([]string, 0, len(pages))
	for index := range pages {
		pageObject := 3 + index*2
		kids = append(kids, fmt.Sprintf("%d 0 R", pageObject))
	}
	writeObject(2, fmt.Sprintf("<< /Type /Pages /Kids [%s] /Count %d >>", strings.Join(kids, " "), len(pages)))

	for index, page := range pages {
		pageObject := 3 + index*2
		contentObject := pageObject + 1
		writeObject(pageObject, fmt.Sprintf("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>", fontObject, contentObject))

		stream := buildPDFContentStream(page)
		writeObject(contentObject, fmt.Sprintf("<< /Length %d >>\nstream\n%s\nendstream", len(stream), stream))
	}

	writeObject(fontObject, fmt.Sprintf("<< /Type /Font /Subtype /Type0 /BaseFont /STSong-Light /Encoding /UniGB-UCS2-H /DescendantFonts [%d 0 R] >>", descendantFontObject))
	writeObject(descendantFontObject, "<< /Type /Font /Subtype /CIDFontType0 /BaseFont /STSong-Light /CIDSystemInfo << /Registry (Adobe) /Ordering (GB1) /Supplement 4 >> /DW 1000 >>")

	xrefOffset := output.Len()
	_, _ = fmt.Fprintf(&output, "xref\n0 %d\n", len(offsets))
	output.WriteString("0000000000 65535 f \n")
	for index := 1; index < len(offsets); index++ {
		_, _ = fmt.Fprintf(&output, "%010d 00000 n \n", offsets[index])
	}

	_, _ = fmt.Fprintf(&output, "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF", len(offsets), xrefOffset)
	return output.Bytes(), nil
}

func paginatePDFLines(lines []pdfTextLine) [][]pdfTextLine {
	const (
		pageHeight    = 842.0
		topMargin     = 52.0
		bottomMargin  = 52.0
		lineSpacing   = 6.0
		bodyMaxUnits  = 68
		titleMaxUnits = 52
	)

	pages := make([][]pdfTextLine, 0, 1)
	currentPage := make([]pdfTextLine, 0, 32)
	remainingHeight := pageHeight - topMargin - bottomMargin

	appendLine := func(line pdfTextLine) {
		lineHeight := line.Size + lineSpacing
		if len(currentPage) > 0 && remainingHeight < lineHeight {
			pages = append(pages, currentPage)
			currentPage = make([]pdfTextLine, 0, 32)
			remainingHeight = pageHeight - topMargin - bottomMargin
		}
		currentPage = append(currentPage, line)
		remainingHeight -= lineHeight
	}

	for _, line := range lines {
		if strings.TrimSpace(line.Text) == "" {
			appendLine(pdfTextLine{Text: "", Size: line.Size})
			continue
		}

		maxUnits := bodyMaxUnits
		if line.Size >= 14 {
			maxUnits = titleMaxUnits
		}
		for _, segment := range wrapPDFText(line.Text, maxUnits) {
			appendLine(pdfTextLine{Text: segment, Size: line.Size})
		}
	}

	if len(currentPage) > 0 {
		pages = append(pages, currentPage)
	}
	return pages
}

func buildPDFContentStream(lines []pdfTextLine) string {
	const (
		startX = 48.0
		startY = 790.0
	)

	var builder strings.Builder
	y := startY
	for _, line := range lines {
		size := line.Size
		if size <= 0 {
			size = 12
		}
		if strings.TrimSpace(line.Text) != "" {
			builder.WriteString("BT\n")
			builder.WriteString(fmt.Sprintf("/F1 %.2f Tf\n", size))
			builder.WriteString(fmt.Sprintf("1 0 0 1 %.2f %.2f Tm\n", startX, y))
			builder.WriteString(fmt.Sprintf("<%s> Tj\n", encodePDFText(sanitizePDFText(line.Text))))
			builder.WriteString("ET\n")
		}
		y -= size + 6
	}
	return builder.String()
}

func wrapPDFText(text string, maxUnits int) []string {
	text = sanitizePDFText(text)
	if strings.TrimSpace(text) == "" {
		return []string{""}
	}
	if maxUnits <= 0 {
		return []string{text}
	}

	tokens := tokenizePDFText(text)
	segments := make([]string, 0, utf8.RuneCountInString(text)/maxUnits+1)
	var builder strings.Builder
	currentUnits := 0

	flush := func() {
		value := strings.TrimSpace(builder.String())
		if value != "" {
			segments = append(segments, value)
		}
		builder.Reset()
		currentUnits = 0
	}

	for _, token := range tokens {
		if token == "\n" {
			flush()
			continue
		}
		units := textDisplayUnits(token)
		if currentUnits > 0 && currentUnits+units > maxUnits {
			flush()
		}
		if currentUnits == 0 {
			token = strings.TrimLeftFunc(token, unicode.IsSpace)
			units = textDisplayUnits(token)
		}
		if token == "" {
			continue
		}
		builder.WriteString(token)
		currentUnits += units
	}
	flush()

	if len(segments) == 0 {
		return []string{text}
	}
	return segments
}

func limitPDFLines(lines []string, maxLines int) []string {
	if maxLines <= 0 || len(lines) <= maxLines {
		return lines
	}
	limited := append([]string{}, lines[:maxLines]...)
	last := strings.TrimSpace(limited[maxLines-1])
	if last == "" {
		last = "..."
	} else if !strings.HasSuffix(last, "...") {
		last = last + "..."
	}
	limited[maxLines-1] = last
	return limited
}

func sanitizePDFText(text string) string {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return ""
	}

	normalized := strings.ReplaceAll(trimmed, "\r\n", "\n")
	normalized = strings.ReplaceAll(normalized, "\r", "\n")

	lines := strings.Split(normalized, "\n")
	cleaned := make([]string, 0, len(lines))
	for _, line := range lines {
		value := pdfTextReplacer.Replace(strings.TrimSpace(line))
		value = stripPDFListMarker(value)
		value = strings.Join(strings.Fields(value), " ")
		value = strings.Trim(value, " ,，;；")
		if value != "" {
			cleaned = append(cleaned, value)
		}
	}
	return strings.Join(cleaned, "\n")
}

func stripPDFListMarker(text string) string {
	value := strings.TrimSpace(text)
	if value == "" {
		return ""
	}

	for {
		runes := []rune(value)
		if len(runes) == 0 {
			return ""
		}

		switch runes[0] {
		case '-', '*', '•', '●', '▪', '▫', '◦', '○', '·':
			value = strings.TrimSpace(string(runes[1:]))
			continue
		}

		index := 0
		for index < len(runes) && unicode.IsDigit(runes[index]) {
			index++
		}
		if index > 0 && index < len(runes) {
			switch runes[index] {
			case '.', '．', ')', '）', '、':
				value = strings.TrimSpace(string(runes[index+1:]))
				continue
			}
		}
		return value
	}
}

func tokenizePDFText(text string) []string {
	tokens := make([]string, 0, len(text))
	runes := []rune(text)
	for index := 0; index < len(runes); {
		current := runes[index]
		if current == '\n' {
			tokens = append(tokens, "\n")
			index++
			continue
		}
		if unicode.IsSpace(current) {
			tokens = append(tokens, " ")
			for index < len(runes) && unicode.IsSpace(runes[index]) && runes[index] != '\n' {
				index++
			}
			continue
		}
		if runeDisplayUnits(current) >= 2 {
			tokens = append(tokens, string(current))
			index++
			continue
		}

		start := index
		for index < len(runes) {
			next := runes[index]
			if next == '\n' || unicode.IsSpace(next) || runeDisplayUnits(next) >= 2 {
				break
			}
			index++
		}
		tokens = append(tokens, string(runes[start:index]))
	}
	return tokens
}

func textDisplayUnits(text string) int {
	total := 0
	for _, r := range text {
		total += runeDisplayUnits(r)
	}
	return total
}

func runeDisplayUnits(r rune) int {
	switch {
	case r <= 0x7F:
		return 1
	case r >= 0x2E80:
		return 2
	default:
		return 1
	}
}

func encodePDFText(value string) string {
	encoded := utf16.Encode([]rune(value))
	buffer := make([]byte, 2, len(encoded)*2+2)
	buffer[0] = 0xFE
	buffer[1] = 0xFF
	for _, item := range encoded {
		buffer = append(buffer, byte(item>>8), byte(item))
	}
	return strings.ToUpper(hex.EncodeToString(buffer))
}

func formatTimeValue(value any) string {
	switch typed := value.(type) {
	case *time.Time:
		if typed == nil {
			return "-"
		}
		return typed.Format("2006-01-02 15:04")
	case time.Time:
		return typed.Format("2006-01-02 15:04")
	case *string:
		if typed == nil || strings.TrimSpace(*typed) == "" {
			return "-"
		}
		return strings.TrimSpace(*typed)
	case string:
		if strings.TrimSpace(typed) == "" {
			return "-"
		}
		return strings.TrimSpace(typed)
	default:
		if text := stringValue(value); text != "" {
			return text
		}
		return "-"
	}
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		trimmed := strings.TrimSpace(value)
		if trimmed != "" {
			return trimmed
		}
	}
	return ""
}

func maxInt(left, right int) int {
	if left >= right {
		return left
	}
	return right
}

func minInt(left, right int) int {
	if left <= right {
		return left
	}
	return right
}

func maxFloat64(left, right float64) float64 {
	if left >= right {
		return left
	}
	return right
}

func conditionalMetricHeight(ok bool, height float64) float64 {
	if ok {
		return height
	}
	return 0
}

func estimateWrappedTextHeight(lineCount int, size float64, gap float64) float64 {
	if lineCount <= 0 {
		return 0
	}
	return float64(lineCount)*size + float64(lineCount-1)*gap
}
