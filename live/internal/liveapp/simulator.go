package liveapp

import (
	"context"
	"encoding/csv"
	"errors"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	dayStartSlot = 32
	dayEndSlot   = 72
	daySlots     = 96
)

type daySeries struct {
	Date      string
	Points    []DataPoint
	DayMean   float64
	NightMean float64
	FullMean  float64
}

type Simulator struct {
	mu                   sync.RWMutex
	source               SourceInfo
	days                 []daySeries
	dayIndex             int
	slotIndex            int
	weekLoop             int
	running              bool
	tickInterval         time.Duration
	latestClassification DayClassification
	activeForecast       DayForecast
	subscribers          map[chan Snapshot]struct{}
	serviceClient        *ServiceClient
	chatHistory          []chatHistoryItem
}

func NewSimulator(csvPath string, tickInterval time.Duration, serviceClient *ServiceClient) (*Simulator, error) {
	if tickInterval <= 0 {
		tickInterval = time.Second
	}

	source, days, err := loadWeekData(csvPath)
	if err != nil {
		return nil, err
	}
	if len(days) == 0 {
		return nil, errors.New("live 数据为空")
	}

	simulator := &Simulator{
		source:        source,
		days:          days,
		running:       true,
		tickInterval:  tickInterval,
		subscribers:   make(map[chan Snapshot]struct{}),
		serviceClient: serviceClient,
		chatHistory: []chatHistoryItem{
			{
				Role:    "system",
				Content: "你正在分析 live 模块中的单个虚拟住户实时用电态势。",
			},
		},
	}
	simulator.latestClassification = simulator.refreshClassification((len(days) - 1 + len(days)) % len(days))
	simulator.activeForecast = simulator.refreshForecast(0)
	return simulator, nil
}

func (s *Simulator) Start() {
	ticker := time.NewTicker(s.tickInterval)
	go func() {
		defer ticker.Stop()
		for range ticker.C {
			s.Tick()
		}
	}()
}

func (s *Simulator) Tick() Snapshot {
	s.mu.Lock()
	currentDay := s.dayIndex
	currentSlot := s.slotIndex + 1
	if currentSlot >= len(s.days[currentDay].Points) {
		s.latestClassification = s.refreshClassification(currentDay)
		s.dayIndex = (s.dayIndex + 1) % len(s.days)
		if s.dayIndex == 0 {
			s.weekLoop++
		}
		s.slotIndex = 0
		s.activeForecast = s.refreshForecast(s.dayIndex)
	} else {
		s.slotIndex = currentSlot
	}

	snapshot := s.snapshotLocked()
	s.broadcastLocked(snapshot)
	s.mu.Unlock()
	return snapshot
}

func (s *Simulator) Snapshot() Snapshot {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.snapshotLocked()
}

func (s *Simulator) Subscribe() (<-chan Snapshot, func()) {
	ch := make(chan Snapshot, 1)

	s.mu.Lock()
	s.subscribers[ch] = struct{}{}
	initial := s.snapshotLocked()
	s.mu.Unlock()

	ch <- initial

	cancel := func() {
		s.mu.Lock()
		if _, exists := s.subscribers[ch]; exists {
			delete(s.subscribers, ch)
			close(ch)
		}
		s.mu.Unlock()
	}

	return ch, cancel
}

func (s *Simulator) Answer(question string) ChatResponse {
	trimmed := strings.TrimSpace(question)
	if trimmed == "" {
		trimmed = "请总结当前实时用电状态，并说明下一日预测重点。"
	}

	s.mu.RLock()
	snapshot := s.snapshotLocked()
	history := append([]chatHistoryItem(nil), s.chatHistory...)
	s.mu.RUnlock()

	if s.serviceClient == nil {
		return ChatResponse{
			Answer:      "智能体服务未配置，当前无法完成直接提问。",
			CreatedAt:   time.Now().Format(time.RFC3339),
			Degraded:    true,
			ErrorReason: "LIVE_AGENT_SERVICE_BASE_URL_MISSING",
		}
	}

	contextPayload := s.buildAgentContext(snapshot)
	response, err := s.serviceClient.Ask(context.Background(), trimmed, history, contextPayload)
	if err != nil {
		return ChatResponse{
			Answer:      fmt.Sprintf("智能体服务调用失败：%v", err),
			CreatedAt:   time.Now().Format(time.RFC3339),
			Degraded:    true,
			ErrorReason: "LIVE_AGENT_REQUEST_FAILED",
		}
	}

	result := ChatResponse{
		Answer:      strings.TrimSpace(response.Answer),
		CreatedAt:   time.Now().Format(time.RFC3339),
		Degraded:    response.Degraded,
		ErrorReason: strings.TrimSpace(response.ErrorReason),
	}
	if result.Answer == "" {
		result.Answer = "智能体返回了空结果，当前无法给出有效回答。"
		result.Degraded = true
		if result.ErrorReason == "" {
			result.ErrorReason = "LIVE_AGENT_EMPTY_ANSWER"
		}
	}

	s.mu.Lock()
	s.chatHistory = append(s.chatHistory,
		chatHistoryItem{Role: "user", Content: trimmed},
		chatHistoryItem{Role: "assistant", Content: result.Answer},
	)
	if len(s.chatHistory) > 13 {
		s.chatHistory = append([]chatHistoryItem{s.chatHistory[0]}, s.chatHistory[len(s.chatHistory)-12:]...)
	}
	s.mu.Unlock()

	return result
}

func (s *Simulator) snapshotLocked() Snapshot {
	day := s.days[s.dayIndex]
	current := day.Points[s.slotIndex]
	todayPoints := clonePoints(day.Points[:s.slotIndex+1])

	return Snapshot{
		Running:              s.running,
		Source:               s.source,
		VirtualTime:          current.Timestamp,
		VirtualDate:          day.Date,
		VirtualClockLabel:    current.TimeLabel,
		CurrentPoint:         current,
		TodayPoints:          todayPoints,
		Metrics:              buildMetrics(todayPoints),
		LatestClassification: cloneClassification(s.latestClassification),
		ActiveForecast:       cloneForecast(s.activeForecast),
		WeekLoop:             s.weekLoop,
	}
}

func (s *Simulator) broadcastLocked(snapshot Snapshot) {
	for ch := range s.subscribers {
		select {
		case ch <- snapshot:
		default:
			select {
			case <-ch:
			default:
			}
			select {
			case ch <- snapshot:
			default:
			}
		}
	}
}

func (s *Simulator) refreshClassification(dayIndex int) DayClassification {
	series := s.days[dayIndex]

	if s.serviceClient == nil {
		return DayClassification{
			Date:        series.Date,
			DayMean:     roundFloat(series.DayMean, 2),
			NightMean:   roundFloat(series.NightMean, 2),
			FullMean:    roundFloat(series.FullMean, 2),
			ModelType:   "tcn",
			Explanation: "模型服务未配置，无法生成最近完整日分类结果。",
			Error:       "LIVE_MODEL_SERVICE_BASE_URL_MISSING",
		}
	}

	response, err := s.serviceClient.PredictClassification(context.Background(), series)
	if err != nil {
		return DayClassification{
			Date:        series.Date,
			DayMean:     roundFloat(series.DayMean, 2),
			NightMean:   roundFloat(series.NightMean, 2),
			FullMean:    roundFloat(series.FullMean, 2),
			ModelType:   "tcn",
			Explanation: "分类模型调用失败，当前未刷新出新的分类结果。",
			Error:       err.Error(),
		}
	}

	return DayClassification{
		Date:      firstNonEmpty(strings.TrimSpace(response.Date), series.Date),
		Label:     strings.TrimSpace(response.PredictedLabel),
		DayMean:   roundFloat(series.DayMean, 2),
		NightMean: roundFloat(series.NightMean, 2),
		FullMean:  roundFloat(series.FullMean, 2),
		ModelType: firstNonEmpty(strings.TrimSpace(response.ModelType), "tcn"),
		Explanation: buildClassificationExplanation(
			strings.TrimSpace(response.PredictedLabel),
			series.DayMean,
			series.NightMean,
			series.FullMean,
		),
	}
}

func (s *Simulator) refreshForecast(targetDayIndex int) DayForecast {
	targetDay := s.days[targetDayIndex]
	historyPoints := s.buildForecastHistory(targetDayIndex)

	if s.serviceClient == nil {
		return DayForecast{
			Date:            targetDay.Date,
			Explanation:     "模型服务未配置，无法生成下一日预测。",
			HistoryDaysUsed: 3,
			ModelType:       "transformer",
			Error:           "LIVE_MODEL_SERVICE_BASE_URL_MISSING",
		}
	}
	if len(historyPoints) != daySlots*3 {
		return DayForecast{
			Date:            targetDay.Date,
			Explanation:     "历史窗口不足，无法为下一日生成预测。",
			HistoryDaysUsed: len(historyPoints) / daySlots,
			ModelType:       s.serviceClient.forecastModelType,
			Error:           "INSUFFICIENT_FORECAST_HISTORY",
		}
	}

	response, err := s.serviceClient.Forecast(context.Background(), historyPoints, targetDay)
	if err != nil {
		return DayForecast{
			Date:            targetDay.Date,
			Explanation:     "预测模型调用失败，当前未刷新出新的预测结果。",
			HistoryDaysUsed: 3,
			ModelType:       s.serviceClient.forecastModelType,
			Error:           err.Error(),
		}
	}
	if len(response.Predictions) != daySlots {
		return DayForecast{
			Date:            targetDay.Date,
			Explanation:     "预测模型返回的数据点数量不正确。",
			HistoryDaysUsed: 3,
			ModelType:       firstNonEmpty(strings.TrimSpace(response.ModelType), s.serviceClient.forecastModelType),
			Error:           fmt.Sprintf("expected=%d actual=%d", daySlots, len(response.Predictions)),
		}
	}

	points := make([]ForecastPoint, 0, daySlots)
	total := 0.0
	peak := 0.0
	peakEnergy := 0.0
	valleyEnergy := 0.0
	peakSlotSet := make(map[string]struct{})
	peakPeriods := make([]string, 0, 4)

	for index, value := range response.Predictions {
		predictedValue := roundFloat(value, 2)
		total += predictedValue
		if predictedValue > peak {
			peak = predictedValue
		}
		if isPeakSlot(index) {
			peakEnergy += predictedValue
		} else if isValleySlot(index) {
			valleyEnergy += predictedValue
		}
		points = append(points, ForecastPoint{
			SlotIndex: index,
			TimeLabel: targetDay.Points[index].TimeLabel,
			Predicted: predictedValue,
		})
	}

	for _, point := range points {
		if peak > 0 && point.Predicted >= peak*0.88 {
			if _, exists := peakSlotSet[point.TimeLabel]; !exists {
				peakSlotSet[point.TimeLabel] = struct{}{}
				peakPeriods = append(peakPeriods, point.TimeLabel)
			}
		}
	}

	sort.Strings(peakPeriods)
	avgLoad := safeDivide(total, float64(daySlots))
	flatEnergy := max(total-peakEnergy-valleyEnergy, 0)
	riskFlags := buildRiskFlags(avgLoad, peak, safeDivide(peakEnergy, total), safeDivide(valleyEnergy, total))

	return DayForecast{
		Date:            targetDay.Date,
		Points:          points,
		AvgLoadW:        roundFloat(avgLoad, 2),
		PeakLoadW:       roundFloat(peak, 2),
		PeakPeriods:     peakPeriods,
		PeakRatio:       roundFloat(safeDivide(peakEnergy, total), 4),
		ValleyRatio:     roundFloat(safeDivide(valleyEnergy, total), 4),
		FlatRatio:       roundFloat(safeDivide(flatEnergy, total), 4),
		RiskFlags:       riskFlags,
		Explanation:     fmt.Sprintf("基于最近 3 个完整虚拟日，通过 %s 模型生成下一日预测。", firstNonEmpty(strings.TrimSpace(response.ModelType), s.serviceClient.forecastModelType)),
		HistoryDaysUsed: 3,
		ModelType:       firstNonEmpty(strings.TrimSpace(response.ModelType), s.serviceClient.forecastModelType),
	}
}

func (s *Simulator) buildForecastHistory(targetDayIndex int) []DataPoint {
	historyPoints := make([]DataPoint, 0, daySlots*3)
	for offset := 3; offset >= 1; offset-- {
		index := (targetDayIndex - offset + len(s.days)) % len(s.days)
		historyPoints = append(historyPoints, clonePoints(s.days[index].Points)...)
	}
	return historyPoints
}

func (s *Simulator) buildAgentContext(snapshot Snapshot) map[string]any {
	classification := snapshot.LatestClassification
	forecast := snapshot.ActiveForecast

	return map[string]any{
		"dataset": map[string]any{
			"id":           defaultLiveDatasetID,
			"name":         "live_sample",
			"description":  "live 模块实时演示数据集",
			"household_id": snapshot.Source.HouseID,
			"source":       snapshot.Source.Dataset,
			"days":         snapshot.Source.Days,
			"loop_index":   snapshot.WeekLoop,
		},
		"analysis_summary": map[string]any{
			"current_power_w":      roundFloat(snapshot.Metrics.CurrentPowerW, 2),
			"today_cumulative_kwh": roundFloat(snapshot.Metrics.TodayCumulativeKWH, 4),
			"today_peak_load_w":    roundFloat(snapshot.Metrics.TodayPeakLoadW, 2),
			"today_base_load_w":    roundFloat(snapshot.Metrics.TodayBaseLoadW, 2),
			"progress_percent":     roundFloat(snapshot.Metrics.ProgressPercent, 2),
			"virtual_time":         snapshot.VirtualTime,
			"observed_slots":       snapshot.Metrics.ObservedSlots,
			"total_slots":          snapshot.Metrics.TotalSlots,
		},
		"recent_history_summary": buildLiveRecentHistorySummary(snapshot.TodayPoints),
		"classification_result": map[string]any{
			"date":        classification.Date,
			"label":       classification.Label,
			"label_name":  localizeLabel(classification.Label),
			"model_type":  classification.ModelType,
			"day_mean":    classification.DayMean,
			"night_mean":  classification.NightMean,
			"full_mean":   classification.FullMean,
			"explanation": classification.Explanation,
			"error":       classification.Error,
		},
		"forecast_summary": map[string]any{
			"date":              forecast.Date,
			"model_type":        forecast.ModelType,
			"avg_load_w":        forecast.AvgLoadW,
			"peak_load_w":       forecast.PeakLoadW,
			"peak_periods":      append([]string(nil), forecast.PeakPeriods...),
			"peak_ratio":        forecast.PeakRatio,
			"valley_ratio":      forecast.ValleyRatio,
			"flat_ratio":        forecast.FlatRatio,
			"risk_flags":        append([]string(nil), forecast.RiskFlags...),
			"history_days_used": forecast.HistoryDaysUsed,
			"explanation":       forecast.Explanation,
			"error":             forecast.Error,
		},
		"rule_advices": buildLiveRuleAdvices(classification, forecast),
	}
}

func buildLiveRecentHistorySummary(points []DataPoint) map[string]any {
	if len(points) == 0 {
		return map[string]any{}
	}

	total := 0.0
	maxAggregate := 0.0
	activeTotal := 0.0
	burstTotal := 0.0

	for _, point := range points {
		total += point.Aggregate
		activeTotal += point.ActiveApplianceCount
		burstTotal += point.BurstEventCount
		if point.Aggregate > maxAggregate {
			maxAggregate = point.Aggregate
		}
	}

	return map[string]any{
		"observed_points":            len(points),
		"avg_load_w":                 roundFloat(safeDivide(total, float64(len(points))), 2),
		"max_load_w":                 roundFloat(maxAggregate, 2),
		"avg_active_appliance_count": roundFloat(safeDivide(activeTotal, float64(len(points))), 2),
		"avg_burst_event_count":      roundFloat(safeDivide(burstTotal, float64(len(points))), 2),
		"latest_time":                points[len(points)-1].Timestamp,
	}
}

func buildLiveRuleAdvices(classification DayClassification, forecast DayForecast) []map[string]any {
	items := make([]map[string]any, 0, 4)

	switch classification.Label {
	case "day_high_night_low":
		items = append(items, map[string]any{
			"title":   "白天负荷偏高",
			"content": "适合优先检查白天持续运行设备，避免工作时段叠加高功率电器。",
		})
	case "day_low_night_high":
		items = append(items, map[string]any{
			"title":   "夜间负荷偏高",
			"content": "建议重点核查夜间待机设备或持续运行电器，降低夜间基荷。",
		})
	case "all_day_high":
		items = append(items, map[string]any{
			"title":   "全天负荷偏高",
			"content": "全天负荷都较高，应优先排查长时运行设备与空调、热水等大负荷来源。",
		})
	case "all_day_low":
		items = append(items, map[string]any{
			"title":   "全天负荷平稳",
			"content": "当前整体负荷不高，可继续维持现有用电节奏。",
		})
	}

	for _, flag := range forecast.RiskFlags {
		items = append(items, map[string]any{
			"title":   "预测风险提示",
			"content": flag,
		})
	}

	if len(items) == 0 {
		items = append(items, map[string]any{
			"title":   "暂无建议",
			"content": "当前没有额外规则建议。",
		})
	}
	return items
}

func loadWeekData(csvPath string) (SourceInfo, []daySeries, error) {
	file, err := os.Open(csvPath)
	if err != nil {
		return SourceInfo{}, nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	rows, err := reader.ReadAll()
	if err != nil {
		return SourceInfo{}, nil, err
	}
	if len(rows) < 2 {
		return SourceInfo{}, nil, errors.New("live csv 至少需要一行表头和一行数据")
	}

	headerIndex := make(map[string]int, len(rows[0]))
	for index, name := range rows[0] {
		headerIndex[strings.TrimSpace(name)] = index
	}
	required := []string{
		"house_id",
		"source_dataset",
		"timestamp",
		"date",
		"slot_index",
		"aggregate",
		"active_appliance_count",
		"burst_event_count",
		"is_weekend",
	}
	for _, column := range required {
		if _, exists := headerIndex[column]; !exists {
			return SourceInfo{}, nil, fmt.Errorf("live csv 缺少列: %s", column)
		}
	}

	dayMap := make(map[string][]DataPoint)
	dayOrder := make([]string, 0, 8)
	var houseID string
	var dataset string

	for _, row := range rows[1:] {
		dateValue := row[headerIndex["date"]]
		if _, exists := dayMap[dateValue]; !exists {
			dayOrder = append(dayOrder, dateValue)
		}
		point, buildErr := buildPoint(row, headerIndex)
		if buildErr != nil {
			return SourceInfo{}, nil, buildErr
		}
		houseID = row[headerIndex["house_id"]]
		dataset = row[headerIndex["source_dataset"]]
		dayMap[dateValue] = append(dayMap[dateValue], point)
	}

	days := make([]daySeries, 0, len(dayOrder))
	for _, dateValue := range dayOrder {
		points := dayMap[dateValue]
		sort.Slice(points, func(left, right int) bool {
			return points[left].SlotIndex < points[right].SlotIndex
		})
		if len(points) != daySlots {
			return SourceInfo{}, nil, fmt.Errorf("日期 %s 的点数不是 96，当前为 %d", dateValue, len(points))
		}

		dayMean, nightMean, fullMean := computeMeans(points)
		days = append(days, daySeries{
			Date:      dateValue,
			Points:    points,
			DayMean:   dayMean,
			NightMean: nightMean,
			FullMean:  fullMean,
		})
	}

	source := SourceInfo{
		HouseID: houseID,
		Dataset: dataset,
		Days:    len(days),
	}
	return source, days, nil
}

func buildPoint(row []string, headerIndex map[string]int) (DataPoint, error) {
	timestampValue := row[headerIndex["timestamp"]]
	parsedTime, err := time.ParseInLocation("2006-01-02 15:04:05", timestampValue, time.Local)
	if err != nil {
		return DataPoint{}, fmt.Errorf("解析 timestamp 失败: %w", err)
	}

	slotIndex, err := strconv.Atoi(row[headerIndex["slot_index"]])
	if err != nil {
		return DataPoint{}, fmt.Errorf("解析 slot_index 失败: %w", err)
	}
	aggregate, err := strconv.ParseFloat(row[headerIndex["aggregate"]], 64)
	if err != nil {
		return DataPoint{}, fmt.Errorf("解析 aggregate 失败: %w", err)
	}
	activeCount, err := strconv.ParseFloat(row[headerIndex["active_appliance_count"]], 64)
	if err != nil {
		return DataPoint{}, fmt.Errorf("解析 active_appliance_count 失败: %w", err)
	}
	burstCount, err := strconv.ParseFloat(row[headerIndex["burst_event_count"]], 64)
	if err != nil {
		return DataPoint{}, fmt.Errorf("解析 burst_event_count 失败: %w", err)
	}
	isWeekend := strings.TrimSpace(row[headerIndex["is_weekend"]]) == "1"

	return DataPoint{
		Timestamp:            parsedTime.Format("2006-01-02 15:04:05"),
		Date:                 row[headerIndex["date"]],
		SlotIndex:            slotIndex,
		TimeLabel:            parsedTime.Format("15:04"),
		Aggregate:            roundFloat(aggregate, 2),
		ActiveApplianceCount: roundFloat(activeCount, 2),
		BurstEventCount:      roundFloat(burstCount, 2),
		IsWeekend:            isWeekend,
	}, nil
}

func computeMeans(points []DataPoint) (float64, float64, float64) {
	dayTotal := 0.0
	nightTotal := 0.0
	fullTotal := 0.0
	dayCount := 0
	nightCount := 0

	for _, point := range points {
		fullTotal += point.Aggregate
		if point.SlotIndex >= dayStartSlot && point.SlotIndex < dayEndSlot {
			dayTotal += point.Aggregate
			dayCount++
		} else {
			nightTotal += point.Aggregate
			nightCount++
		}
	}

	return safeDivide(dayTotal, float64(dayCount)), safeDivide(nightTotal, float64(nightCount)), safeDivide(fullTotal, float64(len(points)))
}

func buildClassificationExplanation(label string, dayMean float64, nightMean float64, fullMean float64) string {
	switch label {
	case "day_high_night_low":
		return fmt.Sprintf("白天均值 %.2fW 明显高于夜间均值 %.2fW，属于白天活跃型。", dayMean, nightMean)
	case "day_low_night_high":
		return fmt.Sprintf("夜间均值 %.2fW 明显高于白天均值 %.2fW，存在夜间持续负荷特征。", nightMean, dayMean)
	case "all_day_high":
		return fmt.Sprintf("白天与夜间均处于高负荷区间，全天均值 %.2fW 偏高。", fullMean)
	case "all_day_low":
		return fmt.Sprintf("白天与夜间均处于低负荷区间，全天均值 %.2fW 较低。", fullMean)
	default:
		return fmt.Sprintf("模型返回的分类标签为 %s，全天均值 %.2fW。", label, fullMean)
	}
}

func buildMetrics(points []DataPoint) LiveMetrics {
	current := points[len(points)-1]
	total := 0.0
	peak := 0.0
	base := math.MaxFloat64

	for _, point := range points {
		total += point.Aggregate * 0.25 / 1000
		if point.Aggregate > peak {
			peak = point.Aggregate
		}
		if point.Aggregate < base {
			base = point.Aggregate
		}
	}
	if base == math.MaxFloat64 {
		base = 0
	}

	return LiveMetrics{
		CurrentPowerW:      current.Aggregate,
		TodayCumulativeKWH: roundFloat(total, 4),
		ObservedSlots:      len(points),
		TotalSlots:         daySlots,
		ProgressPercent:    roundFloat(float64(len(points))*100/float64(daySlots), 2),
		TodayPeakLoadW:     roundFloat(peak, 2),
		TodayBaseLoadW:     roundFloat(base, 2),
	}
}

func buildRiskFlags(avgLoad float64, peakLoad float64, peakRatio float64, valleyRatio float64) []string {
	flags := make([]string, 0, 4)
	if peakRatio >= 0.42 {
		flags = append(flags, "晚高峰占比偏高")
	}
	if valleyRatio <= 0.16 {
		flags = append(flags, "夜间低谷利用不足")
	}
	if peakLoad >= avgLoad*1.85 {
		flags = append(flags, "存在明显尖峰冲击")
	}
	if avgLoad >= 900 {
		flags = append(flags, "全天基荷偏高")
	}
	if len(flags) == 0 {
		flags = append(flags, "整体负荷节奏平稳")
	}
	return flags
}

func isPeakSlot(slot int) bool {
	return slot >= 72 && slot < 88
}

func isValleySlot(slot int) bool {
	return slot < 24
}

func safeDivide(numerator float64, denominator float64) float64 {
	if denominator == 0 {
		return 0
	}
	return numerator / denominator
}

func roundFloat(value float64, precision int) float64 {
	factor := math.Pow10(precision)
	return math.Round(value*factor) / factor
}

func clonePoints(points []DataPoint) []DataPoint {
	return append([]DataPoint(nil), points...)
}

func cloneClassification(value DayClassification) DayClassification {
	return value
}

func cloneForecast(value DayForecast) DayForecast {
	value.Points = append([]ForecastPoint(nil), value.Points...)
	value.PeakPeriods = append([]string(nil), value.PeakPeriods...)
	value.RiskFlags = append([]string(nil), value.RiskFlags...)
	return value
}

func localizeLabel(label string) string {
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

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func max(left float64, right float64) float64 {
	if left > right {
		return left
	}
	return right
}
