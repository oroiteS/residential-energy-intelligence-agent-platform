package liveapp

type DataPoint struct {
	Timestamp            string  `json:"timestamp"`
	Date                 string  `json:"date"`
	SlotIndex            int     `json:"slotIndex"`
	TimeLabel            string  `json:"timeLabel"`
	Aggregate            float64 `json:"aggregate"`
	ActiveApplianceCount float64 `json:"activeApplianceCount"`
	BurstEventCount      float64 `json:"burstEventCount"`
	IsWeekend            bool    `json:"isWeekend"`
}

type SourceInfo struct {
	HouseID  string `json:"houseId"`
	Dataset  string `json:"dataset"`
	DataFile string `json:"dataFile"`
	Days     int    `json:"days"`
}

type LiveMetrics struct {
	CurrentPowerW      float64 `json:"currentPowerW"`
	TodayCumulativeKWH float64 `json:"todayCumulativeKWH"`
	ObservedSlots      int     `json:"observedSlots"`
	TotalSlots         int     `json:"totalSlots"`
	ProgressPercent    float64 `json:"progressPercent"`
	TodayPeakLoadW     float64 `json:"todayPeakLoadW"`
	TodayBaseLoadW     float64 `json:"todayBaseLoadW"`
}

type DayClassification struct {
	Date        string  `json:"date"`
	Label       string  `json:"label"`
	DayMean     float64 `json:"dayMean"`
	NightMean   float64 `json:"nightMean"`
	FullMean    float64 `json:"fullMean"`
	ModelType   string  `json:"modelType"`
	Explanation string  `json:"explanation"`
	Error       string  `json:"error,omitempty"`
}

type ForecastPoint struct {
	SlotIndex int     `json:"slotIndex"`
	TimeLabel string  `json:"timeLabel"`
	Predicted float64 `json:"predicted"`
}

type DayForecast struct {
	Date            string          `json:"date"`
	Points          []ForecastPoint `json:"points"`
	AvgLoadW        float64         `json:"avgLoadW"`
	PeakLoadW       float64         `json:"peakLoadW"`
	PeakPeriods     []string        `json:"peakPeriods"`
	PeakRatio       float64         `json:"peakRatio"`
	ValleyRatio     float64         `json:"valleyRatio"`
	FlatRatio       float64         `json:"flatRatio"`
	RiskFlags       []string        `json:"riskFlags"`
	Explanation     string          `json:"explanation"`
	HistoryDaysUsed int             `json:"historyDaysUsed"`
	ModelType       string          `json:"modelType"`
	Error           string          `json:"error,omitempty"`
}

type Snapshot struct {
	Running              bool              `json:"running"`
	Source               SourceInfo        `json:"source"`
	VirtualTime          string            `json:"virtualTime"`
	VirtualDate          string            `json:"virtualDate"`
	VirtualClockLabel    string            `json:"virtualClockLabel"`
	CurrentPoint         DataPoint         `json:"currentPoint"`
	TodayPoints          []DataPoint       `json:"todayPoints"`
	Metrics              LiveMetrics       `json:"metrics"`
	LatestClassification DayClassification `json:"latestClassification"`
	TodayForecast        DayForecast       `json:"todayForecast"`
	NextDayForecast      DayForecast       `json:"nextDayForecast"`
	ActiveForecast       DayForecast       `json:"activeForecast"`
	WeekLoop             int               `json:"weekLoop"`
}

type ChatRequest struct {
	Question string `json:"question"`
}

type ChatMissingInformation struct {
	Key      string `json:"key"`
	Question string `json:"question"`
	Reason   string `json:"reason"`
}

type ChatResponse struct {
	Answer             string                   `json:"answer"`
	CreatedAt          string                   `json:"createdAt"`
	Actions            []string                 `json:"actions,omitempty"`
	Intent             string                   `json:"intent,omitempty"`
	ConfidenceLevel    string                   `json:"confidenceLevel,omitempty"`
	MissingInformation []ChatMissingInformation `json:"missingInformation,omitempty"`
	Degraded           bool                     `json:"degraded"`
	ErrorReason        string                   `json:"errorReason,omitempty"`
}
