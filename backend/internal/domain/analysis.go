package domain

import "time"

type AnalysisResultRecord struct {
	ID          uint64     `gorm:"column:id;primaryKey;autoIncrement"`
	DatasetID   uint64     `gorm:"column:dataset_id;not null"`
	TotalKWH    float64    `gorm:"column:total_kwh;type:decimal(12,4)"`
	DailyAvgKWH float64    `gorm:"column:daily_avg_kwh;type:decimal(10,4)"`
	MaxLoadW    float64    `gorm:"column:max_load_w;type:decimal(10,2)"`
	MaxLoadTime *time.Time `gorm:"column:max_load_time;type:datetime"`
	MinLoadW    float64    `gorm:"column:min_load_w;type:decimal(10,2)"`
	MinLoadTime *time.Time `gorm:"column:min_load_time;type:datetime"`
	PeakKWH     float64    `gorm:"column:peak_kwh;type:decimal(12,4)"`
	ValleyKWH   float64    `gorm:"column:valley_kwh;type:decimal(12,4)"`
	FlatKWH     float64    `gorm:"column:flat_kwh;type:decimal(12,4)"`
	PeakRatio   float64    `gorm:"column:peak_ratio;type:decimal(5,4)"`
	ValleyRatio float64    `gorm:"column:valley_ratio;type:decimal(5,4)"`
	FlatRatio   float64    `gorm:"column:flat_ratio;type:decimal(5,4)"`
	DetailPath  *string    `gorm:"column:detail_path;type:varchar(512)"`
	CreatedAt   *time.Time `gorm:"column:created_at;type:datetime"`
}

func (AnalysisResultRecord) TableName() string {
	return "analysis_results"
}

type AnalysisSummary struct {
	TotalKWH    float64    `json:"total_kwh"`
	DailyAvgKWH float64    `json:"daily_avg_kwh"`
	MaxLoadW    float64    `json:"max_load_w"`
	MaxLoadTime *time.Time `json:"max_load_time"`
	MinLoadW    float64    `json:"min_load_w"`
	MinLoadTime *time.Time `json:"min_load_time"`
	PeakKWH     float64    `json:"peak_kwh"`
	ValleyKWH   float64    `json:"valley_kwh"`
	FlatKWH     float64    `json:"flat_kwh"`
	PeakRatio   float64    `json:"peak_ratio"`
	ValleyRatio float64    `json:"valley_ratio"`
	FlatRatio   float64    `json:"flat_ratio"`
}

type AnalysisCharts struct {
	DailyTrend      []map[string]any `json:"daily_trend"`
	WeeklyTrend     []map[string]any `json:"weekly_trend"`
	TypicalDayCurve []map[string]any `json:"typical_day_curve"`
	PeakValleyPie   []map[string]any `json:"peak_valley_pie"`
}

type AnalysisDetail struct {
	Summary           AnalysisSummary   `json:"summary"`
	PeakValleyConfig  PeakValleyConfig  `json:"peak_valley_config"`
	Charts            AnalysisCharts    `json:"charts"`
	DetailPath        string            `json:"detail_path"`
	UpdatedAt         *time.Time        `json:"updated_at"`
}
