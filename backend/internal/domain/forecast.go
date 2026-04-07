package domain

import "time"

type ForecastResultRecord struct {
	ID            uint64     `gorm:"column:id;primaryKey;autoIncrement"`
	DatasetID     uint64     `gorm:"column:dataset_id;not null"`
	ModelType     string     `gorm:"column:model_type;type:enum('lstm','transformer');not null"`
	ForecastStart time.Time  `gorm:"column:forecast_start;type:datetime;not null"`
	ForecastEnd   time.Time  `gorm:"column:forecast_end;type:datetime;not null"`
	Granularity   string     `gorm:"column:granularity;type:enum('15min','hourly','daily');not null"`
	Summary       []byte     `gorm:"column:summary;type:json"`
	DetailPath    string     `gorm:"column:detail_path;type:varchar(512);not null"`
	Metrics       []byte     `gorm:"column:metrics;type:json"`
	CreatedAt     *time.Time `gorm:"column:created_at;type:datetime"`
}

func (ForecastResultRecord) TableName() string {
	return "forecast_results"
}

type ForecastSummary struct {
	ForecastStart        time.Time `json:"forecast_start"`
	ForecastEnd          time.Time `json:"forecast_end"`
	Granularity          string    `json:"granularity"`
	PredictedAvgLoadW    float64   `json:"predicted_avg_load_w"`
	PredictedPeakLoadW   float64   `json:"predicted_peak_load_w"`
	ForecastPeakPeriods  []string  `json:"forecast_peak_periods"`
	PredictedPeakRatio   float64   `json:"predicted_peak_ratio"`
	PredictedValleyRatio float64   `json:"predicted_valley_ratio"`
	PredictedFlatRatio   float64   `json:"predicted_flat_ratio"`
	RiskFlags            []string  `json:"risk_flags"`
}

type ForecastSeriesPoint struct {
	Timestamp time.Time `json:"timestamp"`
	Predicted float64   `json:"predicted"`
}

type ForecastDetail struct {
	Forecast map[string]any        `json:"forecast"`
	Series   []ForecastSeriesPoint `json:"series"`
}
