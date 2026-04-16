package domain

import "time"

type ClassificationResultRecord struct {
	ID             uint64     `gorm:"column:id;primaryKey;autoIncrement"`
	DatasetID      uint64     `gorm:"column:dataset_id;not null"`
	ModelType      string     `gorm:"column:model_type;type:enum('xgboost');not null"`
	PredictedLabel string     `gorm:"column:predicted_label;type:varchar(32);not null"`
	Confidence     float64    `gorm:"column:confidence;type:decimal(5,4)"`
	Probabilities  []byte     `gorm:"column:probabilities;type:json"`
	Explanation    *string    `gorm:"column:explanation;type:text"`
	WindowStart    *time.Time `gorm:"column:window_start;type:datetime"`
	WindowEnd      *time.Time `gorm:"column:window_end;type:datetime"`
	CreatedAt      *time.Time `gorm:"column:created_at;type:datetime"`
}

func (ClassificationResultRecord) TableName() string {
	return "classification_results"
}

type ClassificationResult struct {
	ID             uint64             `json:"id"`
	DatasetID      uint64             `json:"dataset_id"`
	ModelType      string             `json:"model_type"`
	PredictedLabel string             `json:"predicted_label"`
	Confidence     float64            `json:"confidence"`
	Probabilities  map[string]float64 `json:"probabilities"`
	Explanation    *string            `json:"explanation"`
	WindowStart    *time.Time         `json:"window_start"`
	WindowEnd      *time.Time         `json:"window_end"`
	CreatedAt      *time.Time         `json:"created_at"`
}
