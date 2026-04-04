package domain

import "time"

type EnergyAdviceRecord struct {
	ID               uint64     `gorm:"column:id;primaryKey;autoIncrement"`
	DatasetID        uint64     `gorm:"column:dataset_id;not null"`
	ClassificationID *uint64    `gorm:"column:classification_id"`
	AdviceType       string     `gorm:"column:advice_type;type:enum('rule','llm');not null"`
	ContentPath      string     `gorm:"column:content_path;type:varchar(512);not null"`
	Summary          *string    `gorm:"column:summary;type:varchar(512)"`
	CreatedAt        *time.Time `gorm:"column:created_at;type:datetime"`
}

func (EnergyAdviceRecord) TableName() string {
	return "energy_advices"
}

type AdviceContent struct {
	Items []AdviceItem `json:"items"`
}

type AdviceItem struct {
	Reason string `json:"reason"`
	Action string `json:"action"`
}
