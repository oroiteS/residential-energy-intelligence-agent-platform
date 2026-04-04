package domain

import "time"

type LLMConfigRecord struct {
	ID             uint64     `gorm:"column:id;primaryKey;autoIncrement" json:"id"`
	Name           string     `gorm:"column:name;type:varchar(64);not null" json:"name"`
	BaseURL        string     `gorm:"column:base_url;type:varchar(512);not null" json:"base_url"`
	APIKey         string     `gorm:"column:api_key;type:varchar(512);not null" json:"api_key"`
	ModelName      string     `gorm:"column:model_name;type:varchar(128);not null" json:"model_name"`
	Temperature    float64    `gorm:"column:temperature;type:decimal(3,2);not null" json:"temperature"`
	TimeoutSeconds int        `gorm:"column:timeout_seconds;type:int unsigned;not null" json:"timeout_seconds"`
	IsDefault      bool       `gorm:"column:is_default;type:tinyint(1);not null" json:"is_default"`
	CreatedAt      *time.Time `gorm:"column:created_at;type:datetime" json:"created_at"`
	UpdatedAt      *time.Time `gorm:"column:updated_at;type:datetime" json:"updated_at"`
}

func (LLMConfigRecord) TableName() string {
	return "llm_configs"
}

type LLMConfigSummary struct {
	ID             uint64     `json:"id"`
	Name           string     `json:"name"`
	BaseURL        string     `json:"base_url"`
	ModelName      string     `json:"model_name"`
	Temperature    float64    `json:"temperature"`
	TimeoutSeconds int        `json:"timeout_seconds"`
	IsDefault      bool       `json:"is_default"`
	CreatedAt      *time.Time `json:"created_at"`
	UpdatedAt      *time.Time `json:"updated_at"`
}

type LLMConfigPayload struct {
	Name           string  `json:"name"`
	BaseURL        string  `json:"base_url"`
	APIKey         string  `json:"api_key"`
	ModelName      string  `json:"model_name"`
	Temperature    float64 `json:"temperature"`
	TimeoutSeconds int     `json:"timeout_seconds"`
	IsDefault      bool    `json:"is_default"`
}
