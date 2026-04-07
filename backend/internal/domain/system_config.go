package domain

import "time"

type SystemConfigRecord struct {
	ConfigKey   string     `gorm:"column:config_key;type:varchar(64);primaryKey;not null" json:"config_key"`
	ConfigValue string     `gorm:"column:config_value;type:text;not null" json:"config_value"`
	Description *string    `gorm:"column:description;type:varchar(255)" json:"description"`
	UpdatedAt   *time.Time `gorm:"column:updated_at;type:datetime" json:"updated_at"`
}

func (SystemConfigRecord) TableName() string {
	return "system_config"
}

type PeakValleyConfig struct {
	Peak   []string `json:"peak"`
	Valley []string `json:"valley"`
}

type ModelHistoryWindowConfig struct {
	ClassificationDays  int `json:"classification_days"`
	ForecastHistoryDays int `json:"forecast_history_days"`
}

type SystemRuntimeConfig struct {
	PeakValleyConfig           PeakValleyConfig         `json:"peak_valley_config"`
	ModelHistoryWindowConfig   ModelHistoryWindowConfig `json:"model_history_window_config"`
	EnergyAdvicePromptTemplate string                   `json:"energy_advice_prompt_template"`
	DataUploadDir              string                   `json:"data_upload_dir"`
	ReportOutputDir            string                   `json:"report_output_dir"`
}
