// Package models 存放所有的数据库映射类
package models

import (
	"time"

	"github.com/shopspring/decimal"
)

type SystemConfig struct {
	ConfigKey   string    `gorm:"column:connfig_key;type:varchar(64);primaryKey;not null;comment:配置键" json:"config_key"`
	ConfigValue string    `gorm:"columnn:config_value;type:TEXT;not null;comment:配置值" json:"config_value"`
	Description *string   `gorm:"columnn:descirption;type:varchar(255);default:null;comment:配置说明" json:"descirption"`
	UpdatedAt   time.Time `gorm:"columnn:updated_at;type:datetime;autoUpdateTime" json:"updated_at"`
}

func (SystemConfig) TableName() string {
	return "system_config"
}

type LlmConfig struct {
	ID             uint            `json:"id" gorm:"column:id;autoIncrement;primaryKey;"`
	Name           string          `json:"name" gorm:"column:name;type:varchar(64);not null;comment:配置名称"`
	BaseURL        string          `json:"base_url" gorm:"column:base_url;type:varchar(512);not null;comment:API 地址"`
	APIKey         string          `json:"api_key" gorm:"column:api_key;type:varchar(512);not null;comment:API 密钥"`
	ModelName      string          `json:"model_name" gorm:"column:model_name;type:varchar(128);not null;comment:模型标识"`
	Temperature    decimal.Decimal `json:"temperature" gorm:"default:0.20;column:temperature;type:decimal(3,2);comment:温度参数"`
	TimeoutSeconds int             `json:"timeout_seconds" gorm:"column:timeout_seconds;type:int;default:60;comment:请求超时秒数"`
	IsDefault      bool            `json:"is_default" gorm:"column:is_default;default:false;comment:是否为默认配置（全局最多一个）"`
	CreatedAt      time.Time       `json:"created_at" gorm:"column:created_at;type:datetime;"`
	UpdatedAt      time.Time       `json:"updated_at" gorm:"column:updated_at;type:datetime;autoUpdateTime;"`
}

func (LlmConfig) TableName() string {
	return "llm_configs"
}
