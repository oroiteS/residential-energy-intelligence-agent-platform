package bootstrap

import (
	"go.uber.org/zap"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"

	"residential-energy-intelligence-agent-platform/internal/config"
	"residential-energy-intelligence-agent-platform/pkg/loggerx"
)

func NewDatabase(cfg *config.AppConfig, logger *zap.Logger) (*gorm.DB, error) {
	if cfg.MySQLDSN == "" {
		logger.Warn("未配置 MYSQL_DSN，数据库相关能力将以降级模式启动")
		return nil, nil
	}

	db, err := gorm.Open(mysql.Open(cfg.MySQLDSN), &gorm.Config{
		Logger: loggerx.NewGormZapLogger(logger),
	})
	if err != nil {
		return nil, err
	}

	return db, nil
}
