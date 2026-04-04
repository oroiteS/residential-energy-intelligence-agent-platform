package repository

import (
	"context"
	"time"

	"gorm.io/gorm"
	"gorm.io/gorm/clause"

	"residential-energy-intelligence-agent-platform/internal/domain"
)

type SystemConfigRepository interface {
	List(ctx context.Context) ([]domain.SystemConfigRecord, error)
	Upsert(ctx context.Context, record *domain.SystemConfigRecord) error
	Delete(ctx context.Context, configKey string) error
}

type systemConfigRepository struct {
	db *gorm.DB
}

func NewSystemConfigRepository(db *gorm.DB) SystemConfigRepository {
	return &systemConfigRepository{db: db}
}

func (r *systemConfigRepository) List(ctx context.Context) ([]domain.SystemConfigRecord, error) {
	var records []domain.SystemConfigRecord
	if err := r.db.WithContext(ctx).
		Order("config_key ASC").
		Find(&records).Error; err != nil {
		return nil, err
	}
	return records, nil
}

func (r *systemConfigRepository) Upsert(ctx context.Context, record *domain.SystemConfigRecord) error {
	now := time.Now()
	return r.db.WithContext(ctx).
		Model(&domain.SystemConfigRecord{}).
		Clauses(clause.OnConflict{
			Columns:   []clause.Column{{Name: "config_key"}},
			DoUpdates: clause.AssignmentColumns([]string{"config_value", "description", "updated_at"}),
		}).
		Create(map[string]any{
			"config_key":   record.ConfigKey,
			"config_value": record.ConfigValue,
			"description":  record.Description,
			"updated_at":   now,
		}).Error
}

func (r *systemConfigRepository) Delete(ctx context.Context, configKey string) error {
	return r.db.WithContext(ctx).
		Where("config_key = ?", configKey).
		Delete(&domain.SystemConfigRecord{}).Error
}
