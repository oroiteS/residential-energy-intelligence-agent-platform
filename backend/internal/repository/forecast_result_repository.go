package repository

import (
	"context"
	"time"

	"gorm.io/gorm"

	"residential-energy-intelligence-agent-platform/internal/domain"
)

type ForecastResultRepository interface {
	Create(ctx context.Context, record *domain.ForecastResultRecord) error
	GetByID(ctx context.Context, id uint64) (*domain.ForecastResultRecord, error)
	GetLatestByRange(ctx context.Context, datasetID uint64, modelType string, forecastStart, forecastEnd time.Time, granularity string) (*domain.ForecastResultRecord, error)
	ListByDatasetID(ctx context.Context, datasetID uint64, modelType string, page, pageSize int) ([]domain.ForecastResultRecord, int64, error)
	ListAllByDatasetID(ctx context.Context, datasetID uint64) ([]domain.ForecastResultRecord, error)
}

type forecastResultRepository struct {
	db *gorm.DB
}

func NewForecastResultRepository(db *gorm.DB) ForecastResultRepository {
	return &forecastResultRepository{db: db}
}

func (r *forecastResultRepository) Create(ctx context.Context, record *domain.ForecastResultRecord) error {
	return r.db.WithContext(ctx).Create(record).Error
}

func (r *forecastResultRepository) GetByID(ctx context.Context, id uint64) (*domain.ForecastResultRecord, error) {
	var record domain.ForecastResultRecord
	if err := r.db.WithContext(ctx).First(&record, id).Error; err != nil {
		return nil, err
	}
	return &record, nil
}

func (r *forecastResultRepository) GetLatestByRange(ctx context.Context, datasetID uint64, modelType string, forecastStart, forecastEnd time.Time, granularity string) (*domain.ForecastResultRecord, error) {
	var record domain.ForecastResultRecord
	if err := r.db.WithContext(ctx).
		Where("dataset_id = ?", datasetID).
		Where("model_type = ?", modelType).
		Where("forecast_start = ?", forecastStart).
		Where("forecast_end = ?", forecastEnd).
		Where("granularity = ?", granularity).
		Order("created_at DESC, id DESC").
		First(&record).Error; err != nil {
		return nil, err
	}
	return &record, nil
}

func (r *forecastResultRepository) ListByDatasetID(ctx context.Context, datasetID uint64, modelType string, page, pageSize int) ([]domain.ForecastResultRecord, int64, error) {
	query := r.db.WithContext(ctx).Model(&domain.ForecastResultRecord{}).Where("dataset_id = ?", datasetID)
	if modelType != "" {
		query = query.Where("model_type = ?", modelType)
	}

	var total int64
	if err := query.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	if page <= 0 {
		page = 1
	}
	if pageSize <= 0 {
		pageSize = 20
	}
	if pageSize > 100 {
		pageSize = 100
	}

	var records []domain.ForecastResultRecord
	if err := query.Order("created_at DESC, id DESC").
		Offset((page - 1) * pageSize).
		Limit(pageSize).
		Find(&records).Error; err != nil {
		return nil, 0, err
	}
	return records, total, nil
}

func (r *forecastResultRepository) ListAllByDatasetID(ctx context.Context, datasetID uint64) ([]domain.ForecastResultRecord, error) {
	var records []domain.ForecastResultRecord
	if err := r.db.WithContext(ctx).
		Where("dataset_id = ?", datasetID).
		Find(&records).Error; err != nil {
		return nil, err
	}
	return records, nil
}
