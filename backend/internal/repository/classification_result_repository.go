package repository

import (
	"context"
	"time"

	"gorm.io/gorm"

	"residential-energy-intelligence-agent-platform/internal/domain"
)

type ClassificationResultRepository interface {
	Create(ctx context.Context, record *domain.ClassificationResultRecord) error
	GetLatest(ctx context.Context, datasetID uint64, modelType string) (*domain.ClassificationResultRecord, error)
	GetLatestByWindow(ctx context.Context, datasetID uint64, modelType string, windowStart, windowEnd *time.Time) (*domain.ClassificationResultRecord, error)
	ListByDatasetID(ctx context.Context, datasetID uint64, modelType string, page, pageSize int) ([]domain.ClassificationResultRecord, int64, error)
}

type classificationResultRepository struct {
	db *gorm.DB
}

func NewClassificationResultRepository(db *gorm.DB) ClassificationResultRepository {
	return &classificationResultRepository{db: db}
}

func (r *classificationResultRepository) Create(ctx context.Context, record *domain.ClassificationResultRecord) error {
	return r.db.WithContext(ctx).Create(record).Error
}

func (r *classificationResultRepository) GetLatest(ctx context.Context, datasetID uint64, modelType string) (*domain.ClassificationResultRecord, error) {
	var record domain.ClassificationResultRecord
	query := r.db.WithContext(ctx).Where("dataset_id = ?", datasetID)
	if modelType != "" {
		query = query.Where("model_type = ?", modelType)
	}
	if err := query.Order("created_at DESC, id DESC").First(&record).Error; err != nil {
		return nil, err
	}
	return &record, nil
}

func (r *classificationResultRepository) GetLatestByWindow(ctx context.Context, datasetID uint64, modelType string, windowStart, windowEnd *time.Time) (*domain.ClassificationResultRecord, error) {
	var record domain.ClassificationResultRecord
	query := r.db.WithContext(ctx).
		Where("dataset_id = ?", datasetID).
		Where("model_type = ?", modelType)

	if windowStart == nil || windowEnd == nil {
		query = query.Where("window_start IS NULL").Where("window_end IS NULL")
	} else {
		query = query.Where("window_start = ?", *windowStart).Where("window_end = ?", *windowEnd)
	}

	if err := query.Order("created_at DESC, id DESC").First(&record).Error; err != nil {
		return nil, err
	}
	return &record, nil
}

func (r *classificationResultRepository) ListByDatasetID(ctx context.Context, datasetID uint64, modelType string, page, pageSize int) ([]domain.ClassificationResultRecord, int64, error) {
	query := r.db.WithContext(ctx).Model(&domain.ClassificationResultRecord{}).Where("dataset_id = ?", datasetID)
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

	var records []domain.ClassificationResultRecord
	if err := query.Order("created_at DESC, id DESC").
		Offset((page - 1) * pageSize).
		Limit(pageSize).
		Find(&records).Error; err != nil {
		return nil, 0, err
	}
	return records, total, nil
}
