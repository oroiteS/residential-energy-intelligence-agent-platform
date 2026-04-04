package repository

import (
	"context"

	"gorm.io/gorm"

	"residential-energy-intelligence-agent-platform/internal/domain"
)

type ReportRepository interface {
	Create(ctx context.Context, record *domain.ReportRecord) error
	GetByID(ctx context.Context, id uint64) (*domain.ReportRecord, error)
	ListByDatasetID(ctx context.Context, datasetID uint64) ([]domain.ReportRecord, error)
	DeleteByDatasetID(ctx context.Context, datasetID uint64) error
}

type reportRepository struct {
	db *gorm.DB
}

func NewReportRepository(db *gorm.DB) ReportRepository {
	return &reportRepository{db: db}
}

func (r *reportRepository) Create(ctx context.Context, record *domain.ReportRecord) error {
	return r.db.WithContext(ctx).Create(record).Error
}

func (r *reportRepository) GetByID(ctx context.Context, id uint64) (*domain.ReportRecord, error) {
	var record domain.ReportRecord
	if err := r.db.WithContext(ctx).First(&record, id).Error; err != nil {
		return nil, err
	}
	return &record, nil
}

func (r *reportRepository) ListByDatasetID(ctx context.Context, datasetID uint64) ([]domain.ReportRecord, error) {
	var records []domain.ReportRecord
	if err := r.db.WithContext(ctx).
		Where("dataset_id = ?", datasetID).
		Order("created_at DESC").
		Find(&records).Error; err != nil {
		return nil, err
	}
	return records, nil
}

func (r *reportRepository) DeleteByDatasetID(ctx context.Context, datasetID uint64) error {
	return r.db.WithContext(ctx).
		Where("dataset_id = ?", datasetID).
		Delete(&domain.ReportRecord{}).Error
}
