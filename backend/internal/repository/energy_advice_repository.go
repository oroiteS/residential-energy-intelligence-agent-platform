package repository

import (
	"context"

	"gorm.io/gorm"

	"residential-energy-intelligence-agent-platform/internal/domain"
)

type EnergyAdviceRepository interface {
	Create(ctx context.Context, record *domain.EnergyAdviceRecord) error
	GetByID(ctx context.Context, id uint64) (*domain.EnergyAdviceRecord, error)
	ListByDatasetID(ctx context.Context, datasetID uint64, adviceType string) ([]domain.EnergyAdviceRecord, error)
	DeleteByDatasetID(ctx context.Context, datasetID uint64) error
}

type energyAdviceRepository struct {
	db *gorm.DB
}

func NewEnergyAdviceRepository(db *gorm.DB) EnergyAdviceRepository {
	return &energyAdviceRepository{db: db}
}

func (r *energyAdviceRepository) Create(ctx context.Context, record *domain.EnergyAdviceRecord) error {
	return r.db.WithContext(ctx).Create(record).Error
}

func (r *energyAdviceRepository) GetByID(ctx context.Context, id uint64) (*domain.EnergyAdviceRecord, error) {
	var record domain.EnergyAdviceRecord
	if err := r.db.WithContext(ctx).First(&record, id).Error; err != nil {
		return nil, err
	}
	return &record, nil
}

func (r *energyAdviceRepository) ListByDatasetID(ctx context.Context, datasetID uint64, adviceType string) ([]domain.EnergyAdviceRecord, error) {
	query := r.db.WithContext(ctx).
		Where("dataset_id = ?", datasetID).
		Order("created_at DESC")
	if adviceType != "" {
		query = query.Where("advice_type = ?", adviceType)
	}

	var records []domain.EnergyAdviceRecord
	if err := query.Find(&records).Error; err != nil {
		return nil, err
	}
	return records, nil
}

func (r *energyAdviceRepository) DeleteByDatasetID(ctx context.Context, datasetID uint64) error {
	return r.db.WithContext(ctx).
		Where("dataset_id = ?", datasetID).
		Delete(&domain.EnergyAdviceRecord{}).Error
}
