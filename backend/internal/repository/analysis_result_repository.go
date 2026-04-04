package repository

import (
	"context"

	"gorm.io/gorm"
	"gorm.io/gorm/clause"

	"residential-energy-intelligence-agent-platform/internal/domain"
)

type AnalysisResultRepository interface {
	GetByDatasetID(ctx context.Context, datasetID uint64) (*domain.AnalysisResultRecord, error)
	Upsert(ctx context.Context, record *domain.AnalysisResultRecord) error
	DeleteByDatasetID(ctx context.Context, datasetID uint64) error
}

type analysisResultRepository struct {
	db *gorm.DB
}

func NewAnalysisResultRepository(db *gorm.DB) AnalysisResultRepository {
	return &analysisResultRepository{db: db}
}

func (r *analysisResultRepository) GetByDatasetID(ctx context.Context, datasetID uint64) (*domain.AnalysisResultRecord, error) {
	var record domain.AnalysisResultRecord
	if err := r.db.WithContext(ctx).
		Where("dataset_id = ?", datasetID).
		First(&record).Error; err != nil {
		return nil, err
	}
	return &record, nil
}

func (r *analysisResultRepository) Upsert(ctx context.Context, record *domain.AnalysisResultRecord) error {
	return r.db.WithContext(ctx).
		Clauses(clause.OnConflict{
			Columns: []clause.Column{{Name: "dataset_id"}},
			DoUpdates: clause.AssignmentColumns([]string{
				"total_kwh",
				"daily_avg_kwh",
				"max_load_w",
				"max_load_time",
				"min_load_w",
				"min_load_time",
				"peak_kwh",
				"valley_kwh",
				"flat_kwh",
				"peak_ratio",
				"valley_ratio",
				"flat_ratio",
				"detail_path",
				"created_at",
			}),
		}).
		Create(record).Error
}

func (r *analysisResultRepository) DeleteByDatasetID(ctx context.Context, datasetID uint64) error {
	return r.db.WithContext(ctx).
		Where("dataset_id = ?", datasetID).
		Delete(&domain.AnalysisResultRecord{}).Error
}
