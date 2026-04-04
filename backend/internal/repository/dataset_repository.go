package repository

import (
	"context"
	"strings"

	"gorm.io/gorm"

	"residential-energy-intelligence-agent-platform/internal/domain"
)

type DatasetRepository interface {
	Create(ctx context.Context, record *domain.DatasetRecord) error
	Update(ctx context.Context, record *domain.DatasetRecord) error
	GetByID(ctx context.Context, id uint64) (*domain.DatasetRecord, error)
	List(ctx context.Context, filter domain.DatasetListFilter) ([]domain.DatasetRecord, int64, error)
	Delete(ctx context.Context, id uint64) error
}

type datasetRepository struct {
	db *gorm.DB
}

func NewDatasetRepository(db *gorm.DB) DatasetRepository {
	return &datasetRepository{db: db}
}

func (r *datasetRepository) Create(ctx context.Context, record *domain.DatasetRecord) error {
	return r.db.WithContext(ctx).Create(record).Error
}

func (r *datasetRepository) Update(ctx context.Context, record *domain.DatasetRecord) error {
	return r.db.WithContext(ctx).Save(record).Error
}

func (r *datasetRepository) GetByID(ctx context.Context, id uint64) (*domain.DatasetRecord, error) {
	var record domain.DatasetRecord
	if err := r.db.WithContext(ctx).First(&record, id).Error; err != nil {
		return nil, err
	}
	return &record, nil
}

func (r *datasetRepository) List(ctx context.Context, filter domain.DatasetListFilter) ([]domain.DatasetRecord, int64, error) {
	query := r.db.WithContext(ctx).Model(&domain.DatasetRecord{})

	if status := strings.TrimSpace(filter.Status); status != "" {
		query = query.Where("status = ?", status)
	}
	if keyword := strings.TrimSpace(filter.Keyword); keyword != "" {
		query = query.Where("name LIKE ? OR household_id LIKE ?", "%"+keyword+"%", "%"+keyword+"%")
	}

	var total int64
	if err := query.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	page := filter.Page
	if page <= 0 {
		page = 1
	}
	pageSize := filter.PageSize
	if pageSize <= 0 {
		pageSize = 20
	}
	if pageSize > 100 {
		pageSize = 100
	}

	var records []domain.DatasetRecord
	if err := query.
		Order("created_at DESC").
		Offset((page - 1) * pageSize).
		Limit(pageSize).
		Find(&records).Error; err != nil {
		return nil, 0, err
	}

	return records, total, nil
}

func (r *datasetRepository) Delete(ctx context.Context, id uint64) error {
	return r.db.WithContext(ctx).Delete(&domain.DatasetRecord{}, id).Error
}
