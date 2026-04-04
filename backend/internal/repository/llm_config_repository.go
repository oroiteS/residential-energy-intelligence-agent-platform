package repository

import (
	"context"

	"gorm.io/gorm"

	"residential-energy-intelligence-agent-platform/internal/domain"
)

type LLMConfigRepository interface {
	List(ctx context.Context) ([]domain.LLMConfigRecord, error)
	GetByID(ctx context.Context, id uint64) (*domain.LLMConfigRecord, error)
	Count(ctx context.Context) (int64, error)
	Create(ctx context.Context, record *domain.LLMConfigRecord) error
	Update(ctx context.Context, record *domain.LLMConfigRecord) error
	Delete(ctx context.Context, id uint64) error
	ClearDefault(ctx context.Context) error
}

type llmConfigRepository struct {
	db *gorm.DB
}

func NewLLMConfigRepository(db *gorm.DB) LLMConfigRepository {
	return &llmConfigRepository{db: db}
}

func (r *llmConfigRepository) List(ctx context.Context) ([]domain.LLMConfigRecord, error) {
	var records []domain.LLMConfigRecord
	if err := r.db.WithContext(ctx).
		Order("is_default DESC").
		Order("created_at DESC").
		Find(&records).Error; err != nil {
		return nil, err
	}
	return records, nil
}

func (r *llmConfigRepository) GetByID(ctx context.Context, id uint64) (*domain.LLMConfigRecord, error) {
	var record domain.LLMConfigRecord
	if err := r.db.WithContext(ctx).
		First(&record, id).Error; err != nil {
		return nil, err
	}
	return &record, nil
}

func (r *llmConfigRepository) Count(ctx context.Context) (int64, error) {
	var total int64
	if err := r.db.WithContext(ctx).
		Model(&domain.LLMConfigRecord{}).
		Count(&total).Error; err != nil {
		return 0, err
	}
	return total, nil
}

func (r *llmConfigRepository) Create(ctx context.Context, record *domain.LLMConfigRecord) error {
	return r.db.WithContext(ctx).Create(record).Error
}

func (r *llmConfigRepository) Update(ctx context.Context, record *domain.LLMConfigRecord) error {
	return r.db.WithContext(ctx).Save(record).Error
}

func (r *llmConfigRepository) Delete(ctx context.Context, id uint64) error {
	return r.db.WithContext(ctx).Delete(&domain.LLMConfigRecord{}, id).Error
}

func (r *llmConfigRepository) ClearDefault(ctx context.Context) error {
	return r.db.WithContext(ctx).
		Model(&domain.LLMConfigRecord{}).
		Where("is_default = ?", true).
		Update("is_default", false).Error
}
