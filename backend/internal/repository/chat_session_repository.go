package repository

import (
	"context"
	"time"

	"gorm.io/gorm"

	"residential-energy-intelligence-agent-platform/internal/domain"
)

type ChatSessionRepository interface {
	Create(ctx context.Context, record *domain.ChatSessionRecord) error
	GetByID(ctx context.Context, id uint64) (*domain.ChatSessionRecord, error)
	List(ctx context.Context, datasetID *uint64, page, pageSize int) ([]domain.ChatSessionRecord, int64, error)
	Touch(ctx context.Context, id uint64, updatedAt time.Time) error
	Delete(ctx context.Context, id uint64) error
}

type chatSessionRepository struct {
	db *gorm.DB
}

func NewChatSessionRepository(db *gorm.DB) ChatSessionRepository {
	return &chatSessionRepository{db: db}
}

func (r *chatSessionRepository) Create(ctx context.Context, record *domain.ChatSessionRecord) error {
	return r.db.WithContext(ctx).Create(record).Error
}

func (r *chatSessionRepository) GetByID(ctx context.Context, id uint64) (*domain.ChatSessionRecord, error) {
	var record domain.ChatSessionRecord
	if err := r.db.WithContext(ctx).First(&record, id).Error; err != nil {
		return nil, err
	}
	return &record, nil
}

func (r *chatSessionRepository) List(ctx context.Context, datasetID *uint64, page, pageSize int) ([]domain.ChatSessionRecord, int64, error) {
	query := r.db.WithContext(ctx).Model(&domain.ChatSessionRecord{})
	if datasetID != nil {
		query = query.Where("dataset_id = ?", *datasetID)
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

	var records []domain.ChatSessionRecord
	if err := query.Order("updated_at DESC, id DESC").
		Offset((page - 1) * pageSize).
		Limit(pageSize).
		Find(&records).Error; err != nil {
		return nil, 0, err
	}
	return records, total, nil
}

func (r *chatSessionRepository) Touch(ctx context.Context, id uint64, updatedAt time.Time) error {
	return r.db.WithContext(ctx).
		Model(&domain.ChatSessionRecord{}).
		Where("id = ?", id).
		Update("updated_at", updatedAt).Error
}

func (r *chatSessionRepository) Delete(ctx context.Context, id uint64) error {
	return r.db.WithContext(ctx).Delete(&domain.ChatSessionRecord{}, id).Error
}
