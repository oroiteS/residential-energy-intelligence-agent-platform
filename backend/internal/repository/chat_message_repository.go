package repository

import (
	"context"

	"gorm.io/gorm"

	"residential-energy-intelligence-agent-platform/internal/domain"
)

type ChatMessageRepository interface {
	Create(ctx context.Context, record *domain.ChatMessageRecord) error
	Update(ctx context.Context, record *domain.ChatMessageRecord) error
	ListBySessionID(ctx context.Context, sessionID uint64, page, pageSize int) ([]domain.ChatMessageRecord, int64, error)
	ListAllBySessionID(ctx context.Context, sessionID uint64) ([]domain.ChatMessageRecord, error)
}

type chatMessageRepository struct {
	db *gorm.DB
}

func NewChatMessageRepository(db *gorm.DB) ChatMessageRepository {
	return &chatMessageRepository{db: db}
}

func (r *chatMessageRepository) Create(ctx context.Context, record *domain.ChatMessageRecord) error {
	return r.db.WithContext(ctx).Create(record).Error
}

func (r *chatMessageRepository) Update(ctx context.Context, record *domain.ChatMessageRecord) error {
	return r.db.WithContext(ctx).Save(record).Error
}

func (r *chatMessageRepository) ListBySessionID(ctx context.Context, sessionID uint64, page, pageSize int) ([]domain.ChatMessageRecord, int64, error) {
	query := r.db.WithContext(ctx).Model(&domain.ChatMessageRecord{}).Where("session_id = ?", sessionID)

	var total int64
	if err := query.Count(&total).Error; err != nil {
		return nil, 0, err
	}

	if page <= 0 {
		page = 1
	}
	if pageSize <= 0 {
		pageSize = 50
	}
	if pageSize > 200 {
		pageSize = 200
	}

	var records []domain.ChatMessageRecord
	if err := query.Order("created_at ASC, id ASC").
		Offset((page - 1) * pageSize).
		Limit(pageSize).
		Find(&records).Error; err != nil {
		return nil, 0, err
	}
	return records, total, nil
}

func (r *chatMessageRepository) ListAllBySessionID(ctx context.Context, sessionID uint64) ([]domain.ChatMessageRecord, error) {
	var records []domain.ChatMessageRecord
	if err := r.db.WithContext(ctx).
		Where("session_id = ?", sessionID).
		Order("created_at ASC, id ASC").
		Find(&records).Error; err != nil {
		return nil, err
	}
	return records, nil
}
