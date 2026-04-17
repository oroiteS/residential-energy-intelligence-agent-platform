package service

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"gorm.io/gorm"

	"residential-energy-intelligence-agent-platform/internal/config"
	"residential-energy-intelligence-agent-platform/internal/domain"
	"residential-energy-intelligence-agent-platform/internal/repository"
	"residential-energy-intelligence-agent-platform/pkg/apperror"
)

type ChatService struct {
	cfg         *config.AppConfig
	datasetRepo repository.DatasetRepository
	sessionRepo repository.ChatSessionRepository
	messageRepo repository.ChatMessageRepository
}

type CreateChatSessionInput struct {
	DatasetID uint64 `json:"dataset_id"`
	Title     string `json:"title"`
}

type ChatSessionListParams struct {
	Page      int
	PageSize  int
	DatasetID uint64
}

type ChatMessageListParams struct {
	Page     int
	PageSize int
}

func NewChatService(
	cfg *config.AppConfig,
	datasetRepo repository.DatasetRepository,
	sessionRepo repository.ChatSessionRepository,
	messageRepo repository.ChatMessageRepository,
) *ChatService {
	return &ChatService{
		cfg:         cfg,
		datasetRepo: datasetRepo,
		sessionRepo: sessionRepo,
		messageRepo: messageRepo,
	}
}

func (s *ChatService) CreateSession(ctx context.Context, input CreateChatSessionInput) (map[string]any, *apperror.AppError) {
	if s.sessionRepo == nil || s.datasetRepo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持聊天会话", nil)
	}
	if input.DatasetID == 0 {
		return nil, apperror.InvalidRequest("dataset_id 不能为空", nil)
	}

	title := strings.TrimSpace(input.Title)
	if title == "" {
		return nil, apperror.InvalidRequest("title 不能为空", nil)
	}

	if _, err := s.datasetRepo.GetByID(ctx, input.DatasetID); err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, apperror.NotFound("DATASET_NOT_FOUND", "数据集不存在", map[string]any{"dataset_id": input.DatasetID})
		}
		return nil, apperror.Internal(err)
	}

	now := time.Now()
	record := &domain.ChatSessionRecord{
		DatasetID: &input.DatasetID,
		Title:     &title,
		CreatedAt: &now,
		UpdatedAt: &now,
	}
	if err := s.sessionRepo.Create(ctx, record); err != nil {
		return nil, apperror.Internal(err)
	}
	return map[string]any{"session": chatSessionDTO(record)}, nil
}

func (s *ChatService) ListSessions(ctx context.Context, params ChatSessionListParams) (map[string]any, *apperror.AppError) {
	if s.sessionRepo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持聊天会话", nil)
	}

	var datasetID *uint64
	if params.DatasetID > 0 {
		datasetID = &params.DatasetID
	}
	records, total, err := s.sessionRepo.List(ctx, datasetID, params.Page, params.PageSize)
	if err != nil {
		return nil, apperror.Internal(err)
	}

	items := make([]map[string]any, 0, len(records))
	for _, record := range records {
		recordCopy := record
		items = append(items, chatSessionDTO(&recordCopy))
	}

	page := params.Page
	if page <= 0 {
		page = 1
	}
	pageSize := params.PageSize
	if pageSize <= 0 {
		pageSize = 20
	}
	if pageSize > 100 {
		pageSize = 100
	}

	return map[string]any{
		"items": items,
		"pagination": map[string]any{
			"page":      page,
			"page_size": pageSize,
			"total":     total,
		},
	}, nil
}

func (s *ChatService) DeleteSession(ctx context.Context, sessionID uint64) *apperror.AppError {
	if s.sessionRepo == nil || s.messageRepo == nil {
		return apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持聊天会话", nil)
	}

	if _, err := s.sessionRepo.GetByID(ctx, sessionID); err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return apperror.NotFound("CHAT_SESSION_NOT_FOUND", "聊天会话不存在", map[string]any{"session_id": sessionID})
		}
		return apperror.Internal(err)
	}

	messages, err := s.messageRepo.ListAllBySessionID(ctx, sessionID)
	if err != nil {
		return apperror.Internal(err)
	}
	for _, message := range messages {
		if message.ContentPath != nil && strings.TrimSpace(*message.ContentPath) != "" {
			_ = os.Remove(*message.ContentPath)
		}
	}

	if err := s.sessionRepo.Delete(ctx, sessionID); err != nil {
		return apperror.Internal(err)
	}
	return nil
}

func (s *ChatService) ListMessages(ctx context.Context, sessionID uint64, params ChatMessageListParams) (map[string]any, *apperror.AppError) {
	if s.sessionRepo == nil || s.messageRepo == nil {
		return nil, apperror.ServiceUnavailable("INTERNAL_ERROR", "数据库未配置，暂不支持聊天消息", nil)
	}

	if _, err := s.sessionRepo.GetByID(ctx, sessionID); err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, apperror.NotFound("CHAT_SESSION_NOT_FOUND", "聊天会话不存在", map[string]any{"session_id": sessionID})
		}
		return nil, apperror.Internal(err)
	}

	records, total, err := s.messageRepo.ListBySessionID(ctx, sessionID, params.Page, params.PageSize)
	if err != nil {
		return nil, apperror.Internal(err)
	}

	items := make([]map[string]any, 0, len(records))
	for _, record := range records {
		recordCopy := record
		items = append(items, chatMessageDTO(&recordCopy))
	}

	page := params.Page
	if page <= 0 {
		page = 1
	}
	pageSize := params.PageSize
	if pageSize <= 0 {
		pageSize = 50
	}
	if pageSize > 200 {
		pageSize = 200
	}

	return map[string]any{
		"items": items,
		"pagination": map[string]any{
			"page":      page,
			"page_size": pageSize,
			"total":     total,
		},
	}, nil
}

func writeAssistantMessagePayload(outputRootDir string, sessionID, messageID uint64, payload map[string]any) (string, error) {
	messageDir := filepath.Join(outputRootDir, "chat_messages")
	if err := os.MkdirAll(messageDir, 0o755); err != nil {
		return "", err
	}
	path := filepath.ToSlash(filepath.Join(messageDir, fmt.Sprintf("session_%d_message_%d.json", sessionID, messageID)))
	content, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(path, content, 0o644); err != nil {
		return "", err
	}
	return path, nil
}

func chatSessionDTO(record *domain.ChatSessionRecord) map[string]any {
	var datasetID any
	if record.DatasetID != nil {
		datasetID = *record.DatasetID
	}
	return map[string]any{
		"id":         record.ID,
		"dataset_id": datasetID,
		"title":      nullableString(record.Title),
		"created_at": record.CreatedAt,
		"updated_at": record.UpdatedAt,
	}
}

func chatMessageDTO(record *domain.ChatMessageRecord) map[string]any {
	result := map[string]any{
		"id":           record.ID,
		"session_id":   record.SessionID,
		"role":         record.Role,
		"content":      nullableString(record.Content),
		"content_path": nullableString(record.ContentPath),
		"model_name":   nullableString(record.ModelName),
		"tokens_used":  record.TokensUsed,
		"created_at":   record.CreatedAt,
	}
	if payload := loadAssistantMessagePayload(record.ContentPath); len(payload) > 0 {
		result["assistant_payload"] = payload
	}
	return result
}

func loadAssistantMessagePayload(contentPath *string) map[string]any {
	if contentPath == nil || strings.TrimSpace(*contentPath) == "" {
		return map[string]any{}
	}

	content, err := os.ReadFile(strings.TrimSpace(*contentPath))
	if err != nil {
		return map[string]any{}
	}

	var payload map[string]any
	if err := json.Unmarshal(content, &payload); err != nil {
		return map[string]any{}
	}
	return payload
}
