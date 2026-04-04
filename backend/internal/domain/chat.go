package domain

import "time"

type ChatSessionRecord struct {
	ID        uint64     `gorm:"column:id;primaryKey;autoIncrement"`
	DatasetID *uint64    `gorm:"column:dataset_id"`
	Title     *string    `gorm:"column:title;type:varchar(128)"`
	CreatedAt *time.Time `gorm:"column:created_at;type:datetime"`
	UpdatedAt *time.Time `gorm:"column:updated_at;type:datetime"`
}

func (ChatSessionRecord) TableName() string {
	return "chat_sessions"
}

type ChatMessageRecord struct {
	ID          uint64     `gorm:"column:id;primaryKey;autoIncrement"`
	SessionID   uint64     `gorm:"column:session_id;not null"`
	Role        string     `gorm:"column:role;type:enum('user','assistant','system');not null"`
	Content     *string    `gorm:"column:content;type:text"`
	ContentPath *string    `gorm:"column:content_path;type:varchar(512)"`
	ModelName   *string    `gorm:"column:model_name;type:varchar(128)"`
	TokensUsed  uint32     `gorm:"column:tokens_used;type:int unsigned;not null"`
	CreatedAt   *time.Time `gorm:"column:created_at;type:datetime"`
}

func (ChatMessageRecord) TableName() string {
	return "chat_messages"
}
