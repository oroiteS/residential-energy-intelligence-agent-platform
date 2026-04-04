package domain

import "time"

type ReportRecord struct {
	ID         uint64     `gorm:"column:id;primaryKey;autoIncrement"`
	DatasetID  uint64     `gorm:"column:dataset_id;not null"`
	ReportType string     `gorm:"column:report_type;type:enum('excel','html','pdf');not null"`
	FilePath   string     `gorm:"column:file_path;type:varchar(512);not null"`
	FileSize   uint64     `gorm:"column:file_size;not null"`
	CreatedAt  *time.Time `gorm:"column:created_at;type:datetime"`
}

func (ReportRecord) TableName() string {
	return "reports"
}
