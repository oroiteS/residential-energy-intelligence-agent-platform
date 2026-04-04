package domain

import "time"

type DatasetRecord struct {
	ID                uint64     `gorm:"column:id;primaryKey;autoIncrement"`
	Name              string     `gorm:"column:name;type:varchar(128);not null"`
	Description       *string    `gorm:"column:description;type:text"`
	RawFilePath       string     `gorm:"column:raw_file_path;type:varchar(512);not null"`
	ProcessedFilePath *string    `gorm:"column:processed_file_path;type:varchar(512)"`
	HouseholdID       *string    `gorm:"column:household_id;type:varchar(64)"`
	RowCount          uint32     `gorm:"column:row_count;type:int unsigned;not null"`
	TimeStart         *time.Time `gorm:"column:time_start;type:datetime"`
	TimeEnd           *time.Time `gorm:"column:time_end;type:datetime"`
	FeatureCols       []byte     `gorm:"column:feature_cols;type:json"`
	ColumnMapping     []byte     `gorm:"column:column_mapping;type:json"`
	Status            string     `gorm:"column:status;type:enum('uploaded','processing','ready','error');not null"`
	QualityReportPath *string    `gorm:"column:quality_report_path;type:varchar(512)"`
	ErrorMessage      *string    `gorm:"column:error_message;type:text"`
	CreatedAt         *time.Time `gorm:"column:created_at;type:datetime"`
	UpdatedAt         *time.Time `gorm:"column:updated_at;type:datetime"`
}

func (DatasetRecord) TableName() string {
	return "datasets"
}

type DatasetListFilter struct {
	Page     int
	PageSize int
	Status   string
	Keyword  string
}

type DatasetQualitySummary struct {
	MissingRate        float64  `json:"missing_rate"`
	DuplicateCount     int      `json:"duplicate_count"`
	SamplingInterval   string   `json:"sampling_interval"`
	CleaningStrategies []string `json:"cleaning_strategies"`
}

type DatasetQualityReport struct {
	DatasetID             uint64                `json:"dataset_id"`
	InputRowCount         int                   `json:"input_row_count"`
	ProcessedRowCount     int                   `json:"processed_row_count"`
	MissingRate           float64               `json:"missing_rate"`
	DuplicateCount        int                   `json:"duplicate_count"`
	SamplingInterval      string                `json:"sampling_interval"`
	CleaningStrategies    []string              `json:"cleaning_strategies"`
	FeatureCols           []string              `json:"feature_cols"`
	ResolvedColumnMapping map[string]string     `json:"resolved_column_mapping"`
	DetectedApplianceCols []string              `json:"detected_appliance_cols"`
	QualitySummary        DatasetQualitySummary `json:"quality_summary"`
}
