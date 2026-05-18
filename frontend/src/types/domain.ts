export type DatasetStatus = 'uploaded' | 'processing' | 'ready' | 'error'
export type ForecastModelType = 'lstm'
export type ClassificationModelType = 'xgboost'
export type PredictedLabel =
  | '外出波动型'
  | '峰时集中型'
  | '中高用量型'
  | '高耗持续型'
  | '规律低耗型'
export type ReportType = 'excel' | 'html' | 'pdf'
export type ImportUnit = 'kwh' | 'wh' | 'w'
export type DetectionModelType = 'iforest_rules'
export type RiskFlag =
  | 'evening_peak_risk'
  | 'night_load_risk'
  | 'peak_usage_risk'
  | 'morning_spike_risk'
  | 'evening_peak'
  | 'daytime_peak'
  | 'high_baseload'
  | 'abnormal_rise'
  | 'peak_overlap_risk'
export type ChatRole = 'user' | 'assistant'
export type AgentIntent =
  | 'overview'
  | 'classification'
  | 'forecast'
  | 'advice'
  | 'risk'
  | 'follow_up'
export type AgentConfidenceLevel = 'high' | 'medium' | 'low'

export type PeakValleyConfig = {
  peak: string[]
  valley: string[]
}

export type ModelHistoryWindowConfig = {
  classification_days: number
  forecast_history_days: number
}

export type SystemConfig = {
  peak_valley_config: PeakValleyConfig
  model_history_window_config: ModelHistoryWindowConfig
  data_upload_dir: string
  report_output_dir: string
}

export type SystemConfigPatchInput = Partial<
  Pick<
    SystemConfig,
    | 'peak_valley_config'
    | 'model_history_window_config'
  >
>

export type HealthStatus = {
  status: 'up' | 'down' | 'degraded'
  service: string
  version?: string
  dependencies?: Record<string, string>
  peak_valley_config?: PeakValleyConfig
}

export type DatasetSummary = {
  id: number
  name: string
  description: string | null
  household_id: string | null
  row_count: number
  time_start: string | null
  time_end: string | null
  status: DatasetStatus
  created_at: string
  updated_at: string
}

export type DatasetListQuery = {
  page: number
  page_size: number
  status?: DatasetStatus
  keyword?: string
}

export type DatasetListPayload = {
  items: DatasetSummary[]
  pagination: {
    page: number
    page_size: number
    total: number
  }
}

export type DatasetDetail = DatasetSummary & {
  raw_file_path: string
  processed_file_path: string | null
  feature_cols: string[]
  column_mapping: Record<string, string>
  quality_report_path: string | null
  error_message: string | null
}

export type QualitySummary = {
  missing_rate: number
  duplicate_count: number
  sampling_interval: string
  cleaning_strategies: string[]
  min_granularity_minutes?: number | null
  max_granularity_minutes?: number | null
  accepted_min_granularity_minutes?: number | null
  accepted_max_granularity_minutes?: number | null
}

export type DatasetDetailPayload = {
  dataset: DatasetDetail
  quality_summary: QualitySummary | null
}

export type AnalysisSummary = {
  total_kwh: number
  daily_avg_kwh: number
  max_load_w: number
  max_load_time: string
  min_load_w: number
  min_load_time: string
  peak_kwh: number
  valley_kwh: number
  peak_ratio: number
  valley_ratio: number
}

export type DailyTrendPoint = {
  date: string
  kwh: number
}

export type WeeklyTrendPoint = {
  week_start: string
  week_end: string
  kwh: number
}

export type TypicalDayPoint = {
  hour: number
  avg_load_w: number
}

export type PeakPieItem = {
  name: string
  ratio: number
  kwh: number
}

export type AnalysisCharts = {
  daily_trend: DailyTrendPoint[]
  weekly_trend: WeeklyTrendPoint[]
  typical_day_curve: TypicalDayPoint[]
  peak_valley_pie: PeakPieItem[]
}

export type AnalysisPayload = {
  summary: AnalysisSummary
  peak_valley_config: PeakValleyConfig
  charts: AnalysisCharts
  detail_path: string
  updated_at: string
}

export type ClassificationProbabilities = Record<PredictedLabel, number>

export type ClassificationResult = {
  id: number
  dataset_id: number
  model_type: ClassificationModelType
  schema_version?: 'v1'
  predicted_label: PredictedLabel
  confidence: number
  label_display_name?: string
  probabilities: ClassificationProbabilities
  explanation?: string
  sample_id?: string
  runtime_library?: string
  window_start: string | null
  window_end: string | null
  created_at: string
}

export type ForecastClassificationSummary = {
  schema_version?: 'v1'
  model_type: ClassificationModelType
  predicted_label: PredictedLabel
  label_display_name?: string
  confidence: number
  probabilities: ClassificationProbabilities
  window_start: string | null
  window_end: string | null
  source: string
}

export type ForecastSummary = {
  forecast_start: string
  forecast_end: string
  granularity: string
  schema_version?: 'v1'
  forecast_horizon?: '7d'
  model_type?: ForecastModelType
  predicted_total_kwh?: number
  predicted_peak_kwh?: number
  predicted_valley_kwh?: number
  predicted_avg_daily_kwh?: number
  predicted_peak_ratio?: number
  predicted_valley_ratio?: number
  risk_flags: RiskFlag[]
  forecast_classification?: ForecastClassificationSummary
  future_detection?: DetectionResult | null
  confidence_hint?: AgentConfidenceLevel
}

export type ForecastRecord = {
  id: number
  dataset_id: number
  model_type: ForecastModelType
  forecast_start: string
  forecast_end: string
  granularity: string
  summary: ForecastSummary
  detail_path: string
  created_at: string
}

export type ForecastSeriesPoint = {
  date: string
  predicted_total_kwh: number
  predicted_peak_kwh: number
  predicted_valley_kwh: number
}

export type ForecastDetail = {
  forecast: ForecastRecord
  series: ForecastSeriesPoint[]
}

export type DetectionReason = {
  rule: string
  feature: string
  method: string
  severity: number
  reason: string
}

export type DetectionResult = {
  id: number
  dataset_id: number
  model_type: DetectionModelType
  window_start: string | null
  window_end: string | null
  window_role: 'current' | 'future'
  is_anomaly: boolean
  anomaly_score: number
  severity: 'low' | 'medium' | 'high'
  reasons: DetectionReason[]
  feature_summary: Record<string, number>
  classification_hint?: string | null
  created_at: string
}

export type ReportRecord = {
  id: number
  dataset_id: number
  report_type: ReportType
  file_path: string
  file_size: number
  created_at: string
}

export type ChatSession = {
  id: number
  dataset_id: number
  title: string
  created_at: string
  updated_at: string
}

export type ChatSessionInput = {
  dataset_id: number
  title: string
}

export type ChatMessage = {
  id: number
  session_id: number
  role: ChatRole
  content: string
  assistant_payload?: AssistantAnswer | null
  content_path: string | null
  model_name: string | null
  tokens_used: number | null
  created_at: string
}

export type Citation = {
  key: string
  label: string
  value: string | number | boolean | Array<string | number>
}

export type AssistantMissingInformation = {
  key: string
  question: string
  reason: string
}

export type AssistantAnswer = {
  session_id?: number
  answer: string
  citations: Citation[]
  actions: string[]
  degraded: boolean
  error_reason: string | null
  created_at?: string
  intent?: AgentIntent
  confidence_level?: AgentConfidenceLevel
  missing_information?: AssistantMissingInformation[]
}

export type ImportDatasetInput = {
  name: string
  description: string | null
  household_id: string | null
  unit: ImportUnit
  column_mapping?: Record<string, string>
  file_name: string
  file: File
}

export type ForecastPredictInput = {
  model_type: ForecastModelType
  granularity: 'daily'
  forecast_start: string
  forecast_end: string
  force_refresh: boolean
}
