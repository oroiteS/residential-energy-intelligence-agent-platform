export type DatasetStatus = 'uploaded' | 'processing' | 'ready' | 'error'
export type ForecastModelType = 'tft'
export type ClassificationModelType = 'xgboost'
export type PredictedLabel =
  | 'morning_peak'
  | 'afternoon_peak'
  | 'day_low_night_high'
  | 'all_day_low'
export type AdviceType = 'rule' | 'llm'
export type ReportType = 'excel' | 'html' | 'pdf'
export type ImportUnit = 'kwh' | 'wh' | 'w'
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
  energy_advice_prompt_template: string
  data_upload_dir: string
  report_output_dir: string
}

export type SystemConfigPatchInput = Partial<
  Pick<
    SystemConfig,
    | 'peak_valley_config'
    | 'model_history_window_config'
    | 'energy_advice_prompt_template'
  >
>

export type HealthStatus = {
  status: 'up' | 'down' | 'degraded'
  service: string
  version?: string
  dependencies?: Record<string, string>
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
  flat_kwh: number
  peak_ratio: number
  valley_ratio: number
  flat_ratio: number
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

export type ForecastSummary = {
  forecast_start: string
  forecast_end: string
  granularity: string
  schema_version?: 'v1'
  forecast_horizon?: '1d'
  predicted_avg_load_w: number
  predicted_peak_load_w: number
  predicted_total_kwh?: number
  peak_period?: string
  forecast_peak_periods?: string[]
  predicted_peak_ratio?: number
  predicted_valley_ratio?: number
  predicted_flat_ratio?: number
  risk_flags: RiskFlag[]
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
  timestamp: string
  predicted: number
}

export type ForecastDetail = {
  forecast: ForecastRecord
  series: ForecastSeriesPoint[]
}

export type AdviceSummary = {
  id: number
  dataset_id: number
  classification_id: number | null
  advice_type: AdviceType
  summary: string
  content_path: string
  created_at: string
}

export type AdviceContentItem = {
  reason: string
  action: string
}

export type AdviceDetail = {
  advice: AdviceSummary
  content: {
    items: AdviceContentItem[]
  }
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
  granularity: string
  forecast_start: string
  forecast_end: string
  force_refresh: boolean
}
