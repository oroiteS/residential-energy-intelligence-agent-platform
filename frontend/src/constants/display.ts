import type {
  AdviceType,
  DatasetStatus,
  ForecastModelType,
  PredictedLabel,
  ReportType,
  RiskFlag,
} from '@/types/domain'

export const datasetStatusMap: Record<DatasetStatus, { label: string; color: string }> = {
  uploaded: { label: '已接收', color: 'default' },
  processing: { label: '处理中', color: 'processing' },
  ready: { label: '可查看', color: 'success' },
  error: { label: '处理失败', color: 'error' },
}

export const classificationLabelMap: Record<PredictedLabel, string> = {
  day_high_night_low: '白天高晚上低型',
  day_low_night_high: '白天低晚上高型',
  all_day_high: '全天高负载型',
  all_day_low: '全天低负载型',
}

export const adviceTypeMap: Record<AdviceType, string> = {
  rule: '规则建议',
  llm: '智能建议',
}

export const reportTypeMap: Record<ReportType, string> = {
  excel: 'Excel 报告',
  html: 'HTML 报告',
  pdf: 'PDF 报告',
}

export const riskFlagMap: Record<RiskFlag, string> = {
  evening_peak_risk: '晚高峰风险',
  night_load_risk: '夜间负荷风险',
  peak_usage_risk: '峰时高负荷风险',
  morning_spike_risk: '清晨突增风险',
}

export const forecastModelMap: Record<ForecastModelType, string> = {
  lstm: 'Seq2Seq LSTM',
  transformer: 'Transformer',
}
