import type {
  AgentConfidenceLevel,
  AgentIntent,
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
  外出波动型: '外出波动型',
  峰时集中型: '峰时集中型',
  中高用量型: '中高用量型',
  高耗持续型: '高耗持续型',
  规律低耗型: '规律低耗型',
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
  evening_peak: '晚高峰风险',
  daytime_peak: '白天高峰风险',
  high_baseload: '基线负荷偏高',
  abnormal_rise: '预测窗口需复核',
  peak_overlap_risk: '峰时叠加风险',
}

export const detectionFeatureLabelMap: Record<string, string> = {
  avg_energy: '窗口日均总用电',
  std_energy: '日用电标准差',
  max_energy: '窗口最大单日用电',
  min_energy: '窗口最小单日用电',
  avg_peak: '窗口日均峰时用电',
  avg_valley: '窗口日均谷时用电',
  peak_valley_ratio: '峰谷电量比',
  peak_ratio: '峰时电量占比',
  valley_ratio: '谷时电量占比',
  load_factor: '负荷均衡度',
  workday_avg: '工作日日均用电',
  weekend_avg: '周末日均用电',
  weekend_workday_ratio: '周末/工作日用电比',
  trend_rel: '短期趋势强度',
  volatility: '相邻日波动幅度',
  med_mean_ratio: '中位数/均值比',
}

export const detectionFeatureDescriptionMap: Record<string, string> = {
  avg_energy: '统计当前检测窗口内，每天总用电量的平均水平，用来判断整体耗电高低。',
  std_energy: '衡量窗口内每天总用电量的离散程度，数值越大说明日与日之间波动越明显。',
  max_energy: '窗口内单日总用电量的最高值，用来识别是否出现过特别高耗的一天。',
  min_energy: '窗口内单日总用电量的最低值，用来识别是否出现过特别低耗的一天。',
  avg_peak: '窗口内每天峰时段用电量的平均值，反映高负荷时段的典型用电水平。',
  avg_valley: '窗口内每天谷时段用电量的平均值，反映低价或低负荷时段的典型用电水平。',
  peak_valley_ratio: '峰时总电量与谷时总电量的比值，用来观察用电是否更集中在峰时。',
  peak_ratio: '峰时电量占总电量的比例，用来判断高峰时段用电是否偏集中。',
  valley_ratio: '谷时电量占总电量的比例，用来判断低谷时段用电是否偏集中。',
  load_factor: '用平均负荷相对峰值的稳定程度衡量负荷均衡性，越高通常越平稳。',
  workday_avg: '窗口内工作日的日均总用电量，用于对比日常作息下的典型耗电水平。',
  weekend_avg: '窗口内周末的日均总用电量，用于观察休息日是否存在不同用电习惯。',
  weekend_workday_ratio: '周末日均用电与工作日日均用电的比值，用于判断周末作息偏移程度。',
  trend_rel: '描述窗口内总用电是上升、下降还是平稳，数值越大表示趋势变化越明显。',
  volatility: '衡量相邻日期之间的变化幅度，越大说明短期内跳动越明显。',
  med_mean_ratio: '中位数与均值的比值，可用于识别是否存在少数极端高耗日拉高整体均值。',
}

export const detectionMethodLabelMap: Record<string, string> = {
  percentile: '分位阈值',
  abs_percentile: '绝对分位阈值',
  rule: '规则引擎',
}

export const forecastModelMap: Record<ForecastModelType, string> = {
  lstm: 'LSTM Direct',
}

export const assistantIntentMap: Record<AgentIntent, string> = {
  overview: '总体概览',
  classification: '分类解释',
  forecast: '预测判断',
  advice: '节能建议',
  risk: '风险提醒',
  follow_up: '连续追问',
}

export const assistantConfidenceMap: Record<
  AgentConfidenceLevel,
  { label: string; color: 'success' | 'processing' | 'warning' }
> = {
  high: { label: '高置信', color: 'success' },
  medium: { label: '中置信', color: 'processing' },
  low: { label: '低置信', color: 'warning' },
}
