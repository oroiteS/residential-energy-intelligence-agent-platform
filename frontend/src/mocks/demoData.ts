import type {
  AnalysisPayload,
  AssistantAnswer,
  ChatMessage,
  ChatSession,
  ClassificationResult,
  ClassificationProbabilities,
  DatasetDetailPayload,
  DatasetSummary,
  DetectionResult,
  ForecastDetail,
  ForecastSeriesPoint,
  ForecastModelType,
  ForecastRecord,
  HealthStatus,
  ImportDatasetInput,
  ReportRecord,
  ReportType,
  SystemConfig,
} from '@/types/domain'

function createDailyTrend(base: number, variance: number) {
  return Array.from({ length: 10 }, (_, index) => ({
    date: `2014-12-${String(index + 1).padStart(2, '0')}`,
    kwh: Number((base + Math.sin(index / 1.7) * variance + index * 0.3).toFixed(2)),
  }))
}

function createWeeklyTrend(base: number) {
  return Array.from({ length: 4 }, (_, index) => ({
    week_start: `2014-11-${String(index * 7 + 1).padStart(2, '0')}`,
    week_end: `2014-11-${String(index * 7 + 7).padStart(2, '0')}`,
    kwh: Number((base + index * 4.2).toFixed(2)),
  }))
}

function createTypicalCurve(offset: number) {
  return Array.from({ length: 24 }, (_, hour) => {
    const wave = 180 + Math.sin((hour - 7) / 3) * 120 + (hour > 18 ? 90 : 0) + offset
    return {
      hour,
      avg_load_w: Number(Math.max(52, wave).toFixed(2)),
    }
  })
}

const profileProbabilities = (
  label: keyof ClassificationProbabilities,
  confidence: number,
): ClassificationProbabilities => {
  const labels: Array<keyof ClassificationProbabilities> = [
    '外出波动型',
    '峰时集中型',
    '中高用量型',
    '高耗持续型',
    '规律低耗型',
  ]
  const rest = Number(((1 - confidence) / (labels.length - 1)).toFixed(4))
  return labels.reduce((result, item) => {
    result[item] = item === label ? confidence : rest
    return result
  }, {} as ClassificationProbabilities)
}

function createForecastSeries(day: string, base: number, peakRatio: number): ForecastSeriesPoint[] {
  const startDate = new Date(`${day}T00:00:00+08:00`)
  return Array.from({ length: 7 }, (_, index) => {
    const targetDate = new Date(startDate)
    targetDate.setDate(startDate.getDate() + index)
    const total =
      base + Math.sin(index / 1.8) * 1.2 + Math.cos(index / 2.4) * 0.6 + index * 0.18
    const peak = total * peakRatio
    return {
      date: targetDate.toISOString().slice(0, 10),
      predicted_total_kwh: Number(Math.max(1, total).toFixed(3)),
      predicted_peak_kwh: Number(Math.max(0, peak).toFixed(3)),
      predicted_valley_kwh: Number(Math.max(0, total - peak).toFixed(3)),
    }
  })
}

function createForecastWindow(day: string) {
  const startDate = new Date(`${day}T00:00:00+08:00`)
  const endDate = new Date(startDate)
  endDate.setDate(startDate.getDate() + 6)
  const endDay = endDate.toISOString().slice(0, 10)
  return {
    forecast_start: `${day}T00:00:00+08:00`,
    forecast_end: `${endDay}T00:00:00+08:00`,
  }
}

function createForecastRecord(
  id: number,
  datasetId: number,
  modelType: ForecastModelType,
  createdAt: string,
  overrides?: Partial<ForecastRecord>,
): ForecastRecord {
  const summaryBase = {
    predicted_total_kwh: 136.5,
    predicted_peak_kwh: 60.06,
    predicted_valley_kwh: 76.44,
    predicted_avg_daily_kwh: 19.5,
    predicted_peak_ratio: 0.44,
    predicted_valley_ratio: 0.56,
    risk_flags: ['evening_peak', 'high_baseload'] as const,
  }

  return {
    id,
    dataset_id: datasetId,
    model_type: modelType,
    forecast_start: '2014-12-04T00:00:00+08:00',
    forecast_end: '2014-12-10T00:00:00+08:00',
    granularity: 'daily',
    summary: {
      forecast_start: '2014-12-04T00:00:00+08:00',
      forecast_end: '2014-12-10T00:00:00+08:00',
      granularity: 'daily',
      schema_version: 'v1',
      forecast_horizon: '7d',
      model_type: modelType,
      predicted_total_kwh: summaryBase.predicted_total_kwh,
      predicted_peak_kwh: summaryBase.predicted_peak_kwh,
      predicted_valley_kwh: summaryBase.predicted_valley_kwh,
      predicted_avg_daily_kwh: summaryBase.predicted_avg_daily_kwh,
      predicted_peak_ratio: summaryBase.predicted_peak_ratio,
      predicted_valley_ratio: summaryBase.predicted_valley_ratio,
      risk_flags: [...summaryBase.risk_flags],
      confidence_hint: 'medium',
    },
    detail_path: `./outputs/forecasts/fc_${id}.json`,
    created_at: createdAt,
    ...overrides,
  }
}

export const demoSystemConfig: SystemConfig = {
  peak_valley_config: {
    peak: ['07:00-11:00', '18:00-23:00'],
    valley: ['23:00-07:00', '11:00-18:00'],
  },
  model_history_window_config: {
    classification_days: 1,
    forecast_history_days: 7,
  },
  data_upload_dir: './uploads/datasets',
  report_output_dir: './outputs/reports',
}

export const demoHealthStatus: HealthStatus = {
  status: 'up',
  service: 'go-main-service',
  peak_valley_config: demoSystemConfig.peak_valley_config,
}

export const demoDatasets: DatasetSummary[] = [
  {
    id: 1,
    name: 'REFIT House 1',
    description: '夜间负荷偏高的年度样本',
    household_id: 'house_1',
    row_count: 8760,
    time_start: '2014-01-01T00:00:00+08:00',
    time_end: '2014-12-31T23:45:00+08:00',
    status: 'ready',
    created_at: '2026-04-01T09:10:00+08:00',
    updated_at: '2026-04-01T09:18:00+08:00',
  },
  {
    id: 2,
    name: 'REFIT House 7',
    description: '白天活动更活跃的周内样本',
    household_id: 'house_7',
    row_count: 6528,
    time_start: '2014-03-01T00:00:00+08:00',
    time_end: '2014-09-30T23:45:00+08:00',
    status: 'ready',
    created_at: '2026-04-01T09:30:00+08:00',
    updated_at: '2026-04-01T09:36:00+08:00',
  },
]

export const demoDatasetDetails: Record<number, DatasetDetailPayload> = {
  1: {
    dataset: {
      ...demoDatasets[0],
      raw_file_path: './uploads/datasets/refit_house_1.csv',
      processed_file_path: './uploads/datasets/refit_house_1_15min.csv',
      feature_cols: ['Time', 'Aggregate', 'Appliance1', 'Appliance2'],
      column_mapping: {
        Time: 'timestamp',
        Aggregate: 'value',
      },
      quality_report_path: './outputs/quality/dataset_1.json',
      error_message: null,
    },
    quality_summary: {
      missing_rate: 0.012,
      duplicate_count: 2,
      sampling_interval: '15min',
      cleaning_strategies: ['缺失值插值', '异常值裁剪', '15 分钟粒度重采样'],
    },
  },
  2: {
    dataset: {
      ...demoDatasets[1],
      raw_file_path: './uploads/datasets/refit_house_7.csv',
      processed_file_path: './uploads/datasets/refit_house_7_15min.csv',
      feature_cols: ['Time', 'Aggregate', 'Appliance1', 'Appliance4'],
      column_mapping: {
        Time: 'timestamp',
        Aggregate: 'value',
      },
      quality_report_path: './outputs/quality/dataset_2.json',
      error_message: null,
    },
    quality_summary: {
      missing_rate: 0.006,
      duplicate_count: 0,
      sampling_interval: '15min',
      cleaning_strategies: ['缺失值插值', '异常值裁剪'],
    },
  },
}

export const demoAnalyses: Record<number, AnalysisPayload> = {
  1: {
    summary: {
      total_kwh: 3650.42,
      daily_avg_kwh: 10.01,
      max_load_w: 5120.5,
      max_load_time: '2014-07-12T20:00:00+08:00',
      min_load_w: 52.4,
      min_load_time: '2014-03-11T03:00:00+08:00',
      peak_kwh: 1679.19,
      valley_kwh: 1971.23,
      peak_ratio: 0.46,
      valley_ratio: 0.54,
    },
    peak_valley_config: demoSystemConfig.peak_valley_config,
    charts: {
      daily_trend: createDailyTrend(9.4, 1.4),
      weekly_trend: createWeeklyTrend(66),
      typical_day_curve: createTypicalCurve(12),
      peak_valley_pie: [
        { name: '峰时', ratio: 0.46, kwh: 1679.19 },
        { name: '谷时', ratio: 0.54, kwh: 1971.23 },
      ],
    },
    detail_path: './outputs/analysis/dataset_1.json',
    updated_at: '2026-04-01T09:22:00+08:00',
  },
  2: {
    summary: {
      total_kwh: 2480.71,
      daily_avg_kwh: 8.34,
      max_load_w: 4236.5,
      max_load_time: '2014-06-21T10:15:00+08:00',
      min_load_w: 61.2,
      min_load_time: '2014-05-01T04:00:00+08:00',
      peak_kwh: 921.32,
      valley_kwh: 1559.39,
      peak_ratio: 0.37,
      valley_ratio: 0.63,
    },
    peak_valley_config: demoSystemConfig.peak_valley_config,
    charts: {
      daily_trend: createDailyTrend(7.6, 0.9),
      weekly_trend: createWeeklyTrend(54),
      typical_day_curve: createTypicalCurve(-16),
      peak_valley_pie: [
        { name: '峰时', ratio: 0.37, kwh: 921.32 },
        { name: '谷时', ratio: 0.63, kwh: 1559.39 },
      ],
    },
    detail_path: './outputs/analysis/dataset_2.json',
    updated_at: '2026-04-01T09:38:00+08:00',
  },
}

export const demoClassifications: Record<number, ClassificationResult> = {
  1: {
    id: 8,
    dataset_id: 1,
    model_type: 'xgboost',
    schema_version: 'v1',
    predicted_label: '峰时集中型',
    confidence: 0.83,
    label_display_name: '峰时集中型',
    probabilities: profileProbabilities('峰时集中型', 0.83),
    explanation: '最近窗口峰时占比较高，晚间峰段负荷集中，适合优先做错峰提醒。',
    window_start: null,
    window_end: null,
    created_at: '2026-04-01T09:24:00+08:00',
  },
  2: {
    id: 9,
    dataset_id: 2,
    model_type: 'xgboost',
    schema_version: 'v1',
    predicted_label: '中高用量型',
    confidence: 0.78,
    label_display_name: '中高用量型',
    probabilities: profileProbabilities('中高用量型', 0.78),
    explanation: '整体用电水平处于中高区间，峰谷结构相对稳定但仍有错峰空间。',
    window_start: null,
    window_end: null,
    created_at: '2026-04-01T09:40:00+08:00',
  },
}

export const demoDetections: Record<number, DetectionResult> = {
  1: {
    id: 31,
    dataset_id: 1,
    model_type: 'iforest_rules',
    window_start: '2014-12-25T00:00:00+08:00',
    window_end: '2014-12-31T00:00:00+08:00',
    window_role: 'current',
    is_anomaly: true,
    anomaly_score: 0.7824,
    severity: 'medium',
    reasons: [
      {
        rule: 'volatility_high',
        feature: 'volatility',
        method: 'rule',
        severity: 0.7,
        reason: '最近一周相邻日期波动明显放大，超出历史常态。',
      },
      {
        rule: 'peak_ratio_shift',
        feature: 'peak_ratio',
        method: 'rule',
        severity: 0.62,
        reason: '峰时用电占比较历史窗口明显抬升，存在集中启动迹象。',
      },
    ],
    feature_summary: {
      volatility: 0.82,
      peak_ratio: 0.46,
      trend_rel: 0.31,
    },
    classification_hint: '峰时集中型',
    created_at: '2026-04-01T09:24:30+08:00',
  },
  2: {
    id: 32,
    dataset_id: 2,
    model_type: 'iforest_rules',
    window_start: '2014-09-24T00:00:00+08:00',
    window_end: '2014-09-30T00:00:00+08:00',
    window_role: 'current',
    is_anomaly: false,
    anomaly_score: 0.2184,
    severity: 'low',
    reasons: [],
    feature_summary: {
      volatility: 0.24,
      peak_ratio: 0.37,
      trend_rel: 0.08,
    },
    classification_hint: '中高用量型',
    created_at: '2026-04-01T09:40:30+08:00',
  },
}

export const demoReports: Record<number, ReportRecord[]> = {
  1: [
    {
      id: 6,
      dataset_id: 1,
      report_type: 'pdf',
      file_path: './outputs/reports/report_6.pdf',
      file_size: 128734,
      created_at: '2026-04-01T09:28:00+08:00',
    },
  ],
  2: [
    {
      id: 7,
      dataset_id: 2,
      report_type: 'pdf',
      file_path: './outputs/reports/report_7.pdf',
      file_size: 118420,
      created_at: '2026-04-01T09:43:00+08:00',
    },
  ],
}

const forecast1 = createForecastRecord(3, 1, 'lstm', '2026-04-01T10:18:00+08:00', {
  ...createForecastWindow('2015-01-01'),
  summary: {
    ...createForecastWindow('2015-01-01'),
    granularity: 'daily',
    schema_version: 'v1',
    forecast_horizon: '7d',
    model_type: 'lstm',
    predicted_total_kwh: 136.5,
    predicted_peak_kwh: 60.06,
    predicted_valley_kwh: 76.44,
    predicted_avg_daily_kwh: 19.5,
    predicted_peak_ratio: 0.44,
    predicted_valley_ratio: 0.56,
    risk_flags: ['evening_peak', 'high_baseload'],
    future_detection: {
      ...demoDetections[1],
      id: 41,
      window_role: 'future',
      window_start: '2015-01-01T00:00:00+08:00',
      window_end: '2015-01-07T00:00:00+08:00',
      severity: 'high',
      anomaly_score: 0.884,
      reasons: [
        {
          rule: 'forecast_rise',
          feature: 'trend_rel',
          method: 'rule',
          severity: 0.82,
          reason: '预测窗口呈持续上升趋势，未来负荷抬升风险较高。',
        },
      ],
    },
    confidence_hint: 'high',
  },
})
const forecast2 = createForecastRecord(4, 1, 'lstm', '2026-04-01T10:10:00+08:00', {
  ...createForecastWindow('2015-01-08'),
  summary: {
    ...createForecastWindow('2015-01-08'),
    granularity: 'daily',
    schema_version: 'v1',
    forecast_horizon: '7d',
    model_type: 'lstm',
    predicted_total_kwh: 129.15,
    predicted_peak_kwh: 54.24,
    predicted_valley_kwh: 74.91,
    predicted_avg_daily_kwh: 18.45,
    predicted_peak_ratio: 0.42,
    predicted_valley_ratio: 0.58,
    risk_flags: ['evening_peak'],
    future_detection: null,
    confidence_hint: 'medium',
  },
})
const forecast3 = createForecastRecord(5, 2, 'lstm', '2026-04-01T10:26:00+08:00', {
  ...createForecastWindow('2014-10-01'),
  summary: {
    ...createForecastWindow('2014-10-01'),
    granularity: 'daily',
    schema_version: 'v1',
    forecast_horizon: '7d',
    model_type: 'lstm',
    predicted_total_kwh: 107.73,
    predicted_peak_kwh: 37.71,
    predicted_valley_kwh: 70.02,
    predicted_avg_daily_kwh: 15.39,
    predicted_peak_ratio: 0.35,
    predicted_valley_ratio: 0.65,
    risk_flags: ['daytime_peak'],
    future_detection: null,
    confidence_hint: 'medium',
  },
})

export const demoForecasts: Record<number, ForecastRecord[]> = {
  1: [forecast1, forecast2],
  2: [forecast3],
}

export const demoForecastDetails: Record<number, ForecastDetail> = {
  3: {
    forecast: forecast1,
    series: createForecastSeries('2015-01-01', 19.5, 0.44),
  },
  4: {
    forecast: forecast2,
    series: createForecastSeries('2015-01-08', 18.45, 0.42),
  },
  5: {
    forecast: forecast3,
    series: createForecastSeries('2014-10-01', 15.39, 0.35),
  },
}

export const demoChatSessions: Record<number, ChatSession[]> = {
  1: [
    {
      id: 3,
      dataset_id: 1,
      title: 'House 1 节能建议问答',
      created_at: '2026-04-01T09:26:00+08:00',
      updated_at: '2026-04-01T09:32:00+08:00',
    },
  ],
  2: [],
}

export const demoChatMessages: Record<number, ChatMessage[]> = {
  3: [
    {
      id: 21,
      session_id: 3,
      role: 'user',
      content: '给我看一下整体情况。',
      content_path: null,
      model_name: null,
      tokens_used: null,
      created_at: '2026-04-01T09:26:00+08:00',
    },
    {
      id: 22,
      session_id: 3,
      role: 'assistant',
      content: '你的峰时占比偏高，夜间基础负荷也比较明显，建议先从晚间持续运行设备排查。',
      content_path: null,
      model_name: 'deepseek-chat',
      tokens_used: 382,
      created_at: '2026-04-01T09:26:15+08:00',
    },
  ],
}

export function buildMockImportedDataset(
  nextId: number,
  payload: ImportDatasetInput,
): DatasetDetailPayload {
  const timestamp = '2026-04-01T10:00:00+08:00'
  return {
    dataset: {
      id: nextId,
      name: payload.name,
      description: payload.description || '用户上传的本地样本',
      household_id: payload.household_id || `house_${nextId}`,
      raw_file_path: `./uploads/datasets/${payload.file_name}`,
      processed_file_path: `./uploads/datasets/${payload.file_name.replace(/\.[^.]+$/, '')}_15min.csv`,
      row_count: 384,
      time_start: '2014-10-01T00:00:00+08:00',
      time_end: '2014-10-04T23:45:00+08:00',
      feature_cols: ['Time', 'Aggregate'],
      column_mapping: payload.column_mapping ?? {
        Time: 'timestamp',
        Aggregate: 'value',
      },
      status: 'ready',
      quality_report_path: `./outputs/quality/dataset_${nextId}.json`,
      error_message: null,
      created_at: timestamp,
      updated_at: timestamp,
    },
    quality_summary: {
      missing_rate: 0.0,
      duplicate_count: 0,
      sampling_interval: '15min',
      cleaning_strategies: ['15 分钟粒度重采样', '字段自动识别'],
    },
  }
}

export function buildMockImportedAnalysis(datasetId: number): AnalysisPayload {
  return {
    summary: {
      total_kwh: 42.63,
      daily_avg_kwh: 10.65,
      max_load_w: 2310.8,
      max_load_time: '2014-10-03T20:15:00+08:00',
      min_load_w: 66.4,
      min_load_time: '2014-10-02T04:00:00+08:00',
      peak_kwh: 17.33,
      valley_kwh: 25.3,
      peak_ratio: 0.41,
      valley_ratio: 0.59,
    },
    peak_valley_config: demoSystemConfig.peak_valley_config,
    charts: {
      daily_trend: createDailyTrend(10.1, 1.1).slice(0, 4),
      weekly_trend: createWeeklyTrend(41).slice(0, 1),
      typical_day_curve: createTypicalCurve(0),
      peak_valley_pie: [
        { name: '峰时', ratio: 0.41, kwh: 17.33 },
        { name: '谷时', ratio: 0.59, kwh: 25.3 },
      ],
    },
    detail_path: `./outputs/analysis/dataset_${datasetId}.json`,
    updated_at: '2026-04-01T10:00:00+08:00',
  }
}

export function buildMockImportedForecast(
  datasetId: number,
  modelType: ForecastModelType,
  forecastId: number,
  forecastStart: string,
  forecastEnd: string,
): ForecastDetail {
  const createdAt = '2026-04-01T10:40:00+08:00'
  const datasetEnd = new Date('2014-10-04T23:45:00+08:00')
  const targetStart = new Date(forecastStart)
  const datasetDayStart = new Date(
    datasetEnd.getFullYear(),
    datasetEnd.getMonth(),
    datasetEnd.getDate(),
  )
  const targetDayStart = new Date(
    targetStart.getFullYear(),
    targetStart.getMonth(),
    targetStart.getDate(),
  )
  const dayOffset = Math.max(
    1,
    Math.round((targetDayStart.getTime() - datasetDayStart.getTime()) / (24 * 60 * 60 * 1000)),
  )
  const predictedAvgDailyKwh = Number((12 + dayOffset * 0.82).toFixed(2))
  const predictedPeakRatio = Number((0.39 + dayOffset * 0.015).toFixed(2))
  const predictedValleyRatio = Number((1 - predictedPeakRatio).toFixed(2))
  const predictedTotalKwh = Number((predictedAvgDailyKwh * 7).toFixed(2))
  const predictedPeakKwh = Number((predictedTotalKwh * predictedPeakRatio).toFixed(2))
  const predictedValleyKwh = Number((predictedTotalKwh - predictedPeakKwh).toFixed(2))
  const record = createForecastRecord(forecastId, datasetId, modelType, createdAt, {
    forecast_start: forecastStart,
    forecast_end: forecastEnd,
    summary: {
      forecast_start: forecastStart,
      forecast_end: forecastEnd,
      granularity: 'daily',
      schema_version: 'v1',
      forecast_horizon: '7d',
      model_type: modelType,
      predicted_total_kwh: predictedTotalKwh,
      predicted_peak_kwh: predictedPeakKwh,
      predicted_valley_kwh: predictedValleyKwh,
      predicted_avg_daily_kwh: predictedAvgDailyKwh,
      predicted_peak_ratio: predictedPeakRatio,
      predicted_valley_ratio: predictedValleyRatio,
      risk_flags: dayOffset >= 3 ? ['evening_peak', 'high_baseload'] : ['evening_peak'],
      confidence_hint: dayOffset >= 3 ? 'medium' : 'high',
    },
    detail_path: `./outputs/forecasts/fc_${forecastId}.json`,
  })

  return {
    forecast: record,
    series: createForecastSeries(
      forecastStart.slice(0, 10),
      predictedAvgDailyKwh,
      predictedPeakRatio,
    ),
  }
}

export function buildMockImportedReport(
  datasetId: number,
  reportType: ReportType,
  reportId: number,
): ReportRecord {
  return {
    id: reportId,
    dataset_id: datasetId,
    report_type: reportType,
    file_path: `./outputs/reports/report_${reportId}.${reportType}`,
    file_size: 1024 * 140,
    created_at: '2026-04-01T10:46:00+08:00',
  }
}

export function buildMockAssistantExchange(
  datasetId: number,
  sessionId: number,
  question: string,
  messageIdStart: number,
): {
  answer: AssistantAnswer
  messages: ChatMessage[]
} {
  const answer: AssistantAnswer = {
    session_id: sessionId,
    answer:
      datasetId === 1
        ? '你的夜间用电偏高，重点集中在 18:30 之后。建议先检查热水器、路由器和厨房持续待机设备，并把可延后任务尽量挪到谷时段。'
        : '你的日间峰值更明显，说明高功率活动主要集中在白天。建议把可错峰任务分散到谷时段，避免上午峰段叠加。 ',
    citations: [
      { key: 'peak_ratio', label: '峰时占比', value: datasetId === 1 ? 0.46 : 0.37 },
      {
        key: 'predicted_label',
        label: '行为类型',
        value: datasetId === 1 ? '峰时集中型' : '中高用量型',
      },
      {
        key: 'risk_flags',
        label: '预测风险标签',
        value: datasetId === 1 ? ['evening_peak', 'high_baseload'] : ['daytime_peak'],
      },
    ],
    actions:
      datasetId === 1
        ? ['排查夜间持续运行设备', '将热水器改为定时运行', '把洗衣与充电任务移到谷时段']
        : ['错开上午高功率任务', '优先在谷时段完成洗碗与烘干', '观察午后负荷回落情况'],
    degraded: datasetId === 1,
    error_reason: datasetId === 1 ? 'MOCK_MODE' : null,
    created_at: '2026-04-01T10:52:00+08:00',
    intent: datasetId === 1 ? 'risk' : 'advice',
    confidence_level: datasetId === 1 ? 'medium' : 'high',
	    missing_information:
	      datasetId === 1
	        ? [
            {
              key: 'comfort_priority',
              question: '你更偏向节能优先，还是舒适度优先？',
              reason: '晚间热水和持续负荷调整会影响舒适度，需要先确认偏好。',
            },
          ]
	        : [],
	  }
	const createdAt = answer.created_at ?? new Date().toISOString()

	  return {
    answer,
    messages: [
      {
        id: messageIdStart,
        session_id: sessionId,
        role: 'user',
        content: question,
        content_path: null,
        model_name: null,
        tokens_used: null,
        created_at: '2026-04-01T10:51:00+08:00',
      },
      {
        id: messageIdStart + 1,
        session_id: sessionId,
        role: 'assistant',
        content: answer.answer,
	        content_path: null,
	        model_name: 'deepseek-chat',
	        tokens_used: 486,
	        created_at: createdAt,
	      },
    ],
  }
}
