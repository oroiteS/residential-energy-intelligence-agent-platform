import type {
  AdviceDetail,
  AnalysisPayload,
  AssistantAnswer,
  ChatMessage,
  ChatSession,
  ClassificationResult,
  DatasetDetailPayload,
  DatasetSummary,
  ForecastDetail,
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

function createForecastSeries(day: string, base: number, peakBoost: number) {
  return Array.from({ length: 96 }, (_, index) => {
    const hour = Math.floor(index / 4)
    const minute = (index % 4) * 15
    const eveningBoost = hour >= 18 && hour <= 21 ? peakBoost : 0
    const predicted =
      base + Math.sin((hour - 8) / 3) * 120 + Math.cos(index / 8) * 40 + eveningBoost
    return {
      timestamp: `${day}T${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}:00+08:00`,
      predicted: Number(Math.max(180, predicted).toFixed(3)),
    }
  })
}

function createForecastWindow(day: string) {
  return {
    forecast_start: `${day}T00:00:00+08:00`,
    forecast_end: `${day}T23:45:00+08:00`,
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
    predicted_avg_load_w: 812.4,
    predicted_peak_load_w: 1658.2,
    predicted_peak_ratio: 0.44,
    predicted_valley_ratio: 0.2,
    predicted_flat_ratio: 0.36,
    risk_flags: ['evening_peak', 'high_baseload'] as const,
  }

  return {
    id,
    dataset_id: datasetId,
    model_type: modelType,
    forecast_start: '2014-12-04T00:00:00+08:00',
    forecast_end: '2014-12-04T23:45:00+08:00',
    granularity: '15min',
    summary: {
      forecast_start: '2014-12-04T00:00:00+08:00',
      forecast_end: '2014-12-04T23:45:00+08:00',
      granularity: '15min',
      schema_version: 'v1',
      forecast_horizon: '1d',
      predicted_avg_load_w: summaryBase.predicted_avg_load_w,
      predicted_peak_load_w: summaryBase.predicted_peak_load_w,
      predicted_total_kwh: Number(((summaryBase.predicted_avg_load_w * 24) / 1000).toFixed(2)),
      peak_period: '2014-12-04T18:30:00+08:00/2014-12-04T21:30:00+08:00',
      forecast_peak_periods: ['2014-12-04T18:30:00+08:00/2014-12-04T21:30:00+08:00'],
      predicted_peak_ratio: summaryBase.predicted_peak_ratio,
      predicted_valley_ratio: summaryBase.predicted_valley_ratio,
      predicted_flat_ratio: summaryBase.predicted_flat_ratio,
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
    valley: ['23:00-07:00'],
  },
  model_history_window_config: {
    classification_days: 1,
    forecast_history_days: 7,
  },
  energy_advice_prompt_template:
    '这是居民过去{{history_days}}天的实际用电情况、未来一段时间的预测用电情况，以及居民用电行为分类。请基于统计分析结果、历史用电摘要、未来预测摘要和分类结果，给出具体、可执行、可解释的节能建议，并指出关键依据。',
  data_upload_dir: './uploads/datasets',
  report_output_dir: './outputs/reports',
}

export const demoHealthStatus: HealthStatus = {
  status: 'up',
  service: 'go-main-service',
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
      valley_kwh: 693.58,
      flat_kwh: 1277.65,
      peak_ratio: 0.46,
      valley_ratio: 0.19,
      flat_ratio: 0.35,
    },
    peak_valley_config: demoSystemConfig.peak_valley_config,
    charts: {
      daily_trend: createDailyTrend(9.4, 1.4),
      weekly_trend: createWeeklyTrend(66),
      typical_day_curve: createTypicalCurve(12),
      peak_valley_pie: [
        { name: '峰时', ratio: 0.46, kwh: 1679.19 },
        { name: '谷时', ratio: 0.19, kwh: 693.58 },
        { name: '平时', ratio: 0.35, kwh: 1277.65 },
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
      valley_kwh: 481.52,
      flat_kwh: 1077.87,
      peak_ratio: 0.37,
      valley_ratio: 0.19,
      flat_ratio: 0.44,
    },
    peak_valley_config: demoSystemConfig.peak_valley_config,
    charts: {
      daily_trend: createDailyTrend(7.6, 0.9),
      weekly_trend: createWeeklyTrend(54),
      typical_day_curve: createTypicalCurve(-16),
      peak_valley_pie: [
        { name: '峰时', ratio: 0.37, kwh: 921.32 },
        { name: '谷时', ratio: 0.19, kwh: 481.52 },
        { name: '平时', ratio: 0.44, kwh: 1077.87 },
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
    predicted_label: 'day_low_night_high',
    confidence: 0.83,
    label_display_name: '晚上高峰型',
    probabilities: {
      morning_peak: 0.07,
      afternoon_peak: 0.06,
      day_low_night_high: 0.83,
      all_day_low: 0.04,
    },
    explanation: '夜间均值显著高于白天均值，且 night_mean/day_mean = 1.46。',
    window_start: null,
    window_end: null,
    created_at: '2026-04-01T09:24:00+08:00',
  },
  2: {
    id: 9,
    dataset_id: 2,
    model_type: 'xgboost',
    schema_version: 'v1',
    predicted_label: 'afternoon_peak',
    confidence: 0.78,
    label_display_name: '下午高峰型',
    probabilities: {
      morning_peak: 0.06,
      afternoon_peak: 0.78,
      day_low_night_high: 0.11,
      all_day_low: 0.05,
    },
    explanation: '白天时段平均负荷明显高于夜间，day_mean/night_mean = 1.28。',
    window_start: null,
    window_end: null,
    created_at: '2026-04-01T09:40:00+08:00',
  },
}

export const demoAdvices: Record<number, AdviceDetail[]> = {
  1: [
    {
      advice: {
        id: 12,
        dataset_id: 1,
        classification_id: 8,
        advice_type: 'rule',
        summary: '将洗衣与热水器运行时段调整到谷时段',
        content_path: './outputs/advices/advice_12.json',
        created_at: '2026-04-01T09:25:00+08:00',
      },
      content: {
        items: [
          {
            reason: '峰时占比 46%，高于建议阈值 40%',
            action: '将洗衣、充电等任务改到谷时段执行。',
          },
          {
            reason: '夜间持续负荷偏高，分类结果为“晚上高峰型”',
            action: '检查客厅和厨房晚间持续待机设备，优先排查热水器与路由器周边电器。',
          },
        ],
      },
    },
  ],
  2: [
    {
      advice: {
        id: 15,
        dataset_id: 2,
        classification_id: 9,
        advice_type: 'rule',
        summary: '将高功率任务避开工作日上午峰段',
        content_path: './outputs/advices/advice_15.json',
        created_at: '2026-04-01T09:41:00+08:00',
      },
      content: {
        items: [
          {
            reason: '上午工作时段负荷抬升明显',
            action: '将洗碗机、烘干机等可延后任务安排到午后或谷时段。',
          },
        ],
      },
    },
  ],
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

const forecast1 = createForecastRecord(3, 1, 'tft', '2026-04-01T10:18:00+08:00', {
  ...createForecastWindow('2015-01-01'),
  summary: {
    ...createForecastWindow('2015-01-01'),
    granularity: '15min',
    schema_version: 'v1',
    forecast_horizon: '1d',
    predicted_avg_load_w: 812.4,
    predicted_peak_load_w: 1658.2,
    predicted_total_kwh: 19.5,
    peak_period: '2015-01-01T18:30:00+08:00/2015-01-01T21:30:00+08:00',
    forecast_peak_periods: ['2015-01-01T18:30:00+08:00/2015-01-01T21:30:00+08:00'],
    predicted_peak_ratio: 0.44,
    predicted_valley_ratio: 0.2,
    predicted_flat_ratio: 0.36,
    risk_flags: ['evening_peak', 'high_baseload'],
    confidence_hint: 'high',
  },
})
const forecast2 = createForecastRecord(4, 1, 'tft', '2026-04-01T10:10:00+08:00', {
  ...createForecastWindow('2015-01-02'),
  summary: {
    ...createForecastWindow('2015-01-02'),
    granularity: '15min',
    schema_version: 'v1',
    forecast_horizon: '1d',
    predicted_avg_load_w: 768.9,
    predicted_peak_load_w: 1492.6,
    predicted_total_kwh: 18.45,
    peak_period: '2015-01-02T18:15:00+08:00/2015-01-02T21:15:00+08:00',
    forecast_peak_periods: ['2015-01-02T18:15:00+08:00/2015-01-02T21:15:00+08:00'],
    predicted_peak_ratio: 0.42,
    predicted_valley_ratio: 0.22,
    predicted_flat_ratio: 0.36,
    risk_flags: ['evening_peak'],
    confidence_hint: 'medium',
  },
})
const forecast3 = createForecastRecord(5, 2, 'tft', '2026-04-01T10:26:00+08:00', {
  ...createForecastWindow('2014-10-01'),
  summary: {
    ...createForecastWindow('2014-10-01'),
    granularity: '15min',
    schema_version: 'v1',
    forecast_horizon: '1d',
    predicted_avg_load_w: 641.2,
    predicted_peak_load_w: 1268.5,
    predicted_total_kwh: 15.39,
    peak_period: '2014-10-01T09:00:00+08:00/2014-10-01T11:00:00+08:00',
    forecast_peak_periods: ['2014-10-01T09:00:00+08:00/2014-10-01T11:00:00+08:00'],
    predicted_peak_ratio: 0.35,
    predicted_valley_ratio: 0.22,
    predicted_flat_ratio: 0.43,
    risk_flags: ['daytime_peak'],
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
    series: createForecastSeries('2015-01-01', 480, 90),
  },
  4: {
    forecast: forecast2,
    series: createForecastSeries('2015-01-02', 450, 70),
  },
  5: {
    forecast: forecast3,
    series: createForecastSeries('2014-10-01', 390, 60),
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
      valley_kwh: 8.02,
      flat_kwh: 17.28,
      peak_ratio: 0.41,
      valley_ratio: 0.19,
      flat_ratio: 0.4,
    },
    peak_valley_config: demoSystemConfig.peak_valley_config,
    charts: {
      daily_trend: createDailyTrend(10.1, 1.1).slice(0, 4),
      weekly_trend: createWeeklyTrend(41).slice(0, 1),
      typical_day_curve: createTypicalCurve(0),
      peak_valley_pie: [
        { name: '峰时', ratio: 0.41, kwh: 17.33 },
        { name: '谷时', ratio: 0.19, kwh: 8.02 },
        { name: '平时', ratio: 0.4, kwh: 17.28 },
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
  const predictedAvgLoad = 500 + dayOffset * 34
  const predictedPeakLoad = 860 + dayOffset * 84
  const predictedPeakRatio = Number((0.39 + dayOffset * 0.015).toFixed(2))
  const predictedValleyRatio = Number((0.21 + dayOffset * 0.005).toFixed(2))
  const predictedFlatRatio = Number(
    Math.max(0.15, 1 - predictedPeakRatio - predictedValleyRatio).toFixed(2),
  )
  const record = createForecastRecord(forecastId, datasetId, modelType, createdAt, {
    forecast_start: forecastStart,
    forecast_end: forecastEnd,
    summary: {
      forecast_start: forecastStart,
      forecast_end: forecastEnd,
      granularity: '15min',
      schema_version: 'v1',
      forecast_horizon: '1d',
      predicted_avg_load_w: predictedAvgLoad,
      predicted_peak_load_w: predictedPeakLoad,
      predicted_total_kwh: Number(((predictedAvgLoad * 24) / 1000).toFixed(2)),
      peak_period: `${forecastStart.slice(0, 10)}T${dayOffset === 1 ? '18:15:00+08:00' : dayOffset === 2 ? '19:00:00+08:00' : '20:00:00+08:00'}/${forecastStart.slice(0, 10)}T${dayOffset === 1 ? '21:00:00+08:00' : dayOffset === 2 ? '21:45:00+08:00' : '22:30:00+08:00'}`,
      forecast_peak_periods: [
        `${forecastStart.slice(0, 10)}T${dayOffset === 1 ? '18:15:00+08:00' : dayOffset === 2 ? '19:00:00+08:00' : '20:00:00+08:00'}/${forecastStart.slice(0, 10)}T${dayOffset === 1 ? '21:00:00+08:00' : dayOffset === 2 ? '21:45:00+08:00' : '22:30:00+08:00'}`,
      ],
      predicted_peak_ratio: predictedPeakRatio,
      predicted_valley_ratio: predictedValleyRatio,
      predicted_flat_ratio: predictedFlatRatio,
      risk_flags: dayOffset >= 3 ? ['evening_peak', 'high_baseload'] : ['evening_peak'],
      confidence_hint: dayOffset >= 3 ? 'medium' : 'high',
    },
    detail_path: `./outputs/forecasts/fc_${forecastId}.json`,
  })

  return {
    forecast: record,
    series: createForecastSeries(
      forecastStart.slice(0, 10),
      predictedAvgLoad,
      80 + dayOffset * 12,
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
        : '你的日间峰值更明显，说明高功率活动主要集中在白天。建议把可错峰任务分散到平时段，避免上午峰段叠加。 ',
    citations: [
      { key: 'peak_ratio', label: '峰时占比', value: datasetId === 1 ? 0.46 : 0.37 },
      {
        key: 'predicted_label',
        label: '行为类型',
        value: datasetId === 1 ? 'day_low_night_high' : 'afternoon_peak',
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
        : ['错开上午高功率任务', '优先在平时段完成洗碗与烘干', '观察午后负荷回落情况'],
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
        created_at: answer.created_at,
      },
    ],
  }
}
