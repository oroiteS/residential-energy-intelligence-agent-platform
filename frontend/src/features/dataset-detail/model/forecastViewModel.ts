import type { ForecastSummary, PeakValleyConfig } from '@/types/domain'
import { formatDateTime } from '@/utils/formatters'

// 详情页只展示数据集结束时间之后的未来 7 天预测。
// 这个限制能避免历史预测记录过多时干扰当前分析视图。
export const maxVisibleForecastDays = 7

// 当前端无法从后端健康接口获取峰谷配置时，使用该默认配置保证页面可展示。
export const defaultPeakValleyConfig: PeakValleyConfig = {
  peak: ['07:00-11:00', '18:00-23:00'],
  valley: ['23:00-07:00', '11:00-18:00'],
}

// 根据数据集最后一个时间点构造预测窗口。
// 默认从数据集结束后的第 1 天开始，生成连续 7 天的预测区间。
export function buildDayWindow(
  timeEnd: string | null | undefined,
  dayOffset = 1,
  durationDays = 7,
) {
  const baseDate = timeEnd ? new Date(timeEnd) : new Date('2014-12-04T23:45:00+08:00')
  baseDate.setDate(baseDate.getDate() + dayOffset)
  const year = baseDate.getFullYear()
  const month = String(baseDate.getMonth() + 1).padStart(2, '0')
  const day = String(baseDate.getDate()).padStart(2, '0')
  const dayKey = `${year}-${month}-${day}`
  const endDate = new Date(baseDate)
  endDate.setDate(endDate.getDate() + durationDays - 1)
  const endYear = endDate.getFullYear()
  const endMonth = String(endDate.getMonth() + 1).padStart(2, '0')
  const endDay = String(endDate.getDate()).padStart(2, '0')
  const endDayKey = `${endYear}-${endMonth}-${endDay}`

  return {
    start: `${dayKey}T00:00:00+08:00`,
    end: `${endDayKey}T23:45:00+08:00`,
  }
}

export function getForecastDayOffset(
  forecastStart: string | null | undefined,
  datasetTimeEnd: string | null | undefined,
) {
  if (!forecastStart || !datasetTimeEnd) {
    return null
  }

  const forecastDate = new Date(forecastStart)
  const datasetDate = new Date(datasetTimeEnd)
  const dayStartForecast = new Date(
    forecastDate.getFullYear(),
    forecastDate.getMonth(),
    forecastDate.getDate(),
  )
  const dayStartDataset = new Date(
    datasetDate.getFullYear(),
    datasetDate.getMonth(),
    datasetDate.getDate(),
  )
  // 只比较自然日，避免具体小时分钟影响“未来第几天”的展示判断。
  const dayOffset = Math.round(
    (dayStartForecast.getTime() - dayStartDataset.getTime()) / (24 * 60 * 60 * 1000),
  )

  if (dayOffset < 1 || dayOffset > maxVisibleForecastDays) {
    return null
  }
  return dayOffset
}

export function getForecastWindowLabel(summary: ForecastSummary | null | undefined) {
  if (summary?.forecast_horizon === '7d') {
    return '未来 7 天'
  }
  return '预测窗口'
}

export function getForecastRangeLabel(
  forecastStart: string | null | undefined,
  forecastEnd: string | null | undefined,
) {
  if (!forecastStart || !forecastEnd) {
    return '--'
  }
  return `${formatDateTime(forecastStart).slice(0, 10)} ~ ${formatDateTime(forecastEnd).slice(5, 10)}`
}

export function getPredictedTotalKwh(summary: ForecastSummary) {
  if (summary.predicted_total_kwh !== undefined && summary.predicted_total_kwh !== null) {
    return summary.predicted_total_kwh
  }
  return 0
}
