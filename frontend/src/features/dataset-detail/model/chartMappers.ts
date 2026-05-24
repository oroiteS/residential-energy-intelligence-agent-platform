import type { AnalysisPayload, ForecastDetail } from '@/types/domain'

// 图表组件只关心 label/value 结构。
// 这里把后端领域数据转换为通用图表数据，避免展示组件直接依赖后端字段细节。
export function buildDailyTrendChartData(analysis: AnalysisPayload) {
  return analysis.charts.daily_trend.map((item) => ({
    label: item.date.slice(5),
    value: item.kwh,
  }))
}

export function buildWeeklyTrendChartData(analysis: AnalysisPayload | null) {
  return analysis && analysis.charts.weekly_trend.length >= 2
    ? analysis.charts.weekly_trend.map((item) => ({
        label: `${item.week_start.slice(5)} ~ ${item.week_end.slice(5)}`,
        value: item.kwh,
      }))
    : []
}

export function buildTypicalDayChartData(analysis: AnalysisPayload) {
  return analysis.charts.typical_day_curve.map((item) => ({
    label: `${String(item.hour).padStart(2, '0')}:00`,
    value: item.avg_load_w,
  }))
}

export function buildForecastSeriesChartData(forecastDetail: ForecastDetail) {
  return forecastDetail.series.map((item) => ({
    label: item.date.slice(5),
    value: item.predicted_total_kwh,
  }))
}
