import { useCallback, useEffect, useMemo, useState } from 'react'
import { message } from 'antd'
import {
  exportDatasetReport,
  extractApiErrorMessage,
  fetchClassifications,
  fetchCurrentDetection,
  fetchDatasetAnalysis,
  fetchDatasetDetail,
  fetchForecastDetail,
  fetchForecasts,
  fetchHealth,
  fetchReports,
  runClassification,
  runDetection,
  runForecast,
} from '@/services/dashboard'
import type {
  AnalysisPayload,
  ClassificationResult,
  DatasetDetailPayload,
  DetectionResult,
  ForecastDetail,
  ForecastModelType,
  ForecastRecord,
  PeakValleyConfig,
  ReportRecord,
  ReportType,
} from '@/types/domain'
import {
  buildDayWindow,
  defaultPeakValleyConfig,
  getForecastDayOffset,
} from '../model/forecastViewModel'

// 数据集详情页的数据调度层。
// 页面需要同时展示基础信息、分析图表、分类、异常检测、预测和报告；
// 这些数据来自多个接口，因此集中放在一个 hook 中统一管理加载状态、错误提示和刷新动作。
export function useDatasetDetailPage(datasetId: number) {
  const [loading, setLoading] = useState(true)
  const [forecastDetailLoading, setForecastDetailLoading] = useState(false)
  const [forecastActionLoading, setForecastActionLoading] = useState(false)
  const [classificationActionLoading, setClassificationActionLoading] = useState(false)
  const [detectionActionLoading, setDetectionActionLoading] = useState(false)
  const [reportActionLoading, setReportActionLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [detail, setDetail] = useState<DatasetDetailPayload | null>(null)
  const [analysis, setAnalysis] = useState<AnalysisPayload | null>(null)
  const [classification, setClassification] = useState<ClassificationResult | null>(null)
  const [classificationHistory, setClassificationHistory] = useState<ClassificationResult[]>([])
  const [detection, setDetection] = useState<DetectionResult | null>(null)
  const [reports, setReports] = useState<ReportRecord[]>([])
  const [forecasts, setForecasts] = useState<ForecastRecord[]>([])
  const [activeForecastId, setActiveForecastId] = useState<number | null>(null)
  const [forecastDetail, setForecastDetail] = useState<ForecastDetail | null>(null)
  const [selectedForecastModel, setSelectedForecastModel] = useState<ForecastModelType>('lstm')
  const [runtimePeakValleyConfig, setRuntimePeakValleyConfig] = useState<PeakValleyConfig>(
    defaultPeakValleyConfig,
  )

  // 首次进入详情页时，先加载数据集基础信息。
  // 只有数据集状态为 ready 时，才继续并行加载分析、分类、检测、报告和预测结果。
  const loadDetail = useCallback(async () => {
    if (!Number.isFinite(datasetId)) {
      setError('数据集编号无效。')
      setLoading(false)
      return
    }

    setLoading(true)
    setError(null)
    try {
      const detailResult = await fetchDatasetDetail(datasetId)
      setDetail(detailResult)
      // 峰谷时段配置属于系统级配置。
      // 如果健康接口暂时不可用，则使用前端默认配置，避免详情页整体不可用。
      try {
        const healthResult = await fetchHealth()
        if (healthResult.peak_valley_config) {
          setRuntimePeakValleyConfig(healthResult.peak_valley_config)
        }
      } catch {
        setRuntimePeakValleyConfig(defaultPeakValleyConfig)
      }

      if (detailResult.dataset.status !== 'ready') {
        setAnalysis(null)
        setClassification(null)
        setClassificationHistory([])
        setDetection(null)
        setReports([])
        setForecasts([])
        setActiveForecastId(null)
        setSelectedForecastModel('lstm')
        return
      }

      // 详情页主体数据彼此独立，可以并行请求，减少页面等待时间。
      const [
        analysisResult,
        classificationResults,
        detectionResult,
        reportResult,
        forecastResult,
      ] = await Promise.all([
        fetchDatasetAnalysis(datasetId),
        fetchClassifications(datasetId),
        fetchCurrentDetection(datasetId),
        fetchReports(datasetId),
        fetchForecasts(datasetId),
      ])

      setAnalysis(analysisResult)
      setClassification(classificationResults[0] ?? null)
      setClassificationHistory(classificationResults)
      setDetection(detectionResult)
      setReports(reportResult)
      setForecasts(forecastResult)
      setActiveForecastId(forecastResult[0]?.id ?? null)
      setSelectedForecastModel(forecastResult[0]?.model_type ?? 'lstm')
    } catch {
      setError('数据集详情加载失败，请稍后重试。')
    } finally {
      setLoading(false)
    }
  }, [datasetId])

  // 预测详情依赖当前选中的预测记录。
  // active 标记用于避免组件卸载或快速切换时继续写入过期请求结果。
  useEffect(() => {
    void loadDetail()
  }, [loadDetail])

  // 预测列表只展示数据集结束日期之后的未来窗口。
  // 如果当前选中的预测记录不在可见范围内，则自动切换到第一条可见记录。
  useEffect(() => {
    if (!activeForecastId) {
      setForecastDetail(null)
      return
    }

    let active = true

    const loadForecastDetail = async () => {
      setForecastDetailLoading(true)
      try {
        const result = await fetchForecastDetail(activeForecastId)
        if (active) {
          setForecastDetail(result)
        }
      } catch (error) {
        if (active) {
          message.error(extractApiErrorMessage(error, '预测详情加载失败。'))
        }
      } finally {
        if (active) {
          setForecastDetailLoading(false)
        }
      }
    }

    void loadForecastDetail()

    return () => {
      active = false
    }
  }, [activeForecastId])

  useEffect(() => {
    const visibleForecast = forecasts.find((item) =>
      getForecastDayOffset(item.forecast_start, detail?.dataset.time_end ?? null) !== null,
    )
    if (!visibleForecast) {
      return
    }
    const activeForecastVisible =
      activeForecastId &&
      forecasts.some(
        (item) =>
          item.id === activeForecastId &&
          getForecastDayOffset(item.forecast_start, detail?.dataset.time_end ?? null) !== null,
      )

    if (!activeForecastVisible) {
      setActiveForecastId(visibleForecast.id)
    }
  }, [activeForecastId, detail?.dataset.time_end, forecasts])

  const refreshForecasts = useCallback(async () => {
    const items = await fetchForecasts(datasetId)
    setForecasts(items)
    return items
  }, [datasetId])

  // 可见预测记录按“未来第几天”排序，同一天内优先展示最新生成的结果。
  const visibleForecasts = useMemo(
    () =>
      forecasts
        .filter(
          (item) =>
            getForecastDayOffset(item.forecast_start, detail?.dataset.time_end ?? null) !== null,
        )
        .sort((left, right) => {
          const leftOffset =
            getForecastDayOffset(left.forecast_start, detail?.dataset.time_end ?? null) ?? 99
          const rightOffset =
            getForecastDayOffset(right.forecast_start, detail?.dataset.time_end ?? null) ?? 99
          if (leftOffset !== rightOffset) {
            return leftOffset - rightOffset
          }
          return new Date(right.created_at).getTime() - new Date(left.created_at).getTime()
        }),
    [detail?.dataset.time_end, forecasts],
  )

  const selectedForecast =
    visibleForecasts.find((item) => item.id === activeForecastId) ?? null

  // 分析结果可能携带峰谷配置；如果分析结果没有提供，则回退到系统配置或默认配置。
  const peakValleyConfig = {
    peak: analysis?.peak_valley_config.peak.length
      ? analysis.peak_valley_config.peak
      : runtimePeakValleyConfig.peak,
    valley: analysis?.peak_valley_config.valley.length
      ? analysis.peak_valley_config.valley
      : runtimePeakValleyConfig.valley,
  }

  // 生成预测时以前端计算出的未来 7 天窗口作为请求参数。
  // 生成完成后刷新预测列表，并把当前视图切到最新预测结果。
  const handleGenerateForecast = async () => {
    if (!detail) {
      return
    }

    const window = buildDayWindow(detail.dataset.time_end)
    setForecastActionLoading(true)
    try {
      const result = await runForecast(datasetId, {
        model_type: selectedForecastModel,
        granularity: 'daily',
        forecast_start: window.start,
        forecast_end: window.end,
        force_refresh: true,
      })
      const nextForecasts = await refreshForecasts()
      setSelectedForecastModel(result.model_type)
      setActiveForecastId(result.id ?? nextForecasts[0]?.id ?? null)
      message.success('预测任务已完成并刷新展示。')
    } catch (error) {
      message.error(extractApiErrorMessage(error, '生成预测失败，请稍后重试。'))
    } finally {
      setForecastActionLoading(false)
    }
  }

  // 分类、检测和报告导出都是用户主动触发的刷新动作；
  // 每个动作单独维护 loading 状态，避免互相阻塞。
  const handleRunClassification = async () => {
    setClassificationActionLoading(true)
    try {
      await runClassification(datasetId)
      const results = await fetchClassifications(datasetId)
      setClassification(results[0] ?? null)
      setClassificationHistory(results)
      message.success('行为分类结果已刷新。')
    } catch (error) {
      message.error(extractApiErrorMessage(error, '生成分类结果失败，请稍后重试。'))
    } finally {
      setClassificationActionLoading(false)
    }
  }

  const handleRunDetection = async () => {
    setDetectionActionLoading(true)
    try {
      const result = await runDetection(datasetId)
      setDetection(result)
      message.success('异常检测结果已刷新。')
    } catch (error) {
      message.error(extractApiErrorMessage(error, '生成异常检测结果失败，请稍后重试。'))
    } finally {
      setDetectionActionLoading(false)
    }
  }

  const handleExportReport = async (reportType: ReportType) => {
    setReportActionLoading(true)
    try {
      await exportDatasetReport(datasetId, reportType)
      setReports(await fetchReports(datasetId))
      message.success('PDF导出任务已创建。')
    } catch (error) {
      message.error(extractApiErrorMessage(error, '导出报告失败。'))
    } finally {
      setReportActionLoading(false)
    }
  }

  return {
    activeForecastId,
    analysis,
    classification,
    classificationActionLoading,
    classificationHistory,
    detail,
    detection,
    detectionActionLoading,
    error,
    forecastActionLoading,
    forecastDetail,
    forecastDetailLoading,
    handleExportReport,
    handleGenerateForecast,
    handleRunClassification,
    handleRunDetection,
    loadDetail,
    loading,
    peakValleyConfig,
    reportActionLoading,
    reports,
    selectedForecast,
    selectedForecastModel,
    setActiveForecastId,
    setSelectedForecastModel,
    visibleForecasts,
  }
}
