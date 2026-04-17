import { useCallback, useEffect, useState } from 'react'
import {
  ArrowLeftOutlined,
  DownloadOutlined,
  FilePdfOutlined,
  LineChartOutlined,
  RocketOutlined,
  RobotOutlined,
} from '@ant-design/icons'
import {
  Alert,
  Button,
  Col,
  Descriptions,
  Empty,
  List,
  Progress,
  Row,
  Select,
  Space,
  Spin,
  Tabs,
  Tag,
  Typography,
  message,
} from 'antd'
import { useNavigate, useParams } from 'react-router-dom'
import { PeakRatioChart } from '@/components/charts/PeakRatioChart'
import { TrendChart } from '@/components/charts/TrendChart'
import { PageHero } from '@/components/common/PageHero'
import { SectionCard } from '@/components/common/SectionCard'
import { StatusTag } from '@/components/common/StatusTag'
import { MetricCard } from '@/components/sections/MetricCard'
import {
  adviceTypeMap,
  classificationLabelMap,
  forecastModelMap,
  reportTypeMap,
  riskFlagMap,
} from '@/constants/display'
import {
  downloadReport,
  extractApiErrorMessage,
  exportDatasetReport,
  fetchAdvices,
  fetchClassifications,
  fetchDatasetAnalysis,
  fetchDatasetDetail,
  fetchForecastDetail,
  fetchForecasts,
  fetchReports,
  generateAdvices,
  runClassification,
  runForecast,
} from '@/services/dashboard'
import type {
  AdviceDetail,
  AnalysisPayload,
  ClassificationResult,
  DatasetDetailPayload,
  ForecastDetail,
  ForecastModelType,
  ForecastRecord,
  ForecastSummary,
  PeakValleyConfig,
  ReportRecord,
  ReportType,
} from '@/types/domain'
import {
  formatDateTime,
  formatFileLabel,
  formatFileSize,
  formatNumber,
  formatPercent,
  formatPeriodRange,
  formatTime,
} from '@/utils/formatters'

const maxVisibleForecastDays = 7

function buildDayWindow(timeEnd: string | null | undefined, dayOffset = 0) {
  const baseDate = timeEnd ? new Date(timeEnd) : new Date('2014-12-04T23:45:00+08:00')
  baseDate.setDate(baseDate.getDate() + dayOffset)
  const year = baseDate.getFullYear()
  const month = String(baseDate.getMonth() + 1).padStart(2, '0')
  const day = String(baseDate.getDate()).padStart(2, '0')
  const dayKey = `${year}-${month}-${day}`

  return {
    start: `${dayKey}T00:00:00+08:00`,
    end: `${dayKey}T23:45:00+08:00`,
  }
}

function getForecastDayOffset(
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
  const dayOffset = Math.round(
    (dayStartForecast.getTime() - dayStartDataset.getTime()) / (24 * 60 * 60 * 1000),
  )

  if (dayOffset < 1 || dayOffset > maxVisibleForecastDays) {
    return null
  }
  return dayOffset
}

function formatClassificationDay(item: ClassificationResult) {
  const source = item.window_start || item.created_at
  if (!source) {
    return '未知日期'
  }
  return formatDateTime(source).slice(0, 10)
}

function getForecastDayLabel(
  forecastStart: string | null | undefined,
  datasetTimeEnd: string | null | undefined,
) {
  const dayOffset = getForecastDayOffset(forecastStart, datasetTimeEnd)
  if (dayOffset === null) {
    return '历史记录'
  }
  return `未来第 ${dayOffset} 天`
}

type MinuteRange = {
  start: number
  end: number
}

function parseClockMinute(raw: string) {
  const parts = raw.trim().split(':')
  if (parts.length !== 2) {
    return null
  }

  const hour = Number(parts[0])
  const minute = Number(parts[1])
  if (
    !Number.isInteger(hour) ||
    !Number.isInteger(minute) ||
    hour < 0 ||
    hour > 23 ||
    minute < 0 ||
    minute > 59
  ) {
    return null
  }

  return hour * 60 + minute
}

function expandPeriodToRanges(period: string): MinuteRange[] {
  const parts = period.split('-')
  if (parts.length !== 2) {
    return []
  }

  const start = parseClockMinute(parts[0])
  const end = parseClockMinute(parts[1])
  if (start === null || end === null) {
    return []
  }

  if (start === end) {
    return [{ start: 0, end: 24 * 60 }]
  }
  if (start < end) {
    return [{ start, end }]
  }
  return [
    { start, end: 24 * 60 },
    { start: 0, end },
  ]
}

function mergeMinuteRanges(ranges: MinuteRange[]) {
  if (ranges.length === 0) {
    return []
  }

  const sorted = [...ranges].sort((left, right) => left.start - right.start)
  const merged: MinuteRange[] = [sorted[0]]

  for (const current of sorted.slice(1)) {
    const last = merged[merged.length - 1]
    if (current.start <= last.end) {
      last.end = Math.max(last.end, current.end)
      continue
    }
    merged.push({ ...current })
  }

  return merged
}

function formatMinute(minute: number) {
  const normalized = ((minute % (24 * 60)) + 24 * 60) % (24 * 60)
  const hour = Math.floor(normalized / 60)
  const remain = normalized % 60
  return `${String(hour).padStart(2, '0')}:${String(remain).padStart(2, '0')}`
}

function buildFlatPeriods(config: PeakValleyConfig) {
  const occupied = mergeMinuteRanges(
    [...config.peak, ...config.valley].flatMap((period) => expandPeriodToRanges(period)),
  )
  if (occupied.length === 0) {
    return ['00:00-24:00']
  }

  const flat: MinuteRange[] = []
  let cursor = 0
  for (const range of occupied) {
    if (cursor < range.start) {
      flat.push({ start: cursor, end: range.start })
    }
    cursor = Math.max(cursor, range.end)
  }
  if (cursor < 24 * 60) {
    flat.push({ start: cursor, end: 24 * 60 })
  }

  return flat
    .filter((range) => range.end > range.start)
    .map((range) => `${formatMinute(range.start)}-${range.end === 24 * 60 ? '24:00' : formatMinute(range.end)}`)
}

function getForecastPeakPeriods(summary: ForecastSummary) {
  if (summary.forecast_peak_periods?.length) {
    return summary.forecast_peak_periods
  }
  if (summary.peak_period) {
    return [summary.peak_period]
  }
  return []
}

function getPredictedTotalKwh(summary: ForecastSummary) {
  if (summary.predicted_total_kwh !== undefined && summary.predicted_total_kwh !== null) {
    return summary.predicted_total_kwh
  }
  return (summary.predicted_avg_load_w * 24) / 1000
}

export function DatasetDetailPage() {
  const navigate = useNavigate()
  const params = useParams()
  const datasetId = Number(params.datasetId)

  const [loading, setLoading] = useState(true)
  const [forecastDetailLoading, setForecastDetailLoading] = useState(false)
  const [forecastActionLoading, setForecastActionLoading] = useState(false)
  const [classificationActionLoading, setClassificationActionLoading] = useState(false)
  const [adviceActionLoading, setAdviceActionLoading] = useState(false)
  const [reportActionLoading, setReportActionLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [detail, setDetail] = useState<DatasetDetailPayload | null>(null)
  const [analysis, setAnalysis] = useState<AnalysisPayload | null>(null)
  const [classification, setClassification] = useState<ClassificationResult | null>(null)
  const [classificationHistory, setClassificationHistory] = useState<ClassificationResult[]>([])
  const [advices, setAdvices] = useState<AdviceDetail[]>([])
  const [reports, setReports] = useState<ReportRecord[]>([])
  const [forecasts, setForecasts] = useState<ForecastRecord[]>([])
  const [activeForecastId, setActiveForecastId] = useState<number | null>(null)
  const [forecastDetail, setForecastDetail] = useState<ForecastDetail | null>(null)
  const [selectedForecastModel, setSelectedForecastModel] = useState<ForecastModelType>('tft')
  const [selectedFutureDay, setSelectedFutureDay] = useState(1)

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

      if (detailResult.dataset.status !== 'ready') {
        setAnalysis(null)
        setClassification(null)
        setClassificationHistory([])
        setAdvices([])
        setReports([])
        setForecasts([])
        setActiveForecastId(null)
        setSelectedForecastModel('tft')
        return
      }

      const [
        analysisResult,
        classificationResults,
        adviceResult,
        reportResult,
        forecastResult,
      ] = await Promise.all([
        fetchDatasetAnalysis(datasetId),
        fetchClassifications(datasetId),
        fetchAdvices(datasetId),
        fetchReports(datasetId),
        fetchForecasts(datasetId),
      ])

      setAnalysis(analysisResult)
      setClassification(classificationResults[0] ?? null)
      setClassificationHistory(classificationResults)
      setAdvices(adviceResult)
      setReports(reportResult)
      setForecasts(forecastResult)
      setActiveForecastId(forecastResult[0]?.id ?? null)
      setSelectedForecastModel(forecastResult[0]?.model_type ?? 'tft')
    } catch {
      setError('数据集详情加载失败，请稍后重试。')
    } finally {
      setLoading(false)
    }
  }, [datasetId])

  useEffect(() => {
    void loadDetail()
  }, [loadDetail])

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
    if (!activeForecastId || !forecasts.some((item) => item.id === activeForecastId && getForecastDayOffset(item.forecast_start, detail?.dataset.time_end ?? null) !== null)) {
      setActiveForecastId(visibleForecast.id)
    }
  }, [activeForecastId, detail?.dataset.time_end, forecasts])

  const refreshForecasts = async () => {
    const items = await fetchForecasts(datasetId)
    setForecasts(items)
    return items
  }

  const refreshAdvices = async () => {
    const result = await fetchAdvices(datasetId)
    setAdvices(result)
    return result
  }

  const visibleForecasts = forecasts.filter(
    (item) => getForecastDayOffset(item.forecast_start, detail?.dataset.time_end ?? null) !== null,
  ).sort((left, right) => {
    const leftOffset =
      getForecastDayOffset(left.forecast_start, detail?.dataset.time_end ?? null) ?? 99
    const rightOffset =
      getForecastDayOffset(right.forecast_start, detail?.dataset.time_end ?? null) ?? 99
    if (leftOffset !== rightOffset) {
      return leftOffset - rightOffset
    }
    return new Date(right.created_at).getTime() - new Date(left.created_at).getTime()
  })
  const selectedForecast = visibleForecasts.find((item) => item.id === activeForecastId) ?? null
  const weeklyTrendChartData =
    analysis && analysis.charts.weekly_trend.length >= 2
      ? analysis.charts.weekly_trend.map((item) => ({
          label: `${item.week_start.slice(5)} ~ ${item.week_end.slice(5)}`,
          value: item.kwh,
        }))
      : []
  const flatPeriods = analysis ? buildFlatPeriods(analysis.peak_valley_config) : []

  if (loading) {
    return (
      <div className="page-state">
        <Spin size="large" />
      </div>
    )
  }

  if (error || !detail) {
    return (
      <div className="page-state">
        <Alert
          type="error"
          showIcon
          message="详情页暂不可用"
          description={error ?? '未找到对应数据集。'}
          action={
            <Button size="small" onClick={() => void loadDetail()}>
              重试
            </Button>
          }
        />
      </div>
    )
  }

  const probabilityEntries = Object.entries(classification?.probabilities ?? {})
  const classificationItems = classificationHistory.map((item) => ({
    ...item,
    dayLabel: formatClassificationDay(item),
  }))
  const adviceItems = advices.flatMap((item) =>
    item.content.items.map((contentItem, index) => ({
      key: `${item.advice.id}-${index}`,
      advice: item.advice,
      ...contentItem,
    })),
  )
  const datasetReady = detail.dataset.status === 'ready'
  const handleGenerateForecast = async () => {
    const window = buildDayWindow(detail.dataset.time_end, selectedFutureDay)
    setForecastActionLoading(true)
    try {
      const result = await runForecast(datasetId, {
        model_type: selectedForecastModel,
        granularity: '15min',
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

  const handleGenerateAdvices = async () => {
    setAdviceActionLoading(true)
    try {
      await generateAdvices(datasetId)
      await refreshAdvices()
      message.success('规则建议已生成。')
    } catch (error) {
      message.error(extractApiErrorMessage(error, '生成规则建议失败，请稍后重试。'))
    } finally {
      setAdviceActionLoading(false)
    }
  }

  const handleExportReport = async (reportType: ReportType) => {
    setReportActionLoading(true)
    try {
      await exportDatasetReport(datasetId, reportType)
      setReports(await fetchReports(datasetId))
      message.success(`${reportTypeMap[reportType]}导出任务已创建。`)
    } catch (error) {
      message.error(extractApiErrorMessage(error, '导出报告失败。'))
    } finally {
      setReportActionLoading(false)
    }
  }

  return (
    <div className="page-stack">
      <PageHero
        eyebrow="数据集详情"
        title={detail.dataset.name}
        description={`${detail.dataset.description || '未填写描述'} · ${formatDateTime(detail.dataset.time_start)} 至 ${formatDateTime(detail.dataset.time_end)}`}
        icon={<LineChartOutlined />}
        extra={
          <div className="hero-side-card">
            <Space direction="vertical" size={14} style={{ width: '100%' }}>
              <Button icon={<ArrowLeftOutlined />} onClick={() => navigate('/datasets')}>
                返回列表
              </Button>
              <StatusTag status={detail.dataset.status} />
              <Button
                type="primary"
                icon={<RobotOutlined />}
                disabled={!datasetReady}
                onClick={() => navigate(`/chat?dataset=${detail.dataset.id}`)}
              >
                进入智能问答
              </Button>
            </Space>
          </div>
        }
      >
        <Space wrap size={10}>
          {detail.dataset.household_id ? <Tag className="tone-tag tone-tag--accent">{detail.dataset.household_id}</Tag> : null}
          <Tag className="tone-tag">{detail.quality_summary?.sampling_interval ?? '质量摘要待生成'}</Tag>
          <Tag className="tone-tag tone-tag--muted">{detail.dataset.row_count} 条记录</Tag>
        </Space>
      </PageHero>

      <Row gutter={[16, 16]}>
        {!analysis ? (
          <Col span={24}>
            <Alert
              type={detail.dataset.status === 'error' ? 'error' : 'info'}
              showIcon
              message={
                detail.dataset.status === 'processing'
                  ? '数据集仍在处理中'
                  : detail.dataset.status === 'uploaded'
                    ? '数据集已接收，等待处理'
                    : '当前数据集暂不可分析'
              }
              description={
                detail.dataset.error_message ??
                '分析、分类、预测和报告会在处理完成后开放查看。'
              }
            />
          </Col>
        ) : null}

        {analysis ? (
          <>
            <Col xs={24} md={12} xl={6}>
              <MetricCard
                label="总用电量"
                value={`${formatNumber(analysis.summary.total_kwh)} kWh`}
                hint="整个样本时间范围内的累计电量"
                accent="amber"
              />
            </Col>
            <Col xs={24} md={12} xl={6}>
              <MetricCard
                label="日均用电"
                value={`${formatNumber(analysis.summary.daily_avg_kwh)} kWh`}
                hint="按天聚合后的平均水平"
                accent="teal"
              />
            </Col>
            <Col xs={24} md={12} xl={6}>
              <MetricCard
                label="最高负荷"
                value={`${formatNumber(analysis.summary.max_load_w)} W`}
                hint={formatDateTime(analysis.summary.max_load_time)}
                accent="coral"
              />
            </Col>
            <Col xs={24} md={12} xl={6}>
              <MetricCard
                label="最低负荷"
                value={`${formatNumber(analysis.summary.min_load_w)} W`}
                hint={formatDateTime(analysis.summary.min_load_time)}
                accent="olive"
              />
            </Col>
          </>
        ) : null}
      </Row>

      {analysis ? (
        <Tabs
          className="detail-tabs"
          items={[
          {
            key: 'analysis',
            label: '分析图表',
            children: (
              <div className="page-stack">
                <SectionCard
                  title="数据质量摘要"
                  subtitle="帮助快速判断样本完整性与清洗情况"
                >
                    <Descriptions column={{ xs: 1, lg: 2 }} size="small">
                      <Descriptions.Item label="缺失率">
                        {detail.quality_summary
                          ? formatPercent(detail.quality_summary.missing_rate)
                          : '--'}
                      </Descriptions.Item>
                      <Descriptions.Item label="重复时间戳">
                        {detail.quality_summary?.duplicate_count ?? '--'}
                      </Descriptions.Item>
                      <Descriptions.Item label="采样间隔">
                        {detail.quality_summary?.sampling_interval ?? '--'}
                      </Descriptions.Item>
                      <Descriptions.Item label="清洗策略" span={2}>
                        {detail.quality_summary?.cleaning_strategies?.length ? (
                          <Space wrap>
                            {detail.quality_summary.cleaning_strategies.map((item) => (
                              <Tag key={item}>{item}</Tag>
                            ))}
                          </Space>
                        ) : (
                          '--'
                        )}
                      </Descriptions.Item>
                    </Descriptions>
                </SectionCard>

                <Row gutter={[16, 16]}>
                  <Col xs={24} xl={12}>
                    <SectionCard title="按日趋势" subtitle="查看每日用电变化节奏">
                      <TrendChart
                        data={analysis.charts.daily_trend.map((item) => ({
                          label: item.date.slice(5),
                          value: item.kwh,
                        }))}
                        lineColor="#9b876d"
                      />
                    </SectionCard>
                  </Col>
                  <Col xs={24} xl={12}>
                    <SectionCard title="按周趋势" subtitle="用于观察跨周波动与整体走势">
                      {weeklyTrendChartData.length >= 2 ? (
                        <TrendChart
                          data={weeklyTrendChartData}
                          lineColor="#5d6d5e"
                        />
                      ) : (
                        <Empty description="样本不足两周，暂不展示按周趋势" />
                      )}
                    </SectionCard>
                  </Col>
                  <Col xs={24} xl={14}>
                    <SectionCard title="典型日曲线" subtitle="24 小时平均负荷曲线">
                      <TrendChart
                        data={analysis.charts.typical_day_curve.map((item) => ({
                          label: `${String(item.hour).padStart(2, '0')}:00`,
                          value: item.avg_load_w,
                        }))}
                        lineColor="#8c7b62"
                        unit="W"
                      />
                    </SectionCard>
                  </Col>
                  <Col xs={24} xl={10}>
                    <SectionCard title="峰谷平占比" subtitle="从时段结构理解用电分布">
                      <PeakRatioChart data={analysis.charts.peak_valley_pie} />
                      <Descriptions column={1} size="small">
                        <Descriptions.Item label="峰时">
                          {analysis.peak_valley_config.peak.join(' / ')}
                        </Descriptions.Item>
                        <Descriptions.Item label="谷时">
                          {analysis.peak_valley_config.valley.join(' / ')}
                        </Descriptions.Item>
                        <Descriptions.Item label="平时">
                          {flatPeriods.join(' / ')}
                        </Descriptions.Item>
                      </Descriptions>
                      <Typography.Text type="secondary">
                        平时为未落入峰时、谷时配置的其余时段，所以峰时和谷时本身不需要互补。
                      </Typography.Text>
                    </SectionCard>
                  </Col>
                </Row>
              </div>
            ),
          },
          {
            key: 'classification',
            label: '分类与建议',
            children: (
              <div className="page-stack">
                <Row gutter={[16, 16]}>
                  <Col xs={24} xl={10}>
                    <SectionCard
                      title="行为分类结果"
                      subtitle="识别样本更接近哪一类日常用电模式。"
                      extra={
                        <Button
                          loading={classificationActionLoading}
                          onClick={() => void handleRunClassification()}
                        >
                          重新生成分类
                        </Button>
                      }
                    >
                      {classification ? (
                        <Space direction="vertical" size={16} style={{ width: '100%' }}>
                          <Tag className="tone-tag tone-tag--accent">
                            {classification.label_display_name ??
                              classificationLabelMap[classification.predicted_label]}
                          </Tag>
                          {classification.explanation ? (
                            <Typography.Paragraph>{classification.explanation}</Typography.Paragraph>
                          ) : (
                            <Typography.Paragraph>
                              当前分类模型将该样本识别为
                              {classification.label_display_name ??
                                classificationLabelMap[classification.predicted_label]}
                              ，可结合下方概率分布判断稳定性。
                            </Typography.Paragraph>
                          )}
                          <Typography.Text strong>
                            最高置信度：{formatPercent(classification.confidence)}
                          </Typography.Text>
                          <Typography.Text type="secondary">
                            当前摘要对应日期：{formatClassificationDay(classification)}
                          </Typography.Text>
                          <div className="probability-list">
                            {probabilityEntries.map(([label, probability]) => (
                              <div key={label} className="probability-list__item">
                                <span>{classificationLabelMap[label as keyof typeof classificationLabelMap]}</span>
                                <Progress
                                  percent={Number((probability * 100).toFixed(2))}
                                  showInfo={false}
                                  strokeColor="#0f766e"
                                />
                              </div>
                            ))}
                          </div>
                          <List
                            size="small"
                            header={<Typography.Text strong>按日分类历史</Typography.Text>}
                            dataSource={classificationItems}
                            locale={{ emptyText: '暂无按日分类记录' }}
                            renderItem={(item) => (
                              <List.Item key={item.id}>
                                <Space
                                  align="start"
                                  size={12}
                                  style={{ justifyContent: 'space-between', width: '100%' }}
                                >
                                  <div>
                                    <Typography.Text strong>{item.dayLabel}</Typography.Text>
                                    <br />
                                    <Typography.Text type="secondary">
                                      {item.explanation || '暂无分类说明'}
                                    </Typography.Text>
                                  </div>
                                  <div style={{ textAlign: 'right' }}>
                                    <Tag>{item.label_display_name ?? classificationLabelMap[item.predicted_label]}</Tag>
                                    <br />
                                    <Typography.Text type="secondary">
                                      置信度 {formatPercent(item.confidence)}
                                    </Typography.Text>
                                  </div>
                                </Space>
                              </List.Item>
                            )}
                          />
                        </Space>
                      ) : (
                        <Empty description="暂无分类结果" />
                      )}
                    </SectionCard>
                  </Col>
                  <Col xs={24} xl={14}>
                    <SectionCard
                      title="节能建议面板"
                      subtitle="根据当前样本特征给出可执行建议。"
                      extra={
                        <Button
                          loading={adviceActionLoading}
                          onClick={() => void handleGenerateAdvices()}
                        >
                          生成规则建议
                        </Button>
                      }
                    >
                      <List
                        itemLayout="vertical"
                        dataSource={adviceItems}
                        locale={{ emptyText: '暂无节能建议' }}
                        renderItem={(item) => (
                          <List.Item key={item.key}>
                            <Space direction="vertical" size={8} style={{ width: '100%' }}>
                              <Space wrap>
                                <Tag className="tone-tag tone-tag--warm">
                                  {adviceTypeMap[item.advice.advice_type]}
                                </Tag>
                                <Typography.Text strong>{item.action}</Typography.Text>
                              </Space>
                              <Typography.Paragraph style={{ marginBottom: 0 }}>
                                触发依据：{item.reason}
                              </Typography.Paragraph>
                            </Space>
                          </List.Item>
                        )}
                      />
                    </SectionCard>
                  </Col>
                </Row>
              </div>
            ),
          },
          {
            key: 'forecast',
            label: '未来预测',
            children: (
              <div className="page-stack">
                <SectionCard
                  title="预测操作台"
                  subtitle={`基于最近 7 天窗口，递推预测未来第 1 到第 ${maxVisibleForecastDays} 天。`}
                  extra={
                    <Space wrap>
                      <Select
                        value={selectedFutureDay}
                        style={{ width: 160 }}
                        options={Array.from({ length: maxVisibleForecastDays }, (_, index) => ({
                          label: `未来第 ${index + 1} 天`,
                          value: index + 1,
                        }))}
                        onChange={(value: number) => setSelectedFutureDay(value)}
                      />
                      <Select
                        value={selectedForecastModel}
                        style={{ width: 180 }}
                        options={[{ label: forecastModelMap.tft, value: 'tft' }]}
                        onChange={(value: ForecastModelType) => setSelectedForecastModel(value)}
                      />
                      <Button
                        type="primary"
                        icon={<RocketOutlined />}
                        loading={forecastActionLoading}
                        onClick={() => void handleGenerateForecast()}
                      >
                        生成预测
                      </Button>
                    </Space>
                  }
                >
                  {selectedForecast ? (
                    <Row gutter={[16, 16]}>
                      <Col xs={24} md={12} xl={6}>
                        <MetricCard
                          label="平均负荷"
                          value={`${formatNumber(selectedForecast.summary.predicted_avg_load_w)} W`}
                          accent="amber"
                        />
                      </Col>
                      <Col xs={24} md={12} xl={6}>
                        <MetricCard
                          label="峰值负荷"
                          value={`${formatNumber(selectedForecast.summary.predicted_peak_load_w)} W`}
                          accent="teal"
                        />
                      </Col>
                      <Col xs={24} md={12} xl={6}>
                        <MetricCard
                          label="预测总量"
                          value={`${formatNumber(getPredictedTotalKwh(selectedForecast.summary))} kWh`}
                          accent="coral"
                        />
                      </Col>
                      <Col xs={24} md={12} xl={6}>
                        <MetricCard
                          label="风险标签"
                          value={String(selectedForecast.summary.risk_flags.length)}
                          accent="olive"
                        />
                      </Col>
                    </Row>
                  ) : (
                    <Empty description="暂无预测结果，请先生成预测" />
                  )}
                </SectionCard>

                <Row gutter={[16, 16]}>
                  <Col xs={24} xl={9}>
                    <SectionCard title="预测结果列表" subtitle="保留最近结果，便于切换查看。">
                      <List
                        dataSource={visibleForecasts}
                        locale={{ emptyText: '暂无预测记录' }}
                        renderItem={(item) => (
                          <List.Item
                            className={
                              item.id === activeForecastId
                                ? 'forecast-list-item forecast-list-item--active'
                                : 'forecast-list-item'
                            }
                            onClick={() => {
                              setActiveForecastId(item.id)
                            }}
                          >
                            <Space direction="vertical" size={6} style={{ width: '100%' }}>
                              <Space wrap>
                                <Tag className="tone-tag tone-tag--accent">
                                  {forecastModelMap[item.model_type]}
                                </Tag>
                                <Tag className="tone-tag tone-tag--warm">
                                  {getForecastDayLabel(item.forecast_start, detail.dataset.time_end)}
                                </Tag>
                                <Typography.Text strong>
                                  {formatDateTime(item.forecast_start)}
                                </Typography.Text>
                              </Space>
                              <Typography.Text type="secondary">
                                {formatNumber(item.summary.predicted_avg_load_w)} W 平均负荷 ·{' '}
                                {formatNumber(getPredictedTotalKwh(item.summary))} kWh 预测总量
                              </Typography.Text>
                            </Space>
                          </List.Item>
                        )}
                      />
                    </SectionCard>
                  </Col>
                  <Col xs={24} xl={15}>
                    <SectionCard title="预测曲线" subtitle="展示所选预测记录的时间序列。">
                      {forecastDetailLoading ? (
                        <div className="page-state page-state--compact">
                          <Spin />
                        </div>
                      ) : forecastDetail ? (
                        <TrendChart
                          data={forecastDetail.series.map((item) => ({
                            label: formatTime(item.timestamp),
                            value: item.predicted,
                          }))}
                          lineColor="#9b876d"
                          unit="W"
                        />
                      ) : (
                        <Empty description="暂无预测曲线" />
                      )}
                    </SectionCard>
                  </Col>
                </Row>

                <Row gutter={[16, 16]}>
                  <Col xs={24}>
                    <SectionCard title="预测摘要" subtitle="汇总关键时段与风险标签。">
                      {selectedForecast ? (
                        <Descriptions column={1} size="small">
                          <Descriptions.Item label="预测天数">
                            {getForecastDayLabel(selectedForecast.forecast_start, detail.dataset.time_end)}
                          </Descriptions.Item>
                          <Descriptions.Item label="模型类型">
                            {forecastModelMap[selectedForecast.model_type]}
                          </Descriptions.Item>
                          <Descriptions.Item label="预测区间">
                            {formatDateTime(selectedForecast.forecast_start)} -{' '}
                            {formatTime(selectedForecast.forecast_end)}
                          </Descriptions.Item>
                          <Descriptions.Item label="高负荷时段">
                            {getForecastPeakPeriods(selectedForecast.summary).length ? (
                              <Space direction="vertical" size={4}>
                                {getForecastPeakPeriods(selectedForecast.summary).map((item) => (
                                  <Typography.Text key={item}>{formatPeriodRange(item)}</Typography.Text>
                                ))}
                              </Space>
                            ) : (
                              '--'
                            )}
                          </Descriptions.Item>
                          <Descriptions.Item label="预测总用电量">
                            {formatNumber(getPredictedTotalKwh(selectedForecast.summary))} kWh
                          </Descriptions.Item>
                          <Descriptions.Item label="风险标签">
                            {selectedForecast.summary.risk_flags.length ? (
                              <Space wrap>
                                {selectedForecast.summary.risk_flags.map((flag) => (
                                  <Tag key={flag} className="tone-tag tone-tag--muted">
                                    {riskFlagMap[flag] ?? flag}
                                  </Tag>
                                ))}
                              </Space>
                            ) : (
                              '--'
                            )}
                          </Descriptions.Item>
                        </Descriptions>
                      ) : (
                        <Empty description="暂无预测摘要" />
                      )}
                    </SectionCard>
                  </Col>
                </Row>
              </div>
            ),
          },
          {
            key: 'reports',
            label: '报告与下载',
            children: (
              <SectionCard
                title="导出报告"
                subtitle="仅保留 PDF 报告导出，便于统一归档与展示。"
                extra={
                  <Space wrap>
                    <Button
                      type="primary"
                      icon={<FilePdfOutlined />}
                      loading={reportActionLoading}
                      onClick={() => void handleExportReport('pdf')}
                    >
                      导出 PDF
                    </Button>
                  </Space>
                }
              >
                <List
                  dataSource={reports}
                  locale={{ emptyText: '暂无报告记录' }}
                  renderItem={(report) => (
                    <List.Item
                      actions={[
                        <Button
                          key="download"
                          icon={<DownloadOutlined />}
                          type="link"
                          onClick={() => void downloadReport(report)}
                        >
                          下载
                        </Button>,
                      ]}
                    >
                      <List.Item.Meta
                        avatar={
                          report.report_type === 'pdf' ? (
                            <FilePdfOutlined className="report-icon" />
                          ) : (
                            <DownloadOutlined className="report-icon" />
                          )
                        }
                        title={`${reportTypeMap[report.report_type]} · #${report.id}`}
                        description={`${formatFileLabel(report.file_path)} · ${formatFileSize(report.file_size)} · ${formatDateTime(report.created_at)}`}
                      />
                    </List.Item>
                  )}
                />
              </SectionCard>
            ),
          },
          ]}
        />
      ) : null}
    </div>
  )
}
