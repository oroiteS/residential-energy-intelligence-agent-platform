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
  Row,
  Select,
  Space,
  Spin,
  Tabs,
  Table,
  Tag,
  Tooltip,
  Typography,
  message,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { useNavigate, useParams } from 'react-router-dom'
import { PeakRatioChart } from '@/components/charts/PeakRatioChart'
import { TrendChart } from '@/components/charts/TrendChart'
import { PageHero } from '@/components/common/PageHero'
import { SectionCard } from '@/components/common/SectionCard'
import { StatusTag } from '@/components/common/StatusTag'
import { MetricCard } from '@/components/sections/MetricCard'
import {
  classificationLabelMap,
  detectionFeatureDescriptionMap,
  detectionFeatureLabelMap,
  detectionMethodLabelMap,
  forecastModelMap,
  reportTypeMap,
  riskFlagMap,
} from '@/constants/display'
import {
  downloadReport,
  extractApiErrorMessage,
  exportDatasetReport,
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
} from '@/utils/formatters'

type ClassificationTableItem = ClassificationResult & {
  dayLabel: string
}

const maxVisibleForecastDays = 7
const defaultPeakValleyConfig: PeakValleyConfig = {
  peak: ['07:00-11:00', '18:00-23:00'],
  valley: ['23:00-07:00', '11:00-18:00'],
}

function buildDayWindow(timeEnd: string | null | undefined, dayOffset = 1, durationDays = 7) {
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
  if (item.window_start && item.window_end) {
    return `${formatDateTime(item.window_start).slice(0, 10)} ~ ${formatDateTime(item.window_end).slice(5, 10)}`
  }
  const source = item.window_start || item.created_at
  return source ? formatDateTime(source).slice(0, 10) : '未知窗口'
}

function getForecastWindowLabel(summary: ForecastSummary | null | undefined) {
  if (summary?.forecast_horizon === '7d') {
    return '未来 7 天'
  }
  return '预测窗口'
}

function getForecastRangeLabel(
  forecastStart: string | null | undefined,
  forecastEnd: string | null | undefined,
) {
  if (!forecastStart || !forecastEnd) {
    return '--'
  }
  return `${formatDateTime(forecastStart).slice(0, 10)} ~ ${formatDateTime(forecastEnd).slice(5, 10)}`
}

function getPredictedTotalKwh(summary: ForecastSummary) {
  if (summary.predicted_total_kwh !== undefined && summary.predicted_total_kwh !== null) {
    return summary.predicted_total_kwh
  }
  return 0
}

export function DatasetDetailPage() {
  const navigate = useNavigate()
  const params = useParams()
  const datasetId = Number(params.datasetId)

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
  const peakValleyConfig = {
    peak: analysis?.peak_valley_config.peak.length
      ? analysis.peak_valley_config.peak
      : runtimePeakValleyConfig.peak,
    valley: analysis?.peak_valley_config.valley.length
      ? analysis.peak_valley_config.valley
      : runtimePeakValleyConfig.valley,
  }
  const weeklyTrendChartData =
    analysis && analysis.charts.weekly_trend.length >= 2
      ? analysis.charts.weekly_trend.map((item) => ({
          label: `${item.week_start.slice(5)} ~ ${item.week_end.slice(5)}`,
          value: item.kwh,
        }))
      : []
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

  const classificationItems: ClassificationTableItem[] = classificationHistory.map((item) => ({
    ...item,
    dayLabel: formatClassificationDay(item),
  }))
  const renderClassificationConfidenceTooltip = (item: ClassificationResult) => (
    <div className="classification-confidence-tooltip">
      {Object.entries(classificationLabelMap).map(([label, displayName]) => {
        const probability =
          item.probabilities[label as keyof typeof classificationLabelMap] ?? 0
        return (
          <div key={label} className="classification-confidence-tooltip__row">
            <span>{displayName}</span>
            <strong>{formatPercent(probability)}</strong>
          </div>
        )
      })}
    </div>
  )
  const classificationColumns: ColumnsType<ClassificationTableItem> = [
    {
      title: '分类窗口',
      dataIndex: 'dayLabel',
      key: 'dayLabel',
      width: 190,
      render: (value: string) => <Typography.Text strong>{value}</Typography.Text>,
    },
    {
      title: '行为类别',
      dataIndex: 'predicted_label',
      key: 'predicted_label',
      width: 150,
      render: (_value, item) => (
        <Tag className="tone-tag tone-tag--accent">
          {item.label_display_name ?? classificationLabelMap[item.predicted_label]}
        </Tag>
      ),
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      width: 110,
      render: (value: number, item) => (
        <Tooltip
          placement="topRight"
          title={renderClassificationConfidenceTooltip(item)}
        >
          <Typography.Text className="classification-confidence-value">
            {formatPercent(value)}
          </Typography.Text>
        </Tooltip>
      ),
    },
    {
      title: '分类摘要',
      dataIndex: 'explanation',
      key: 'explanation',
      render: (value: string | undefined) => (
        <Typography.Text type="secondary">{value || '暂无分类说明'}</Typography.Text>
      ),
    },
  ]
  const datasetReady = detail.dataset.status === 'ready'
  const handleGenerateForecast = async () => {
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
                    <SectionCard title="按日趋势" subtitle="每日总用电量趋势，单位 kWh">
                      <TrendChart
                        data={analysis.charts.daily_trend.map((item) => ({
                          label: item.date.slice(5),
                          value: item.kwh,
                        }))}
                        lineColor="#9b876d"
                        unit="kWh"
                      />
                    </SectionCard>
                  </Col>
                  <Col xs={24} xl={12}>
                    <SectionCard title="按周趋势" subtitle="每周总用电量趋势，单位 kWh">
                      {weeklyTrendChartData.length >= 2 ? (
                        <TrendChart
                          data={weeklyTrendChartData}
                          lineColor="#5d6d5e"
                          unit="kWh"
                        />
                      ) : (
                        <Empty description="样本不足两周，暂不展示按周趋势" />
                      )}
                    </SectionCard>
                  </Col>
                  <Col xs={24} xl={14}>
                    <SectionCard title="典型日曲线" subtitle="跨样本日按小时平均负荷曲线，单位 W">
                      <TrendChart
                        data={analysis.charts.typical_day_curve.map((item) => ({
                          label: `${String(item.hour).padStart(2, '0')}:00`,
                          value: item.avg_load_w,
                        }))}
                        lineColor="#8c7b62"
                        unit="W"
                      />
                      <Typography.Text type="secondary">
                        该曲线是所有日期在同一小时负荷的平均值；按小时聚合后会比 15 分钟原始曲线更平滑。
                      </Typography.Text>
                    </SectionCard>
                  </Col>
                  <Col xs={24} xl={10}>
                    <SectionCard title="峰谷占比" subtitle="从峰时和谷时结构理解用电分布">
                      <PeakRatioChart data={analysis.charts.peak_valley_pie} />
                      <Descriptions column={1} size="small">
                        <Descriptions.Item label="峰时">
                          {peakValleyConfig.peak.join(' / ')}
                        </Descriptions.Item>
                        <Descriptions.Item label="谷时">
                          {peakValleyConfig.valley.join(' / ')}
                        </Descriptions.Item>
                      </Descriptions>
                      <Typography.Text type="secondary">
                        本项目仅区分峰时和谷时，所有上传数据最终都会归入这两类时段。
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
                  <Col xs={24} xl={16}>
                    <SectionCard
                      title="行为分类结果"
                      subtitle="按自然周识别用电行为类型，便于观察近期模式是否稳定。"
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
                        <Space direction="vertical" size={18} style={{ width: '100%' }}>
                          <div className="classification-current">
                            <div>
                              <Tag className="tone-tag tone-tag--accent">
                                {classification.label_display_name ??
                                  classificationLabelMap[classification.predicted_label]}
                              </Tag>
                              <Typography.Title level={4} className="classification-current__title">
                                最新分类窗口：{formatClassificationDay(classification)}
                              </Typography.Title>
                              <Typography.Paragraph className="classification-current__summary">
                                {classification.explanation ||
                                  `模型判定为${
                                    classification.label_display_name ??
                                    classificationLabelMap[classification.predicted_label]
                                  }。`}
                              </Typography.Paragraph>
                            </div>
                            <Tooltip
                              placement="topRight"
                              title={renderClassificationConfidenceTooltip(classification)}
                            >
                              <div className="classification-current__confidence">
                                <span>最高置信度</span>
                                <strong>{formatPercent(classification.confidence)}</strong>
                              </div>
                            </Tooltip>
                          </div>
                          <Table
                            className="classification-history-table"
                            columns={classificationColumns}
                            dataSource={classificationItems}
                            pagination={false}
                            rowKey="id"
                            scroll={{ x: 760 }}
                            size="middle"
                          />
                        </Space>
                      ) : (
                        <Empty description="暂无分类结果" />
                      )}
                    </SectionCard>
                  </Col>
                  <Col xs={24} xl={8}>
                    <SectionCard
                      title="智能体建议"
                      subtitle="基于分类、预测与异常检测生成建议。"
                      className="agent-advice-card"
                      extra={
                        <Button
                          type="primary"
                          icon={<RobotOutlined />}
                          onClick={() => navigate(`/chat?dataset=${detail.dataset.id}`)}
                        >
                          进入智能问答
                        </Button>
                      }
                    >
                      <Space direction="vertical" size={14} className="agent-advice-entry">
                        <Space wrap>
                          <Tag className="tone-tag tone-tag--accent">分类上下文</Tag>
                          <Tag className="tone-tag tone-tag--warm">未来预测</Tag>
                          <Tag className="tone-tag tone-tag--muted">异常复核</Tag>
                        </Space>
                      </Space>
                    </SectionCard>
                  </Col>
                </Row>
              </div>
            ),
          },
          {
            key: 'detection',
            label: '异常检测',
            children: (
              <SectionCard
                title="异常检测结果"
                subtitle="识别最近 7 天窗口是否明显偏离历史用电规律。"
                extra={
                  <Button
                    loading={detectionActionLoading}
                    onClick={() => void handleRunDetection()}
                  >
                    重新检测异常
                  </Button>
                }
              >
                {detection ? (
                  <Space direction="vertical" size={18} style={{ width: '100%' }}>
                    <Space wrap>
                      <Tag color={detection.is_anomaly ? 'error' : 'success'}>
                        {detection.is_anomaly ? '检测到异常' : '未检测到异常'}
                      </Tag>
                      <Typography.Text type="secondary">
                        异常分数 {formatNumber(detection.anomaly_score, 4)}
                      </Typography.Text>
                    </Space>
                    <Descriptions column={{ xs: 1, lg: 2 }} size="small">
                      <Descriptions.Item label="检测窗口开始">
                        {formatDateTime(detection.window_start)}
                      </Descriptions.Item>
                      <Descriptions.Item label="检测窗口结束">
                        {formatDateTime(detection.window_end)}
                      </Descriptions.Item>
                      <Descriptions.Item label="辅助分类">
                        {detection.classification_hint ?? '--'}
                      </Descriptions.Item>
                      <Descriptions.Item label="结果时间">
                        {formatDateTime(detection.created_at)}
                      </Descriptions.Item>
                    </Descriptions>
                    <List
                      size="small"
                      header={<Typography.Text strong>异常原因</Typography.Text>}
                      dataSource={detection.reasons}
                      locale={{ emptyText: '当前窗口未触发异常规则。' }}
                      renderItem={(item, index) => (
                        <List.Item key={`${item.rule}-${index}`}>
                          <Space direction="vertical" size={4} style={{ width: '100%' }}>
                            <Space wrap>
                              <Tooltip
                                placement="topLeft"
                                title={
                                  detectionFeatureDescriptionMap[item.feature] ??
                                  '该指标用于辅助解释当前窗口的用电行为特征。'
                                }
                              >
                                <Tag className="feature-reason-tag">
                                  {detectionFeatureLabelMap[item.feature] ?? item.feature}
                                </Tag>
                              </Tooltip>
                              <Tag color="gold">
                                {detectionMethodLabelMap[item.method] ?? item.method}
                              </Tag>
                              <Typography.Text type="secondary">
                                触发强度 {formatPercent(item.severity)}
                              </Typography.Text>
                            </Space>
                            <Typography.Text>{item.reason}</Typography.Text>
                          </Space>
                        </List.Item>
                      )}
                    />
                    <Descriptions column={{ xs: 1, lg: 3 }} size="small" title="关键特征摘要">
                      {Object.entries(detection.feature_summary).map(([key, value]) => (
                        <Descriptions.Item
                          key={key}
                          label={
                            <Tooltip
                              placement="topLeft"
                              title={
                                detectionFeatureDescriptionMap[key] ??
                                '该指标用于辅助解释当前窗口的用电行为特征。'
                              }
                            >
                              <span className="feature-label-chip">
                                {detectionFeatureLabelMap[key] ?? key}
                              </span>
                            </Tooltip>
                          }
                        >
                          {formatNumber(value, 4)}
                        </Descriptions.Item>
                      ))}
                    </Descriptions>
                  </Space>
                ) : (
                  <Empty description="暂无异常检测结果" />
                )}
              </SectionCard>
            ),
          },
          {
            key: 'forecast',
            label: '未来预测',
            children: (
              <div className="page-stack">
                <SectionCard
                  title="预测操作台"
                  subtitle="基于最近 30 天日级窗口，一次生成未来 7 天总量、峰时和谷时预测。"
                  extra={
                    <Space wrap>
                      <Select
                        value={selectedForecastModel}
                        style={{ width: 180 }}
                        options={[{ label: forecastModelMap.lstm, value: 'lstm' }]}
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
                          label="日均预测"
                          value={`${formatNumber(selectedForecast.summary.predicted_avg_daily_kwh)} kWh`}
                          accent="amber"
                        />
                      </Col>
                      <Col xs={24} md={12} xl={6}>
                        <MetricCard
                          label="峰时预测"
                          value={`${formatNumber(selectedForecast.summary.predicted_peak_kwh)} kWh`}
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
                          label="预测类别"
                          value={
                            selectedForecast.summary.forecast_classification
                              ? selectedForecast.summary.forecast_classification.label_display_name ??
                                classificationLabelMap[
                                  selectedForecast.summary.forecast_classification.predicted_label
                                ]
                              : '--'
                          }
                          hint={
                            selectedForecast.summary.forecast_classification
                              ? `置信度 ${formatPercent(
                                  selectedForecast.summary.forecast_classification.confidence,
                                )}`
                              : '待生成未来分类'
                          }
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
                                  {getForecastWindowLabel(item.summary)}
                                </Tag>
                                <Typography.Text strong>
                                  {getForecastRangeLabel(item.forecast_start, item.forecast_end)}
                                </Typography.Text>
                              </Space>
                              <Typography.Text type="secondary">
                                {formatNumber(item.summary.predicted_avg_daily_kwh)} kWh 日均 ·{' '}
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
                            label: item.date.slice(5),
                            value: item.predicted_total_kwh,
                          }))}
                          lineColor="#9b876d"
                          unit="kWh"
                        />
                      ) : (
                        <Empty description="暂无预测曲线" />
                      )}
                    </SectionCard>
                  </Col>
                </Row>

                <Row gutter={[16, 16]}>
                  <Col xs={24}>
                    <SectionCard title="预测摘要" subtitle="汇总关键时段、未来分类与风险提示。">
                      {selectedForecast ? (
                        <Descriptions column={1} size="small">
                          <Descriptions.Item label="预测天数">
                            {getForecastWindowLabel(selectedForecast.summary)}
                          </Descriptions.Item>
                          <Descriptions.Item label="模型类型">
                            {forecastModelMap[selectedForecast.model_type]}
                          </Descriptions.Item>
                          <Descriptions.Item label="预测区间">
                            {getForecastRangeLabel(
                              selectedForecast.forecast_start,
                              selectedForecast.forecast_end,
                            )}
                          </Descriptions.Item>
                          <Descriptions.Item label="预测总用电量">
                            {formatNumber(getPredictedTotalKwh(selectedForecast.summary))} kWh
                          </Descriptions.Item>
                          <Descriptions.Item label="预测分类">
                            {selectedForecast.summary.forecast_classification ? (
                              <Space wrap>
                                <Tag className="tone-tag tone-tag--accent">
                                  {selectedForecast.summary.forecast_classification.label_display_name ??
                                    classificationLabelMap[
                                      selectedForecast.summary.forecast_classification.predicted_label
                                    ]}
                                </Tag>
                                <Typography.Text type="secondary">
                                  XGBoost 基于预测日曲线判断，置信度{' '}
                                  {formatPercent(selectedForecast.summary.forecast_classification.confidence)}
                                </Typography.Text>
                              </Space>
                            ) : (
                              '--'
                            )}
                          </Descriptions.Item>
                          <Descriptions.Item label="复核提示">
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
                          <Descriptions.Item label="未来异常检测">
                            {selectedForecast.summary.future_detection ? (
                              <div className="future-detection-panel">
                                <div className="future-detection-panel__head">
                                  <Tag
                                    color={
                                      selectedForecast.summary.future_detection.is_anomaly
                                        ? 'warning'
                                        : 'success'
                                    }
                                  >
                                    {selectedForecast.summary.future_detection.is_anomaly
                                      ? '建议复核'
                                      : '未触发复核'}
                                  </Tag>
                                  <Typography.Text type="secondary">
                                    偏离分数{' '}
                                    {formatNumber(
                                      selectedForecast.summary.future_detection.anomaly_score,
                                      4,
                                    )}
                                  </Typography.Text>
                                </div>
                                {selectedForecast.summary.future_detection.reasons.length ? (
                                  <List
                                    className="future-detection-reasons"
                                    size="small"
                                    dataSource={selectedForecast.summary.future_detection.reasons}
                                    renderItem={(item, index) => (
                                      <List.Item key={`${item.rule}-${index}`}>
                                        <Space
                                          direction="vertical"
                                          size={6}
                                          style={{ width: '100%' }}
                                        >
                                          <Space wrap>
                                            <Tooltip
                                              placement="topLeft"
                                              title={
                                                detectionFeatureDescriptionMap[item.feature] ??
                                                '该指标用于辅助解释预测窗口的用电行为特征。'
                                              }
                                            >
                                              <Tag className="feature-reason-tag">
                                                {detectionFeatureLabelMap[item.feature] ??
                                                  item.feature}
                                              </Tag>
                                            </Tooltip>
                                            <Tag color="gold">
                                              {detectionMethodLabelMap[item.method] ?? item.method}
                                            </Tag>
                                            <Typography.Text type="secondary">
                                              触发强度 {formatPercent(item.severity)}
                                            </Typography.Text>
                                          </Space>
                                          <Typography.Text>{item.reason}</Typography.Text>
                                        </Space>
                                      </List.Item>
                                    )}
                                  />
                                ) : (
                                  <Typography.Text type="secondary">
                                    预测窗口没有触发额外复核规则。
                                  </Typography.Text>
                                )}
                              </div>
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
