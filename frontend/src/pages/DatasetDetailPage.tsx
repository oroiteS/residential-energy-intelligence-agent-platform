import { ArrowLeftOutlined, LineChartOutlined, RobotOutlined } from '@ant-design/icons'
import { Alert, Button, Col, Row, Space, Spin, Tabs, Tag } from 'antd'
import { useNavigate, useParams } from 'react-router-dom'
import { PageHero } from '@/components/common/PageHero'
import { StatusTag } from '@/components/common/StatusTag'
import { MetricCard } from '@/components/sections/MetricCard'
import { AnalysisTab } from '@/features/dataset-detail/components/AnalysisTab'
import { ClassificationTab } from '@/features/dataset-detail/components/ClassificationTab'
import { DetectionTab } from '@/features/dataset-detail/components/DetectionTab'
import { ForecastTab } from '@/features/dataset-detail/components/ForecastTab'
import { ReportsTab } from '@/features/dataset-detail/components/ReportsTab'
import { useDatasetDetailPage } from '@/features/dataset-detail/hooks/useDatasetDetailPage'
import { formatDateTime, formatNumber } from '@/utils/formatters'

// 数据集详情页的页面组装层。
// 这里只负责读取路由参数、组织页面头部、指标卡和五个业务 Tab；
// 具体的数据加载、按钮操作和业务规则已经下沉到 features/dataset-detail 中。
export function DatasetDetailPage() {
  const navigate = useNavigate()
  const params = useParams()
  const datasetId = Number(params.datasetId)
  const {
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
  } = useDatasetDetailPage(datasetId)

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

  // 只有处理完成的数据集才能进入智能问答、分析、预测等后续流程。
  const datasetReady = detail.dataset.status === 'ready'

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
          {detail.dataset.household_id ? (
            <Tag className="tone-tag tone-tag--accent">{detail.dataset.household_id}</Tag>
          ) : null}
          <Tag className="tone-tag">
            {detail.quality_summary?.sampling_interval ?? '质量摘要待生成'}
          </Tag>
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
                <AnalysisTab
                  detail={detail}
                  analysis={analysis}
                  peakValleyConfig={peakValleyConfig}
                />
              ),
            },
            {
              key: 'classification',
              label: '分类与建议',
              children: (
                <ClassificationTab
                  classification={classification}
                  classificationHistory={classificationHistory}
                  classificationActionLoading={classificationActionLoading}
                  onRunClassification={() => void handleRunClassification()}
                  onOpenChat={() => navigate(`/chat?dataset=${detail.dataset.id}`)}
                />
              ),
            },
            {
              key: 'detection',
              label: '异常检测',
              children: (
                <DetectionTab
                  detection={detection}
                  detectionActionLoading={detectionActionLoading}
                  onRunDetection={() => void handleRunDetection()}
                />
              ),
            },
            {
              key: 'forecast',
              label: '未来预测',
              children: (
                <ForecastTab
                  visibleForecasts={visibleForecasts}
                  selectedForecast={selectedForecast}
                  activeForecastId={activeForecastId}
                  forecastDetail={forecastDetail}
                  forecastDetailLoading={forecastDetailLoading}
                  forecastActionLoading={forecastActionLoading}
                  selectedForecastModel={selectedForecastModel}
                  onForecastModelChange={setSelectedForecastModel}
                  onActiveForecastChange={setActiveForecastId}
                  onGenerateForecast={() => void handleGenerateForecast()}
                />
              ),
            },
            {
              key: 'reports',
              label: '报告与下载',
              children: (
                <ReportsTab
                  reports={reports}
                  reportActionLoading={reportActionLoading}
                  onExportReport={(reportType) => void handleExportReport(reportType)}
                />
              ),
            },
          ]}
        />
      ) : null}
    </div>
  )
}
