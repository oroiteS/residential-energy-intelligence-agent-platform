import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  DownloadOutlined,
  FileExcelOutlined,
  FileTextOutlined,
  FolderOpenOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
import {
  Alert,
  Button,
  Col,
  Empty,
  List,
  Row,
  Select,
  Space,
  Table,
  Typography,
  message,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { useNavigate } from 'react-router-dom'
import { PageHero } from '@/components/common/PageHero'
import { SectionCard } from '@/components/common/SectionCard'
import { StatusTag } from '@/components/common/StatusTag'
import { MetricCard } from '@/components/sections/MetricCard'
import { reportTypeMap } from '@/constants/display'
import {
  downloadReport,
  exportDatasetReport,
  fetchDatasets,
  fetchReports,
} from '@/services/dashboard'
import type { DatasetSummary, ReportRecord, ReportType } from '@/types/domain'
import {
  formatDateTime,
  formatFileSize,
} from '@/utils/formatters'

type ReportListItem = ReportRecord & {
  dataset_name: string
  dataset_status: DatasetSummary['status']
  household_id: string | null
}

type DatasetFilterValue = number | 'all'
type ReportFilterValue = ReportType | 'all'

export function ReportsPage() {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [exportingDatasetId, setExportingDatasetId] = useState<number | null>(null)
  const [datasets, setDatasets] = useState<DatasetSummary[]>([])
  const [reports, setReports] = useState<ReportListItem[]>([])
  const [selectedDatasetId, setSelectedDatasetId] = useState<DatasetFilterValue>('all')
  const [selectedReportType, setSelectedReportType] = useState<ReportFilterValue>('all')

  const loadPage = useCallback(async () => {
    setLoading(true)
    setError(null)

    try {
      const datasetList = await fetchDatasets()
      const reportGroups = await Promise.all(
        datasetList.map(async (dataset) => {
          const items = await fetchReports(dataset.id)
          return items.map<ReportListItem>((report) => ({
            ...report,
            dataset_name: dataset.name,
            dataset_status: dataset.status,
            household_id: dataset.household_id,
          }))
        }),
      )

      setDatasets(datasetList)
      setReports(
        reportGroups
          .flat()
          .sort(
            (left, right) =>
              new Date(right.created_at).getTime() - new Date(left.created_at).getTime(),
          ),
      )
    } catch {
      setError('报告中心加载失败，请检查接口状态后重试。')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadPage()
  }, [loadPage])

  const readyDatasets = useMemo(
    () => datasets.filter((item) => item.status === 'ready'),
    [datasets],
  )

  const filteredReports = useMemo(
    () =>
      reports.filter((report) => {
        if (selectedDatasetId !== 'all' && report.dataset_id !== selectedDatasetId) {
          return false
        }

        if (selectedReportType !== 'all' && report.report_type !== selectedReportType) {
          return false
        }

        return true
      }),
    [reports, selectedDatasetId, selectedReportType],
  )

  const latestReport = reports[0] ?? null

  const handleExport = async (datasetId: number) => {
    setExportingDatasetId(datasetId)
    try {
      await exportDatasetReport(datasetId, 'excel')
      message.success('Excel 报告已导出并刷新列表。')
      await loadPage()
    } catch {
      message.error('导出报告失败，请稍后重试。')
    } finally {
      setExportingDatasetId(null)
    }
  }

  const columns: ColumnsType<ReportListItem> = [
    {
      title: '报告编号',
      dataIndex: 'id',
      key: 'id',
      width: 96,
    },
    {
      title: '所属数据集',
      dataIndex: 'dataset_name',
      key: 'dataset_name',
      render: (_, record) => (
        <Space direction="vertical" size={2}>
          <Typography.Text strong>{record.dataset_name}</Typography.Text>
          <Space wrap size={6}>
            <StatusTag status={record.dataset_status} />
            {record.household_id ? <Typography.Text type="secondary">{record.household_id}</Typography.Text> : null}
          </Space>
        </Space>
      ),
    },
    {
      title: '报告类型',
      dataIndex: 'report_type',
      key: 'report_type',
      width: 132,
      render: (value: ReportType) => reportTypeMap[value],
    },
    {
      title: '文件信息',
      key: 'file',
      render: (_, record) => (
        <Space direction="vertical" size={2}>
          <Typography.Text>{record.file_path}</Typography.Text>
          <Typography.Text type="secondary">
            {formatFileSize(record.file_size)}
          </Typography.Text>
        </Space>
      ),
    },
    {
      title: '导出时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 188,
      render: (value: string) => formatDateTime(value),
    },
    {
      title: '操作',
      key: 'actions',
      width: 212,
      render: (_, record) => (
        <Space wrap>
          <Button
            type="link"
            icon={<DownloadOutlined />}
            onClick={() => void downloadReport(record)}
          >
            下载
          </Button>
          <Button
            icon={<FolderOpenOutlined />}
            onClick={() => navigate(`/datasets/${record.dataset_id}`)}
          >
            查看数据集
          </Button>
        </Space>
      ),
    },
  ]

  return (
    <div className="page-stack">
      <PageHero
        eyebrow="报告中心"
        title="统一查看导出记录与下载入口"
        description="这里聚合所有数据集的 Excel 报告记录，方便集中导出、筛选和回到对应数据集详情页继续分析。"
        icon={<FileTextOutlined />}
        extra={
          <div className="hero-side-card">
            <Space direction="vertical" size={12} style={{ width: '100%' }}>
              <Typography.Text strong>当前导出口径</Typography.Text>
              <Typography.Paragraph type="secondary" style={{ marginBottom: 0 }}>
                当前后端只开放 Excel 报告导出与下载，列表页按真实能力收敛展示。
              </Typography.Paragraph>
              <Button icon={<ReloadOutlined />} onClick={() => void loadPage()}>
                刷新报告列表
              </Button>
            </Space>
          </div>
        }
      >
        <Space wrap size={[12, 12]}>
          <MetricCard label="报告总数" value={String(reports.length)} accent="amber" />
          <MetricCard
            label="可导出数据集"
            value={String(readyDatasets.length)}
            accent="teal"
          />
          <MetricCard
            label="Excel 报告"
            value={String(reports.filter((item) => item.report_type === 'excel').length)}
            accent="olive"
          />
          <MetricCard
            label="最近导出"
            value={latestReport ? formatDateTime(latestReport.created_at).slice(5, 16) : '--'}
            hint={latestReport ? latestReport.dataset_name : '暂无记录'}
            accent="coral"
          />
        </Space>
      </PageHero>

      {error ? (
        <Alert
          type="error"
          showIcon
          message={error}
          action={
            <Button size="small" onClick={() => void loadPage()}>
              重试
            </Button>
          }
        />
      ) : null}

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={16}>
          <SectionCard
            title="报告列表"
            subtitle="支持按数据集和报告类型过滤，并直接触发下载。"
            extra={
              <Space wrap>
                <Select<DatasetFilterValue>
                  value={selectedDatasetId}
                  style={{ width: 220 }}
                  options={[
                    { label: '全部数据集', value: 'all' },
                    ...datasets.map((item) => ({
                      label: item.name,
                      value: item.id,
                    })),
                  ]}
                  onChange={setSelectedDatasetId}
                />
                <Select<ReportFilterValue>
                  value={selectedReportType}
                  style={{ width: 160 }}
                  options={[
                    { label: '全部类型', value: 'all' },
                    { label: 'Excel 报告', value: 'excel' },
                  ]}
                  onChange={setSelectedReportType}
                />
              </Space>
            }
          >
            <Table
              rowKey="id"
              loading={loading}
              columns={columns}
              dataSource={filteredReports}
              locale={{
                emptyText: loading ? '正在加载报告…' : <Empty description="暂无匹配的报告记录" />,
              }}
              pagination={{ pageSize: 8, hideOnSinglePage: true }}
            />
          </SectionCard>
        </Col>

        <Col xs={24} xl={8}>
          <SectionCard
            title="快速导出"
            subtitle="只对已就绪的数据集开放导出，避免在处理中样本上触发无效操作。"
          >
            <List
              dataSource={readyDatasets}
              locale={{ emptyText: '暂无可导出的数据集' }}
              renderItem={(dataset) => (
                <List.Item
                  actions={[
                    <Button
                      key="export"
                      type="primary"
                      icon={<FileExcelOutlined />}
                      loading={exportingDatasetId === dataset.id}
                      onClick={() => void handleExport(dataset.id)}
                    >
                      导出 Excel
                    </Button>,
                  ]}
                >
                  <List.Item.Meta
                    title={dataset.name}
                    description={`${dataset.household_id ?? '未设置家庭标识'} · ${formatDateTime(dataset.updated_at)}`}
                  />
                </List.Item>
              )}
            />
          </SectionCard>
        </Col>
      </Row>
    </div>
  )
}
