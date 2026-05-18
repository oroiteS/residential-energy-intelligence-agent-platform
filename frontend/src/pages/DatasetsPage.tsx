import { startTransition, useCallback, useDeferredValue, useEffect, useState } from 'react'
import {
  AppstoreOutlined,
  CheckCircleOutlined,
  FileTextOutlined,
  InboxOutlined,
  ReloadOutlined,
  SearchOutlined,
  UploadOutlined,
} from '@ant-design/icons'
import {
  Alert,
  Button,
  Col,
  Empty,
  Form,
  Input,
  message,
  Row,
  Select,
  Table,
  Typography,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { useNavigate } from 'react-router-dom'
import { PageHero } from '@/components/common/PageHero'
import { SectionCard } from '@/components/common/SectionCard'
import { StatusTag } from '@/components/common/StatusTag'
import { MetricCard } from '@/components/sections/MetricCard'
import { extractApiErrorMessage, fetchDatasetList, importDataset } from '@/services/dashboard'
import type { DatasetStatus, DatasetSummary } from '@/types/domain'
import { formatDateTime, formatNumber } from '@/utils/formatters'

const DEFAULT_PAGE_SIZE = 6
const STANDARD_COLUMNS = ['timestamp', 'aggregate_w']
const MIN_UPLOAD_GRANULARITY_MINUTES = 1
const MAX_UPLOAD_GRANULARITY_MINUTES = 60

const columns: ColumnsType<DatasetSummary> = [
  {
    title: '数据集名称',
    dataIndex: 'name',
    key: 'name',
    render: (_, record) => (
      <div>
        <Typography.Text strong>{record.name}</Typography.Text>
        <Typography.Paragraph type="secondary" style={{ marginBottom: 0 }}>
          {record.description || '未填写描述'}
        </Typography.Paragraph>
      </div>
    ),
  },
  {
    title: '家庭标识',
    dataIndex: 'household_id',
    key: 'household_id',
  },
  {
    title: '时间范围',
    key: 'time_range',
    render: (_, record) => (
      <span>
        {formatDateTime(record.time_start)}
        <br />
        {formatDateTime(record.time_end)}
      </span>
    ),
  },
  {
    title: '记录数',
    dataIndex: 'row_count',
    key: 'row_count',
    align: 'right',
    render: (value: number) => formatNumber(value, 0),
  },
  {
    title: '状态',
    dataIndex: 'status',
    key: 'status',
    render: (value) => <StatusTag status={value} />,
  },
]

export function DatasetsPage() {
  const navigate = useNavigate()
  const [datasets, setDatasets] = useState<DatasetSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [keyword, setKeyword] = useState('')
  const [selectedStatus, setSelectedStatus] = useState<DatasetStatus | 'all'>('all')
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE)
  const [total, setTotal] = useState(0)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [datasetName, setDatasetName] = useState('')
  const [description, setDescription] = useState('')

  const deferredKeyword = useDeferredValue(keyword)

  const loadDatasets = useCallback(async (params: {
    page: number
    page_size: number
    keyword?: string
    status?: DatasetStatus
  }) => {
    setLoading(true)
    setError(null)
    try {
      const result = await fetchDatasetList(params)
      setDatasets(result.items)
      setPage(result.pagination.page)
      setPageSize(result.pagination.page_size)
      setTotal(result.pagination.total)
    } catch {
      setError('数据集列表加载失败，请稍后重试。')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadDatasets({
      page,
      page_size: pageSize,
      keyword: deferredKeyword.trim() || undefined,
      status: selectedStatus === 'all' ? undefined : selectedStatus,
    })
  }, [deferredKeyword, loadDatasets, page, pageSize, selectedStatus])

  const readyCount = datasets.filter((item) => item.status === 'ready').length
  const processingCount = datasets.filter((item) => item.status === 'processing').length
  const totalRows = datasets.reduce((sum, item) => sum + item.row_count, 0)

  const downloadTemplate = () => {
    const rows = [
      STANDARD_COLUMNS.join(','),
      '2026-01-01 00:00:00,420',
      '2026-01-01 00:15:00,435',
      '2026-01-01 00:30:00,418',
      '2026-01-01 00:45:00,402',
    ]
    const blob = new Blob([`${rows.join('\n')}\n`], { type: 'text/csv;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'resident_energy_upload_template.csv'
    anchor.click()
    URL.revokeObjectURL(url)
  }

  const validateUploadFile = async (file: File): Promise<string | null> => {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      return '上传格式错误：当前仅支持 CSV 文件。'
    }

    const previewText = await file.slice(0, 256 * 1024).text()
    const lines = previewText
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)

    if (lines.length < 2) {
      return '上传格式错误：文件至少需要包含表头和两条数据。'
    }

    const headers = lines[0].split(',').map((item) => item.trim())
    if (
      headers.length !== STANDARD_COLUMNS.length ||
      headers.some((item, index) => item !== STANDARD_COLUMNS[index])
    ) {
      return `上传格式错误：CSV 表头必须严格为 ${STANDARD_COLUMNS.join(',')}。`
    }

    const timestamps: number[] = []
    const previewRows = lines.slice(1, 501)
    for (const [index, line] of previewRows.entries()) {
      const cells = line.split(',').map((item) => item.trim())
      if (cells.length !== 2) {
        return `上传格式错误：第 ${index + 2} 行必须包含 timestamp 和 aggregate_w 两列。`
      }

      const timestamp = new Date(cells[0].replace(' ', 'T')).getTime()
      const aggregate = Number(cells[1])
      if (!Number.isFinite(timestamp)) {
        return `上传格式错误：第 ${index + 2} 行 timestamp 无法识别。`
      }
      if (!Number.isFinite(aggregate) || aggregate < 0) {
        return `上传格式错误：第 ${index + 2} 行 aggregate_w 必须是非负数字。`
      }
      timestamps.push(timestamp)
    }

    if (timestamps.length < 2) {
      return '上传格式错误：文件至少需要两条有效时间序列记录。'
    }

    const sortedTimestamps = [...timestamps].sort((left, right) => left - right)
    const diffs = sortedTimestamps
      .slice(1)
      .map((value, index) => Math.round((value - sortedTimestamps[index]) / 1000))
      .filter((value) => value > 0)

    if (!diffs.length || new Set(diffs).size !== 1) {
      return '上传格式错误：timestamp 必须按固定间隔采样。'
    }

    const granularitySeconds = diffs[0]
    if (granularitySeconds % 60 !== 0) {
      return '上传格式错误：采样间隔必须是整分钟，不能使用秒级原始数据。'
    }

    const granularityMinutes = granularitySeconds / 60
    if (
      granularityMinutes < MIN_UPLOAD_GRANULARITY_MINUTES ||
      granularityMinutes > MAX_UPLOAD_GRANULARITY_MINUTES
    ) {
      return `上传格式错误：采样间隔必须在 ${MIN_UPLOAD_GRANULARITY_MINUTES} 到 ${MAX_UPLOAD_GRANULARITY_MINUTES} 分钟之间。`
    }

    return null
  }

  const handleSubmit = async () => {
    if (!selectedFile) {
      message.warning('请先选择上传文件。')
      return
    }

    if (!datasetName.trim()) {
      message.warning('请填写数据集名称。')
      return
    }

    setSubmitting(true)
    try {
      await importDataset({
        name: datasetName.trim(),
        description: description.trim() || null,
        household_id: null,
        unit: 'w',
        file_name: selectedFile.name,
        file: selectedFile,
      })

      const result = await fetchDatasetList({
        page: 1,
        page_size: pageSize,
        keyword: keyword.trim() || undefined,
        status: selectedStatus === 'all' ? undefined : selectedStatus,
      })
      startTransition(() => {
        setDatasets(result.items)
        setPage(result.pagination.page)
        setPageSize(result.pagination.page_size)
        setTotal(result.pagination.total)
      })

      setSelectedFile(null)
      setDatasetName('')
      setDescription('')
      message.success('数据集导入请求已创建。')
    } catch (error) {
      message.error(extractApiErrorMessage(error, '导入失败，请稍后重试。'))
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="page-stack">
      <PageHero
        eyebrow="数据接入"
        title="数据集中心"
        description="上传原始用电文件，查看处理进度，并进入详情页继续分析、预测与节能建议。"
        icon={<AppstoreOutlined />}
        extra={
          <div className="hero-side-card dataset-hero__aside">
            <div>
              <Typography.Text className="dataset-hero__aside-label">
                当前页记录量
              </Typography.Text>
              <Typography.Title level={2} className="dataset-hero__aside-value">
                {formatNumber(totalRows, 0)}
              </Typography.Title>
              <Typography.Paragraph className="dataset-hero__aside-desc">
                帮助快速判断本次筛选结果的样本规模。
              </Typography.Paragraph>
            </div>

            <div className="dataset-hero__aside-list">
              <div className="dataset-hero__aside-item">
                <span className="dataset-hero__aside-item-label">分页位置</span>
                <span className="dataset-hero__aside-item-value">
                  第 {page} / {Math.max(1, Math.ceil(total / pageSize))} 页
                </span>
              </div>
              <div className="dataset-hero__aside-item">
                <span className="dataset-hero__aside-item-label">处理中样本</span>
                <span className="dataset-hero__aside-item-value">
                  {processingCount} 个
                </span>
              </div>
            </div>
          </div>
        }
      >
        <div className="dataset-hero__meta">
          <span className="ant-tag tone-tag tone-tag--accent">本页 {datasets.length} 个数据集</span>
          <span className="ant-tag tone-tag">筛选命中 {total}</span>
          <span className="ant-tag tone-tag tone-tag--muted">每页 {pageSize} 项</span>
        </div>

        <div className="dataset-hero__metrics">
          <MetricCard
            label="当前页数据集"
            value={String(datasets.length)}
            hint="本页实际展示的数据集数量"
            accent="amber"
            icon={<AppstoreOutlined />}
          />
          <MetricCard
            label="筛选命中"
            value={String(total)}
            hint="符合当前筛选条件的数据集数量"
            accent="olive"
            icon={<SearchOutlined />}
          />
          <MetricCard
            label="当前页可查看"
            value={String(readyCount)}
            hint="已完成处理，可进入详情分析"
            accent="teal"
            icon={<CheckCircleOutlined />}
          />
        </div>
      </PageHero>

      {error ? (
        <Alert
          type="error"
          showIcon
          message={error}
          action={
            <Button
              size="small"
              onClick={() =>
                void loadDatasets({
                  page,
                  page_size: pageSize,
                  keyword: keyword.trim() || undefined,
                  status: selectedStatus === 'all' ? undefined : selectedStatus,
                })
              }
            >
              重试
            </Button>
          }
        />
      ) : null}

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={15}>
          <SectionCard
            className="dataset-list-card"
            title="数据集列表"
            subtitle="状态、时间范围和记录规模有助于快速判断数据集是否可用。"
            extra={
              <div className="dataset-list__toolbar">
                <div className="dataset-list__toolbar-fields">
                  <Input
                    allowClear
                    prefix={<SearchOutlined />}
                    placeholder="按名称、家庭标识筛选…"
                    value={keyword}
                    onChange={(event) => {
                      setKeyword(event.target.value)
                      setPage(1)
                    }}
                    className="dataset-list__search"
                  />
                  <Select<DatasetStatus | 'all'>
                    value={selectedStatus}
                    className="dataset-list__select"
                    options={[
                      { label: '全部状态', value: 'all' },
                      { label: '已接收', value: 'uploaded' },
                      { label: '处理中', value: 'processing' },
                      { label: '可查看', value: 'ready' },
                      { label: '处理失败', value: 'error' },
                    ]}
                    onChange={(value) => {
                      setSelectedStatus(value)
                      setPage(1)
                    }}
                  />
                </div>

                <div className="dataset-list__toolbar-actions">
                  <span className="dataset-list__toolbar-meta">共 {total} 条筛选结果</span>
                  <Button
                    icon={<ReloadOutlined />}
                    onClick={() =>
                      void loadDatasets({
                        page,
                        page_size: pageSize,
                        keyword: keyword.trim() || undefined,
                        status: selectedStatus === 'all' ? undefined : selectedStatus,
                      })
                    }
                  >
                    刷新
                  </Button>
                </div>
              </div>
            }
          >
            <Table
              rowKey="id"
              loading={loading}
              columns={columns}
              dataSource={datasets}
              locale={{
                emptyText: loading ? '正在加载数据集…' : <Empty description="暂无匹配的数据集" />,
              }}
              pagination={{
                current: page,
                pageSize,
                total,
                showSizeChanger: true,
                pageSizeOptions: [6, 12, 20],
                hideOnSinglePage: total <= pageSize,
              }}
              onChange={(nextPagination) => {
                const nextPageSize = nextPagination.pageSize ?? pageSize
                const nextPage = nextPagination.current ?? 1

                if (nextPageSize !== pageSize) {
                  setPageSize(nextPageSize)
                  setPage(1)
                  return
                }

                setPage(nextPage)
              }}
              onRow={(record) => ({
                onClick: () => navigate(`/datasets/${record.id}`),
              })}
            />
          </SectionCard>
        </Col>

        <Col xs={24} xl={9}>
          <SectionCard
            className="dataset-upload-card"
            title="上传并导入"
            subtitle="选择文件并填写名称，即可完成导入。"
          >
            <div className="dataset-upload__summary">
              <div className="dataset-upload__summary-icon">
                <FileTextOutlined />
              </div>
              <div>
                <Typography.Text strong>导入说明</Typography.Text>
                <Typography.Paragraph className="dataset-upload__summary-text">
                  仅支持标准 CSV：表头必须为 timestamp,aggregate_w，aggregate_w 单位为 W。
                  采样间隔必须固定在 1 到 60 分钟之间，数据覆盖时长至少 30 天。
                  上传后系统会自动开始处理，你可以在列表中持续查看进度。
                </Typography.Paragraph>
              </div>
            </div>
            <div className="dataset-upload__example">
              <Typography.Text strong>标准 CSV 示例</Typography.Text>
              <pre>{`timestamp,aggregate_w
2013-10-09 17:00:00,523
2013-10-09 17:15:00,526`}</pre>
            </div>

            <Form className="dataset-upload__form" layout="vertical" onFinish={() => void handleSubmit()}>
              <Form.Item label="上传文件" required>
                <label className="upload-dropzone">
                  <input
                    className="upload-dropzone__input"
                    type="file"
                    accept=".csv"
                    onChange={(event) => {
                      const file = event.target.files?.[0] ?? null
                      if (!file) {
                        setSelectedFile(null)
                        return
                      }
                      void validateUploadFile(file).then((validationError) => {
                        if (validationError) {
                          message.error(validationError)
                          setSelectedFile(null)
                          event.target.value = ''
                          return
                        }
                        setSelectedFile(file)
                        if (!datasetName) {
                          setDatasetName(file.name.replace(/\.[^.]+$/, ''))
                        }
                      })
                    }}
                  />
                  <InboxOutlined className="upload-dropzone__icon" />
                  <Typography.Text strong>
                    {selectedFile ? selectedFile.name : '点击选择 CSV 文件'}
                  </Typography.Text>
                  <Typography.Paragraph type="secondary" style={{ marginBottom: 0 }}>
                    标准表头：timestamp,aggregate_w；采样间隔 1 到 60 分钟，覆盖至少 30 天。
                  </Typography.Paragraph>
                </label>
              </Form.Item>
              <Button type="link" onClick={downloadTemplate} style={{ paddingLeft: 0 }}>
                下载标准 CSV 模板
              </Button>

              {selectedFile ? (
                <div className="dataset-upload__selected">
                  <span className="dataset-upload__selected-label">已选文件</span>
                  <span className="dataset-upload__selected-name">{selectedFile.name}</span>
                </div>
              ) : null}

              <Form.Item label="数据集名称" required>
                <Input
                  value={datasetName}
                  onChange={(event) => setDatasetName(event.target.value)}
                  placeholder="例如：REFIT House 3…"
                />
              </Form.Item>

              <Form.Item label="描述">
                <Input.TextArea
                  rows={3}
                  value={description}
                  onChange={(event) => setDescription(event.target.value)}
                  placeholder="简要记录样本特征或用途…"
                />
              </Form.Item>

              <div className="dataset-upload__actions">
                <Button
                  type="primary"
                  icon={<UploadOutlined />}
                  htmlType="submit"
                  loading={submitting}
                >
                  提交导入
                </Button>
                <Button
                  onClick={() => {
                    setSelectedFile(null)
                    setDatasetName('')
                    setDescription('')
                  }}
                >
                  重置表单
                </Button>
              </div>
            </Form>
          </SectionCard>
        </Col>
      </Row>

    </div>
  )
}
