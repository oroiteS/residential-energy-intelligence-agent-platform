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
import { fetchDatasetList, importDataset } from '@/services/dashboard'
import type { DatasetStatus, DatasetSummary } from '@/types/domain'
import { formatDateTime, formatNumber } from '@/utils/formatters'

const DEFAULT_PAGE_SIZE = 6

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
      setError('数据集列表加载失败，请检查后端服务或稍后重试。')
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
    } catch {
      message.error('导入失败，请稍后重试。')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="page-stack">
      <PageHero
        eyebrow="首版闭环"
        title="数据集中心"
        description="上传原始用电文件，查看处理状态，并进入详情页查看分析、预测、建议和聊天结果。"
        icon={<AppstoreOutlined />}
        extra={
          <div className="hero-side-card dataset-hero__aside">
            <div>
              <Typography.Text className="dataset-hero__aside-label">
                当前页记录点数
              </Typography.Text>
              <Typography.Title level={2} className="dataset-hero__aside-value">
                {formatNumber(totalRows, 0)}
              </Typography.Title>
              <Typography.Paragraph className="dataset-hero__aside-desc">
                当前筛选结果对应的总记录点数，用来快速判断样本规模是否足够进入后续分析。
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
          <span className="ant-tag tone-tag tone-tag--accent">当前页 {datasets.length} 个条目</span>
          <span className="ant-tag tone-tag">筛选总数 {total}</span>
          <span className="ant-tag tone-tag tone-tag--muted">每页 {pageSize} 项</span>
        </div>

        <div className="dataset-hero__metrics">
          <MetricCard
            label="当前页条目"
            value={String(datasets.length)}
            hint="本页实际展示的数据集数量"
            accent="amber"
            icon={<AppstoreOutlined />}
          />
          <MetricCard
            label="筛选结果"
            value={String(total)}
            hint="后端分页返回的总命中数"
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
            subtitle="状态、时间范围和记录规模会直接影响后续分析与建议质量。"
            extra={
              <div className="dataset-list__toolbar">
                <div className="dataset-list__toolbar-fields">
                  <Input
                    allowClear
                    prefix={<SearchOutlined />}
                    placeholder="按名称、家庭标识筛选"
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
            subtitle="这里只保留用户真正需要填写的内容，内部字段按系统默认值处理。"
          >
            <div className="dataset-upload__summary">
              <div className="dataset-upload__summary-icon">
                <FileTextOutlined />
              </div>
              <div>
                <Typography.Text strong>导入流程</Typography.Text>
                <Typography.Paragraph className="dataset-upload__summary-text">
                  选择原始文件后，补充一个清晰名称即可提交；其余解析参数由系统使用默认策略完成。
                </Typography.Paragraph>
              </div>
            </div>

            <Form className="dataset-upload__form" layout="vertical" onFinish={() => void handleSubmit()}>
              <Form.Item label="上传文件" required>
                <label className="upload-dropzone">
                  <input
                    className="upload-dropzone__input"
                    type="file"
                    accept=".csv,.xlsx"
                    onChange={(event) => {
                      const file = event.target.files?.[0] ?? null
                      setSelectedFile(file)
                      if (file && !datasetName) {
                        setDatasetName(file.name.replace(/\.[^.]+$/, ''))
                      }
                    }}
                  />
                  <InboxOutlined className="upload-dropzone__icon" />
                  <Typography.Text strong>
                    {selectedFile ? selectedFile.name : '点击选择 CSV 或 xlsx 文件'}
                  </Typography.Text>
                  <Typography.Paragraph type="secondary" style={{ marginBottom: 0 }}>
                    支持直接导入原始文件，系统会自动完成默认解析。
                  </Typography.Paragraph>
                </label>
              </Form.Item>

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
                  placeholder="例如：REFIT House 3"
                />
              </Form.Item>

              <Form.Item label="描述">
                <Input.TextArea
                  rows={3}
                  value={description}
                  onChange={(event) => setDescription(event.target.value)}
                  placeholder="简要记录样本特征或用途"
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
