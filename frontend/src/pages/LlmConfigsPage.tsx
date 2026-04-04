import { useEffect, useState } from 'react'
import {
  DeleteOutlined,
  EditOutlined,
  PlusOutlined,
  StarOutlined,
} from '@ant-design/icons'
import {
  Alert,
  Button,
  Col,
  Form,
  Input,
  InputNumber,
  Row,
  Space,
  Switch,
  Table,
  Tag,
  Typography,
  message,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { MetricCard } from '@/components/sections/MetricCard'
import { PageHero } from '@/components/common/PageHero'
import { SectionCard } from '@/components/common/SectionCard'
import {
  createLlmConfig,
  deleteLlmConfig,
  fetchLlmConfigs,
  setDefaultLlmConfig,
  updateLlmConfig,
} from '@/services/dashboard'
import type { LlmConfig, LlmConfigInput } from '@/types/domain'
import { formatDateTime } from '@/utils/formatters'

const initialValues: LlmConfigInput = {
  name: '',
  base_url: '',
  api_key: '',
  model_name: '',
  temperature: 0.2,
  timeout_seconds: 60,
  is_default: false,
}

export function LlmConfigsPage() {
  const [form] = Form.useForm<LlmConfigInput>()
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [editingId, setEditingId] = useState<number | null>(null)
  const [configs, setConfigs] = useState<LlmConfig[]>([])

  useEffect(() => {
    void loadConfigs()
  }, [])

  const loadConfigs = async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await fetchLlmConfigs()
      setConfigs(result)
    } catch {
      setError('LLM 配置加载失败，请稍后重试。')
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = async (values: LlmConfigInput) => {
    setSubmitting(true)
    try {
      if (editingId) {
        await updateLlmConfig(editingId, values)
        message.success('配置已更新。')
      } else {
        await createLlmConfig(values)
        message.success('配置已创建。')
      }
      form.setFieldsValue(initialValues)
      setEditingId(null)
      await loadConfigs()
    } catch {
      message.error('保存配置失败，请检查输入后重试。')
    } finally {
      setSubmitting(false)
    }
  }

  const handleEdit = (record: LlmConfig) => {
    setEditingId(record.id)
    form.setFieldsValue({
      name: record.name,
      base_url: record.base_url,
      api_key: '',
      model_name: record.model_name,
      temperature: record.temperature,
      timeout_seconds: record.timeout_seconds,
      is_default: record.is_default,
    })
  }

  const columns: ColumnsType<LlmConfig> = [
    {
      title: '配置名称',
      dataIndex: 'name',
      key: 'name',
      render: (_, record) => (
        <Space>
          <Typography.Text strong>{record.name}</Typography.Text>
          {record.is_default ? <Tag className="tone-tag tone-tag--warm">默认</Tag> : null}
        </Space>
      ),
    },
    {
      title: '模型',
      dataIndex: 'model_name',
      key: 'model_name',
    },
    {
      title: 'Base URL',
      dataIndex: 'base_url',
      key: 'base_url',
      ellipsis: true,
    },
    {
      title: '温度',
      dataIndex: 'temperature',
      key: 'temperature',
      width: 100,
    },
    {
      title: '超时',
      dataIndex: 'timeout_seconds',
      key: 'timeout_seconds',
      width: 110,
      render: (value: number) => `${value}s`,
    },
    {
      title: '更新时间',
      dataIndex: 'updated_at',
      key: 'updated_at',
      width: 180,
      render: (value: string) => formatDateTime(value),
    },
    {
      title: '操作',
      key: 'actions',
      width: 220,
      render: (_, record) => (
        <Space wrap>
          <Button icon={<EditOutlined />} onClick={() => handleEdit(record)}>
            编辑
          </Button>
          <Button
            icon={<StarOutlined />}
            onClick={async () => {
              try {
                await setDefaultLlmConfig(record.id)
                message.success('默认配置已切换。')
                await loadConfigs()
              } catch {
                message.error('设置默认配置失败。')
              }
            }}
          >
            设为默认
          </Button>
          <Button
            danger
            icon={<DeleteOutlined />}
            disabled={record.is_default}
            onClick={async () => {
              try {
                await deleteLlmConfig(record.id)
                message.success('配置已删除。')
                if (editingId === record.id) {
                  setEditingId(null)
                  form.setFieldsValue(initialValues)
                }
                await loadConfigs()
              } catch {
                message.error('删除配置失败。')
              }
            }}
          >
            删除
          </Button>
        </Space>
      ),
    },
  ]

  const defaultConfig = configs.find((item) => item.is_default)

  return (
    <div className="page-stack">
      <PageHero
        eyebrow="LLM 配置"
        title="统一管理兼容 OpenAI 的模型接入"
        description="该页面对应 `GET/POST/PUT/DELETE /api/v1/llm-configs` 与默认项切换接口，前端只管理配置，不拼接提示词上下文。"
        icon={<StarOutlined />}
      >
        <Space wrap size={[12, 12]}>
          <MetricCard label="配置数量" value={String(configs.length)} accent="amber" />
          <MetricCard
            label="默认模型"
            value={defaultConfig?.model_name ?? '--'}
            accent="teal"
          />
          <MetricCard
            label="最长超时"
            value={`${Math.max(0, ...configs.map((item) => item.timeout_seconds))}s`}
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
            <Button size="small" onClick={() => void loadConfigs()}>
              重试
            </Button>
          }
        />
      ) : null}

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={15}>
          <SectionCard title="配置列表" subtitle="默认项需要保持唯一，删除前必须先切换默认配置。">
            <Table rowKey="id" loading={loading} columns={columns} dataSource={configs} pagination={false} />
          </SectionCard>
        </Col>
        <Col xs={24} xl={9}>
          <SectionCard
            title={editingId ? '编辑配置' : '新建配置'}
            subtitle="当前仅支持 OpenAI-compatible 接口，不做多协议适配。"
          >
            <Form
              form={form}
              layout="vertical"
              initialValues={initialValues}
              onFinish={(values) => void handleSubmit(values)}
            >
              <Form.Item label="配置名称" name="name" rules={[{ required: true, message: '请输入配置名称' }]}>
                <Input placeholder="例如：deepseek-local" />
              </Form.Item>
              <Form.Item label="Base URL" name="base_url" rules={[{ required: true, message: '请输入 Base URL' }]}>
                <Input placeholder="https://example.com/v1" />
              </Form.Item>
              <Form.Item label="API Key" name="api_key" rules={[{ required: true, message: '请输入 API Key' }]}>
                <Input.Password placeholder={editingId ? '编辑时需要重新填写 API Key' : 'sk-xxxx'} />
              </Form.Item>
              <Form.Item label="模型名称" name="model_name" rules={[{ required: true, message: '请输入模型名称' }]}>
                <Input placeholder="deepseek-chat" />
              </Form.Item>
              <Row gutter={12}>
                <Col span={12}>
                  <Form.Item label="温度" name="temperature">
                    <InputNumber min={0} max={2} step={0.1} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item label="超时秒数" name="timeout_seconds">
                    <InputNumber min={10} max={600} step={5} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
              </Row>
              <Form.Item label="设为默认" name="is_default" valuePropName="checked">
                <Switch />
              </Form.Item>
              <Space wrap>
                <Button
                  type="primary"
                  htmlType="submit"
                  icon={<PlusOutlined />}
                  loading={submitting}
                >
                  {editingId ? '保存修改' : '创建配置'}
                </Button>
                <Button
                  onClick={() => {
                    setEditingId(null)
                    form.setFieldsValue(initialValues)
                  }}
                >
                  重置表单
                </Button>
              </Space>
            </Form>
          </SectionCard>
        </Col>
      </Row>
    </div>
  )
}
