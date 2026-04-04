import { useCallback, useEffect, useState } from 'react'
import {
  CheckCircleOutlined,
  ClusterOutlined,
  ReloadOutlined,
  WarningOutlined,
} from '@ant-design/icons'
import {
  Alert,
  Button,
  Col,
  List,
  Row,
  Spin,
  Typography,
} from 'antd'
import { PageHero } from '@/components/common/PageHero'
import { SectionCard } from '@/components/common/SectionCard'
import { MetricCard } from '@/components/sections/MetricCard'
import {
  fetchHealth,
  getRuntimeModeLabel,
  isMockMode,
} from '@/services/dashboard'
import type { HealthStatus } from '@/types/domain'

function getHealthStatusLabel(status: HealthStatus['status']) {
  switch (status) {
    case 'up':
      return '运行正常'
    case 'degraded':
      return '部分降级'
    default:
      return '不可用'
  }
}

function getDependencyLabel(key: string) {
  switch (key) {
    case 'database':
      return '数据库'
    case 'model':
      return '模型服务'
    case 'agent':
      return '智能体服务'
    default:
      return key
  }
}

export function SettingsPage() {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [health, setHealth] = useState<HealthStatus | null>(null)

  const loadPage = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const healthResult = await fetchHealth()
      setHealth(healthResult)
    } catch {
      setError('系统状态加载失败，请稍后重试。')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadPage()
  }, [loadPage])

  if (loading) {
    return (
      <div className="page-state">
        <Spin size="large" />
      </div>
    )
  }

  if (error || !health) {
    return (
      <div className="page-state">
        <Alert
          type="error"
          showIcon
          message={error ?? '状态页暂不可用'}
          action={
            <Button size="small" onClick={() => void loadPage()}>
              重试
            </Button>
          }
        />
      </div>
    )
  }

  const dependencyEntries = Object.entries(health.dependencies ?? {})

  return (
    <div className="page-stack">
      <PageHero
        eyebrow="系统状态"
        title="只展示当前可感知的运行状态"
        description="这里不再暴露模型窗口、提示词、目录等内部参数，只保留用户真正需要知道的服务状态。"
        icon={<CheckCircleOutlined />}
        extra={
          <div className="hero-side-card">
            <Typography.Text strong>当前模式</Typography.Text>
            <Typography.Paragraph type="secondary" style={{ marginBottom: 12 }}>
              {getRuntimeModeLabel()}
            </Typography.Paragraph>
            <Button icon={<ReloadOutlined />} onClick={() => void loadPage()}>
              刷新状态
            </Button>
          </div>
        }
      >
        <span className={isMockMode ? 'ant-tag tone-tag tone-tag--warm' : 'ant-tag tone-tag tone-tag--accent'}>
          {getRuntimeModeLabel()}
        </span>
      </PageHero>

      <Row gutter={[16, 16]}>
        <Col xs={24} md={8}>
          <MetricCard
            label="服务状态"
            value={getHealthStatusLabel(health.status)}
            hint={`${health.service}${health.version ? ` · ${health.version}` : ''}`}
            accent="teal"
            icon={<CheckCircleOutlined />}
          />
        </Col>
        <Col xs={24} md={8}>
          <MetricCard
            label="依赖数量"
            value={String(dependencyEntries.length)}
            hint="当前健康检查覆盖的服务"
            accent="amber"
            icon={<ClusterOutlined />}
          />
        </Col>
        <Col xs={24} md={8}>
          <MetricCard
            label="异常依赖"
            value={String(dependencyEntries.filter(([, status]) => status !== 'up').length)}
            hint="包含降级与不可用"
            accent="coral"
            icon={<WarningOutlined />}
          />
        </Col>
      </Row>

      <SectionCard
        title="依赖服务状态"
        subtitle="用于快速判断当前系统是否能正常完成分析、预测和智能问答。"
      >
        <List
          dataSource={dependencyEntries}
          locale={{ emptyText: '当前没有可展示的依赖状态' }}
          renderItem={([key, status]) => (
            <List.Item>
              <div className="status-list-item">
                <Typography.Text strong>{getDependencyLabel(key)}</Typography.Text>
                <span className={`ant-tag status-tag status-tag--${status}`}>
                  {getHealthStatusLabel(status as HealthStatus['status'])}
                </span>
              </div>
            </List.Item>
          )}
        />
      </SectionCard>
    </div>
  )
}
