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
  Divider,
  List,
  Row,
  Space,
  Typography,
} from 'antd'
import { PageHero } from '@/components/common/PageHero'
import { SectionCard } from '@/components/common/SectionCard'
import { MetricCard } from '@/components/sections/MetricCard'
import { fetchHealth } from '@/services/dashboard'
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
      setError('服务概览加载失败，请稍后重试。')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadPage()
  }, [loadPage])

  const dependencyEntries = Object.entries(health?.dependencies ?? {})
  const degradedCount = dependencyEntries.filter(([, status]) => status !== 'up').length

  return (
    <div className="page-stack">
      <PageHero
        eyebrow="服务概览"
        title="查看服务状态与关键入口"
        description="这里统一展示主系统依赖健康度，并提供实时演示模块的固定访问入口。即使健康检查暂时失败，页面本身也应保持可打开。"
        icon={<CheckCircleOutlined />}
        extra={
          <div className="hero-side-card">
            <Space direction="vertical" size={12} style={{ width: '100%' }}>
              <Button icon={<ReloadOutlined />} onClick={() => void loadPage()}>
                刷新概览
              </Button>
              <Typography.Text type="secondary">
                {loading ? '正在刷新服务状态…' : '页面已加载，可继续操作。'}
              </Typography.Text>
            </Space>
          </div>
        }
      />

      {error ? (
        <Alert
          type="warning"
          showIcon
          message={error}
          description="已回退到静态概览视图。你仍然可以访问其他页面和实时演示模块。"
          action={
            <Button size="small" onClick={() => void loadPage()}>
              重试
            </Button>
          }
        />
      ) : null}

      <Row gutter={[16, 16]}>
        <Col xs={24} md={8}>
          <MetricCard
            label="服务状态"
            value={health ? getHealthStatusLabel(health.status) : '--'}
            hint="当前平台整体可用性"
            accent="teal"
            icon={<CheckCircleOutlined />}
          />
        </Col>
        <Col xs={24} md={8}>
          <MetricCard
            label="依赖数量"
            value={String(dependencyEntries.length)}
            hint="已纳入健康检查的服务"
            accent="amber"
            icon={<ClusterOutlined />}
          />
        </Col>
        <Col xs={24} md={8}>
          <MetricCard
            label="异常依赖"
            value={String(degradedCount)}
            hint="需要优先关注的依赖项"
            accent="coral"
            icon={<WarningOutlined />}
          />
        </Col>
      </Row>

      <SectionCard
        title="常用入口"
        subtitle="减少来回查找页面的成本。"
      >
        <div className="overview-links">
          <a className="overview-link-card" href="/datasets">
            <strong>数据集中心</strong>
            <span>导入数据、查看分析、运行分类与预测。</span>
          </a>
          <a className="overview-link-card" href="/reports">
            <strong>报告中心</strong>
            <span>统一查看 PDF 导出记录，避免手动翻找文件。</span>
          </a>
          <a className="overview-link-card" href="/live">
            <strong>实时演示</strong>
            <span>进入独立实时用电界面，查看循环周样本与动态预测。</span>
          </a>
        </div>
      </SectionCard>

      <SectionCard
        title="依赖健康度"
        subtitle="帮助判断当前是否适合继续分析、预测与问答。"
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
        {!dependencyEntries.length ? (
          <>
            <Divider />
            <Typography.Text type="secondary">
              当前未获取到依赖健康数据。通常是后端尚未启动，或 `/api/v1/health` 暂时不可用。
            </Typography.Text>
          </>
        ) : null}
      </SectionCard>
    </div>
  )
}
