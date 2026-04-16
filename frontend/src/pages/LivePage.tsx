import { useState } from 'react'
import {
  FundProjectionScreenOutlined,
  LinkOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
import { Button, Col, Row, Space, Typography } from 'antd'
import { PageHero } from '@/components/common/PageHero'
import { SectionCard } from '@/components/common/SectionCard'

const liveBaseUrl = (import.meta.env.VITE_LIVE_BASE_URL as string | undefined)?.trim() || 'http://127.0.0.1:8090'
const liveEmbedVersion = '20260417-today-forecast'

function buildLiveEmbedUrl(baseUrl: string, frameKey: number) {
  try {
    const url = new URL(baseUrl)
    url.searchParams.set('embed_version', liveEmbedVersion)
    url.searchParams.set('frame_key', String(frameKey))
    return url.toString()
  } catch {
    const separator = baseUrl.includes('?') ? '&' : '?'
    return `${baseUrl}${separator}embed_version=${liveEmbedVersion}&frame_key=${frameKey}`
  }
}

export function LivePage() {
  const [frameKey, setFrameKey] = useState(0)

  return (
    <div className="page-stack">
      <PageHero
        eyebrow="实时演示"
        title="循环播放独立住户实时用电"
        description="这里直接接入 live 模块。页面会每秒推进一个虚拟 15 分钟点，并同步展示实时曲线、日分类、下一日预测和直接提问结果。"
        icon={<FundProjectionScreenOutlined />}
        extra={
          <div className="hero-side-card live-entry-card">
            <Space direction="vertical" size={12} style={{ width: '100%' }}>
              <Typography.Text strong>访问方式</Typography.Text>
              <Typography.Paragraph type="secondary" style={{ marginBottom: 0 }}>
                如果嵌入区域为空，通常表示 `live` 服务未启动。你也可以直接在新窗口打开独立演示页。
              </Typography.Paragraph>
              <Space wrap>
                <Button
                  type="primary"
                  icon={<LinkOutlined />}
                  onClick={() => window.open(liveBaseUrl, '_blank', 'noopener,noreferrer')}
                >
                  新窗口打开
                </Button>
                <Button
                  icon={<ReloadOutlined />}
                  onClick={() => setFrameKey((value) => value + 1)}
                >
                  刷新嵌入页
                </Button>
              </Space>
            </Space>
          </div>
        }
      />

      <Row gutter={[16, 16]}>
        <Col xs={24}>
          <SectionCard
            title="实时演示画面"
            subtitle="主前端内嵌显示，方便你在同一工作台中切换查看。"
            className="live-embed-card"
          >
            <div className="live-embed-frame">
              <iframe
                key={frameKey}
                src={buildLiveEmbedUrl(liveBaseUrl, frameKey)}
                title="实时用电演示"
                loading="lazy"
              />
            </div>
          </SectionCard>
        </Col>
      </Row>
    </div>
  )
}
