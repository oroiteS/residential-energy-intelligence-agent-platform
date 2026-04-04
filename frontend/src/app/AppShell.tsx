import { useState, type PropsWithChildren } from 'react'
import {
  AppstoreOutlined,
  ControlOutlined,
  DeploymentUnitOutlined,
  FileTextOutlined,
  MenuOutlined,
  RobotOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
import { Button, Drawer, Grid, Layout, Menu, Typography } from 'antd'
import { useLocation, useNavigate } from 'react-router-dom'
import { getRuntimeModeLabel, isMockMode } from '@/services/dashboard'

const { Header, Content, Sider } = Layout
const { useBreakpoint } = Grid

const menuItems = [
  {
    key: '/datasets',
    icon: <AppstoreOutlined />,
    label: '数据集中心',
  },
  {
    key: '/chat',
    icon: <RobotOutlined />,
    label: '智能问答',
  },
  {
    key: '/settings',
    icon: <ControlOutlined />,
    label: '系统状态',
  },
  {
    key: '/reports',
    icon: <FileTextOutlined />,
    label: '报告中心',
  },
]

const titleMap: Record<string, string> = {
  '/datasets': '数据集与分析仪表盘',
  '/chat': '智能问答工作台',
  '/settings': '系统运行状态',
  '/reports': '报告导出与下载中心',
  '/llm-configs': 'LLM 配置管理',
}

export function AppShell({ children }: PropsWithChildren) {
  const location = useLocation()
  const navigate = useNavigate()
  const screens = useBreakpoint()
  const isDesktop = screens.lg ?? false
  const [mobileNavOpen, setMobileNavOpen] = useState(false)

  const selectedKey = menuItems.find((item) => location.pathname.startsWith(item.key))?.key ?? '/datasets'
  const titleKey = Object.keys(titleMap).find((item) => location.pathname.startsWith(item))
  const runtimeLabel = getRuntimeModeLabel()
  const currentTitle = titleMap[titleKey ?? selectedKey] ?? '前端控制台'

  const handleNavigate = (key: string) => {
    navigate(key)
    setMobileNavOpen(false)
  }

  const renderNavigation = () => (
    <div className="app-shell__navigation">
      <div className="brand-block">
        <div className="brand-block__mark">
          <DeploymentUnitOutlined />
        </div>
        <Typography.Text className="brand-block__eyebrow">居民用电分析</Typography.Text>
        <Typography.Title className="brand-block__title" level={3}>
          Energy Canvas
        </Typography.Title>
        <Typography.Paragraph className="brand-block__desc">
          本地部署的居民用电分析台。
        </Typography.Paragraph>
        <div className="brand-block__chips">
          <span className={`brand-chip ${isMockMode ? 'brand-chip--soft' : 'brand-chip--accent'}`}>
            {runtimeLabel}
          </span>
        </div>
      </div>

      <Menu
        className="app-shell__menu"
        mode="inline"
        selectedKeys={[selectedKey]}
        items={menuItems}
        onClick={({ key }) => handleNavigate(String(key))}
      />

      <div className="sider-tip">
        <ThunderboltOutlined />
        <span>导航保持收敛，只保留分析流程里真正高频的入口。</span>
      </div>
    </div>
  )

  return (
    <Layout className="app-shell">
      {isDesktop ? (
        <Sider className="app-shell__sider" width={296}>
          {renderNavigation()}
        </Sider>
      ) : null}

      <Drawer
        className="app-shell__drawer"
        closable={false}
        placement="left"
        open={!isDesktop && mobileNavOpen}
        width={320}
        onClose={() => setMobileNavOpen(false)}
      >
        {renderNavigation()}
      </Drawer>

      <Layout className="app-shell__main">
        <Header className="app-shell__header">
          <div className="app-shell__header-frame">
            <div className="app-shell__header-path">
              {!isDesktop ? (
                <Button
                  className="app-shell__menu-button"
                  icon={<MenuOutlined />}
                  onClick={() => setMobileNavOpen(true)}
                />
              ) : null}
              <Typography.Text className="app-shell__header-label">
                当前路径
              </Typography.Text>
              <span className="app-shell__header-separator">/</span>
              <Typography.Title className="app-shell__header-title" level={4}>
                {currentTitle}
              </Typography.Title>
            </div>
          </div>
        </Header>

        <Content className="app-shell__content">
          <div className="app-shell__content-frame">{children}</div>
        </Content>
      </Layout>
    </Layout>
  )
}
