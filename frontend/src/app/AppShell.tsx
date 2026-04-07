import { useState, type PropsWithChildren } from 'react'
import {
  AppstoreOutlined,
  ControlOutlined,
  DeploymentUnitOutlined,
  FileTextOutlined,
  FundProjectionScreenOutlined,
  MenuOutlined,
  RobotOutlined,
} from '@ant-design/icons'
import { Button, Drawer, Grid, Layout, Menu, Typography } from 'antd'
import { useLocation, useNavigate } from 'react-router-dom'

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
    label: '节能问答',
  },
  {
    key: '/overview',
    icon: <ControlOutlined />,
    label: '服务概览',
  },
  {
    key: '/live',
    icon: <FundProjectionScreenOutlined />,
    label: '实时演示',
  },
  {
    key: '/reports',
    icon: <FileTextOutlined />,
    label: '报告中心',
  },
]

const titleMap: Record<string, string> = {
  '/datasets': '数据集中心',
  '/chat': '节能问答',
  '/overview': '服务概览',
  '/live': '实时演示',
  '/reports': '报告中心',
}

export function AppShell({ children }: PropsWithChildren) {
  const location = useLocation()
  const navigate = useNavigate()
  const screens = useBreakpoint()
  const isDesktop = screens.lg ?? false
  const [mobileNavOpen, setMobileNavOpen] = useState(false)

  const selectedKey = menuItems.find((item) => location.pathname.startsWith(item.key))?.key ?? '/datasets'
  const titleKey = Object.keys(titleMap).find((item) => location.pathname.startsWith(item))
  const currentTitle = titleMap[titleKey ?? selectedKey] ?? 'Energy Canvas'

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
          居民用电工作台
        </Typography.Title>
        <Typography.Paragraph className="brand-block__desc">
          围绕数据接入、负荷洞察、预测与节能建议的统一工作区。
        </Typography.Paragraph>
        <div className="brand-block__chips">
          <span className="brand-chip brand-chip--accent">分析</span>
          <span className="brand-chip">预测</span>
          <span className="brand-chip brand-chip--soft">问答</span>
        </div>
      </div>

      <Menu
        className="app-shell__menu"
        mode="inline"
        selectedKeys={[selectedKey]}
        items={menuItems}
        onClick={({ key }) => handleNavigate(String(key))}
      />

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
                  aria-label="打开导航菜单"
                  icon={<MenuOutlined />}
                  onClick={() => setMobileNavOpen(true)}
                />
              ) : null}
              <Typography.Text className="app-shell__header-label">
                工作区
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
