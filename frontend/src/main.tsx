import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ConfigProvider } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import { BrowserRouter } from 'react-router-dom'
import 'antd/dist/reset.css'
import './index.css'
import App from './App'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ConfigProvider
      locale={zhCN}
      theme={{
        token: {
          colorPrimary: '#4e5d4f',
          colorInfo: '#4e5d4f',
          colorSuccess: '#5f715f',
          colorWarning: '#9e866b',
          colorError: '#8b6759',
          colorBgBase: '#f2eee7',
          colorTextBase: '#202721',
          colorBorder: 'rgba(66, 78, 67, 0.16)',
          borderRadius: 22,
          fontFamily:
            '"Manrope", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif',
        },
      }}
    >
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </ConfigProvider>
  </StrictMode>,
)
