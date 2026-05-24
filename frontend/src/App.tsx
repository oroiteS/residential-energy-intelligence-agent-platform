import { Suspense } from 'react'
import { Spin } from 'antd'
import { AppShell } from '@/app/AppShell'
import { AppRoutes } from '@/app/routes'
import './App.css'

function App() {
  return (
    <AppShell>
      <Suspense
        fallback={
          <div className="page-state">
            <Spin size="large" />
          </div>
        }
      >
        <AppRoutes />
      </Suspense>
    </AppShell>
  )
}

export default App
