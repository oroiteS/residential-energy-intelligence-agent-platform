import { Suspense, lazy } from 'react'
import { Spin } from 'antd'
import { Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from '@/app/AppShell'
import './App.css'

const DatasetsPage = lazy(() =>
  import('@/pages/DatasetsPage').then((module) => ({ default: module.DatasetsPage })),
)
const DatasetDetailPage = lazy(() =>
  import('@/pages/DatasetDetailPage').then((module) => ({
    default: module.DatasetDetailPage,
  })),
)
const ChatPage = lazy(() =>
  import('@/pages/ChatPage').then((module) => ({ default: module.ChatPage })),
)
const SettingsPage = lazy(() =>
  import('@/pages/SettingsPage').then((module) => ({ default: module.SettingsPage })),
)
const ReportsPage = lazy(() =>
  import('@/pages/ReportsPage').then((module) => ({ default: module.ReportsPage })),
)
const LlmConfigsPage = lazy(() =>
  import('@/pages/LlmConfigsPage').then((module) => ({ default: module.LlmConfigsPage })),
)
const NotFoundPage = lazy(() =>
  import('@/pages/NotFoundPage').then((module) => ({ default: module.NotFoundPage })),
)

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
        <Routes>
          <Route path="/" element={<Navigate to="/datasets" replace />} />
          <Route path="/datasets" element={<DatasetsPage />} />
          <Route path="/datasets/:datasetId" element={<DatasetDetailPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/reports" element={<ReportsPage />} />
          <Route path="/llm-configs" element={<LlmConfigsPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </Suspense>
    </AppShell>
  )
}

export default App
