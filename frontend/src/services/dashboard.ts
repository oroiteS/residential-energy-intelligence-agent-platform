import axios from 'axios'
import {
  buildMockImportedAnalysis,
  buildMockImportedBacktest,
  buildMockImportedDataset,
  buildMockImportedForecast,
  buildMockImportedReport,
  demoHealthStatus,
} from '@/mocks/demoData'
import { useMockStore } from '@/store/mockStore'
import type { ApiEnvelope, Pagination } from '@/types/api'
import type {
  AdviceDetail,
  AnalysisPayload,
  AssistantAnswer,
  ChatMessage,
  ChatSession,
  DatasetDetailPayload,
  DatasetListPayload,
  DatasetListQuery,
  DatasetSummary,
  ForecastBacktest,
  ForecastBacktestInput,
  ForecastDetail,
  ForecastPredictInput,
  ForecastRecord,
  HealthStatus,
  ImportDatasetInput,
  LlmConfig,
  LlmConfigInput,
  ReportRecord,
  ReportType,
  SystemConfig,
  SystemConfigPatchInput,
  ClassificationResult,
} from '@/types/domain'

const http = axios.create({
  baseURL: '/api/v1',
  timeout: 10000,
})

export const isMockMode =
  import.meta.env.DEV && import.meta.env.VITE_USE_MOCK !== 'false'

export function getRuntimeModeLabel() {
  return isMockMode ? 'Mock 联调模式' : '实时接口模式'
}

function sleep(ms = 280) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms)
  })
}

function downloadBlob(filename: string, content: string) {
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}

export async function fetchHealth(): Promise<HealthStatus> {
  if (isMockMode) {
    await sleep()
    return demoHealthStatus
  }

  const { data } = await http.get<ApiEnvelope<HealthStatus>>('/health')
  return data.data
}

export async function fetchSystemConfig(): Promise<SystemConfig> {
  if (isMockMode) {
    await sleep()
    return useMockStore.getState().systemConfig
  }

  const { data } = await http.get<ApiEnvelope<SystemConfig>>('/system/config')
  return data.data
}

export async function updateSystemConfig(
  input: SystemConfigPatchInput,
): Promise<SystemConfig> {
  if (isMockMode) {
    await sleep(240)
    return useMockStore.getState().updateSystemConfig(input)
  }

  const { data } = await http.patch<ApiEnvelope<SystemConfig>>('/system/config', input)
  return data.data
}

function buildDatasetListPayload(
  items: DatasetSummary[],
  query: DatasetListQuery,
): DatasetListPayload {
  const keyword = query.keyword?.trim().toLowerCase() ?? ''
  const filteredItems = items.filter((item) => {
    if (query.status && item.status !== query.status) {
      return false
    }

    if (!keyword) {
      return true
    }

    const searchTarget = `${item.name} ${item.description ?? ''} ${item.household_id ?? ''}`.toLowerCase()
    return searchTarget.includes(keyword)
  })

  const safePage = Math.max(1, query.page)
  const safePageSize = Math.max(1, query.page_size)
  const startIndex = (safePage - 1) * safePageSize

  return {
    items: filteredItems.slice(startIndex, startIndex + safePageSize),
    pagination: {
      page: safePage,
      page_size: safePageSize,
      total: filteredItems.length,
    },
  }
}

export async function fetchDatasetList(
  query: DatasetListQuery,
): Promise<DatasetListPayload> {
  if (isMockMode) {
    await sleep()
    return buildDatasetListPayload(useMockStore.getState().datasets, query)
  }

  const { data } = await http.get<ApiEnvelope<DatasetListPayload>>('/datasets', {
    params: {
      page: query.page,
      page_size: query.page_size,
      status: query.status,
      keyword: query.keyword?.trim() || undefined,
    },
  })
  return data.data
}

export async function fetchDatasets(): Promise<DatasetSummary[]> {
  const result = await fetchDatasetList({
    page: 1,
    page_size: 100,
  })
  return result.items
}

export async function fetchDatasetDetail(datasetId: number): Promise<DatasetDetailPayload> {
  if (isMockMode) {
    await sleep()
    const result = useMockStore.getState().datasetDetails[datasetId]
    if (!result) {
      throw new Error('DATASET_NOT_FOUND')
    }
    return result
  }

  const { data } = await http.get<ApiEnvelope<DatasetDetailPayload>>(`/datasets/${datasetId}`)
  return data.data
}

export async function fetchDatasetAnalysis(datasetId: number): Promise<AnalysisPayload> {
  if (isMockMode) {
    await sleep()
    const result = useMockStore.getState().analyses[datasetId]
    if (!result) {
      throw new Error('ANALYSIS_NOT_FOUND')
    }
    return result
  }

  const { data } = await http.get<ApiEnvelope<AnalysisPayload>>(`/datasets/${datasetId}/analysis`)
  return data.data
}

export async function fetchLatestClassification(
  datasetId: number,
): Promise<ClassificationResult | null> {
  if (isMockMode) {
    await sleep()
    return useMockStore.getState().classifications[datasetId] ?? null
  }

  try {
    const { data } = await http.get<ApiEnvelope<{ classification: ClassificationResult }>>(
      `/datasets/${datasetId}/classifications/latest`,
      {
        params: { model_type: 'tcn' },
      },
    )
    return data.data.classification
  } catch (error) {
    if (axios.isAxiosError(error) && error.response?.status === 404) {
      return null
    }
    throw error
  }
}

export async function fetchAdvices(datasetId: number): Promise<AdviceDetail[]> {
  if (isMockMode) {
    await sleep()
    return useMockStore.getState().advices[datasetId] ?? []
  }

  const { data } = await http.get<
    ApiEnvelope<{
      items: Array<AdviceDetail['advice']>
    }>
  >(`/datasets/${datasetId}/advices`, {
    params: { advice_type: 'rule' },
  })

  const details = await Promise.all(
    data.data.items.map(async (item) => {
      const detailResponse = await http.get<ApiEnvelope<AdviceDetail>>(`/advices/${item.id}`)
      return detailResponse.data.data
    }),
  )

  return details
}

export async function fetchReports(datasetId: number): Promise<ReportRecord[]> {
  if (isMockMode) {
    await sleep()
    return useMockStore.getState().reports[datasetId] ?? []
  }

  const { data } = await http.get<ApiEnvelope<{ items: ReportRecord[] }>>(
    `/datasets/${datasetId}/reports`,
  )
  return data.data.items
}

export async function exportDatasetReport(
  datasetId: number,
  reportType: ReportType,
): Promise<ReportRecord> {
  if (isMockMode) {
    await sleep(360)
    const store = useMockStore.getState()
    const reportIds = Object.values(store.reports)
      .flat()
      .map((item) => item.id)
    const nextId = Math.max(0, ...reportIds) + 1
    const report = buildMockImportedReport(datasetId, reportType, nextId)
    store.addReport(report)
    return report
  }

  const { data } = await http.post<ApiEnvelope<{ report: ReportRecord }>>(
    `/datasets/${datasetId}/reports/export`,
    { report_type: reportType },
  )
  return data.data.report
}

export async function downloadReport(report: ReportRecord): Promise<void> {
  if (isMockMode) {
    await sleep(120)
    downloadBlob(`report_${report.id}.txt`, `模拟下载文件\n${report.file_path}`)
    return
  }

  window.open(`/api/v1/reports/${report.id}/download`, '_blank', 'noopener,noreferrer')
}

export async function fetchForecasts(datasetId: number): Promise<ForecastRecord[]> {
  if (isMockMode) {
    await sleep()
    return useMockStore.getState().forecasts[datasetId] ?? []
  }

  const { data } = await http.get<ApiEnvelope<{ items: ForecastRecord[]; pagination: Pagination }>>(
    `/datasets/${datasetId}/forecasts`,
    {
      params: { page: 1, page_size: 20 },
    },
  )
  return data.data.items
}

export async function fetchForecastDetail(forecastId: number): Promise<ForecastDetail> {
  if (isMockMode) {
    await sleep()
    const detail = useMockStore.getState().forecastDetails[forecastId]
    if (!detail) {
      throw new Error('FORECAST_NOT_FOUND')
    }
    return detail
  }

  const { data } = await http.get<ApiEnvelope<ForecastDetail>>(`/forecasts/${forecastId}`)
  return data.data
}

export async function runForecast(
  datasetId: number,
  input: ForecastPredictInput,
): Promise<ForecastRecord> {
  if (isMockMode) {
    await sleep(420)
    const store = useMockStore.getState()
    const forecastIds = Object.values(store.forecastDetails).map((item) => item.forecast.id)
    const nextId = Math.max(0, ...forecastIds) + 1
    const detail = buildMockImportedForecast(datasetId, input.model_type, nextId)
    store.addForecast(detail)
    return detail.forecast
  }

  const { data } = await http.post<ApiEnvelope<{ forecast: ForecastRecord }>>(
    `/datasets/${datasetId}/forecasts/predict`,
    input,
  )
  return data.data.forecast
}

export async function runForecastBacktest(
  datasetId: number,
  input: ForecastBacktestInput,
): Promise<ForecastBacktest> {
  if (isMockMode) {
    await sleep(420)
    const store = useMockStore.getState()
    const payload = buildMockImportedBacktest(datasetId, input.model_type)
    store.upsertBacktest(payload)
    return payload
  }

  const { data } = await http.post<ApiEnvelope<ForecastBacktest>>(
    `/datasets/${datasetId}/forecasts/backtest`,
    input,
  )
  return data.data
}

export async function fetchLlmConfigs(): Promise<LlmConfig[]> {
  if (isMockMode) {
    await sleep()
    return useMockStore.getState().llmConfigs
  }

  const { data } = await http.get<ApiEnvelope<{ items: LlmConfig[] }>>('/llm-configs')
  return data.data.items
}

export async function createLlmConfig(input: LlmConfigInput): Promise<LlmConfig> {
  if (isMockMode) {
    await sleep(240)
    return useMockStore.getState().createLlmConfig(input)
  }

  const { data } = await http.post<ApiEnvelope<LlmConfig>>('/llm-configs', input)
  return data.data
}

export async function updateLlmConfig(id: number, input: LlmConfigInput): Promise<LlmConfig> {
  if (isMockMode) {
    await sleep(240)
    return useMockStore.getState().updateLlmConfig(id, input)
  }

  const { data } = await http.put<ApiEnvelope<LlmConfig>>(`/llm-configs/${id}`, input)
  return data.data
}

export async function deleteLlmConfig(id: number): Promise<void> {
  if (isMockMode) {
    await sleep(180)
    useMockStore.getState().deleteLlmConfig(id)
    return
  }

  await http.delete(`/llm-configs/${id}`)
}

export async function setDefaultLlmConfig(id: number): Promise<void> {
  if (isMockMode) {
    await sleep(180)
    useMockStore.getState().setDefaultLlmConfig(id)
    return
  }

  await http.post(`/llm-configs/${id}/set-default`)
}

export async function fetchChatSessions(datasetId: number): Promise<ChatSession[]> {
  if (isMockMode) {
    await sleep()
    return useMockStore.getState().chatSessions[datasetId] ?? []
  }

  const { data } = await http.get<ApiEnvelope<{ items: ChatSession[] }>>('/chat/sessions', {
    params: { page: 1, page_size: 20, dataset_id: datasetId },
  })
  return data.data.items
}

export async function createChatSession(input: {
  dataset_id: number
  title: string
}): Promise<ChatSession> {
  if (isMockMode) {
    await sleep(240)
    return useMockStore.getState().createChatSession(input.dataset_id, input.title)
  }

  const { data } = await http.post<ApiEnvelope<{ session: ChatSession }>>(
    '/chat/sessions',
    input,
  )
  return data.data.session
}

export async function fetchChatMessages(sessionId: number): Promise<ChatMessage[]> {
  if (isMockMode) {
    await sleep()
    return useMockStore.getState().chatMessages[sessionId] ?? []
  }

  const { data } = await http.get<ApiEnvelope<{ items: ChatMessage[] }>>(
    `/chat/sessions/${sessionId}/messages`,
    {
      params: { page: 1, page_size: 50 },
    },
  )
  return data.data.items
}

export async function importDataset(input: ImportDatasetInput): Promise<void> {
  if (isMockMode) {
    await sleep(360)
    const store = useMockStore.getState()
    const nextId = Math.max(0, ...store.datasets.map((item) => item.id)) + 1
    const detail = buildMockImportedDataset(nextId, input)
    const analysis = buildMockImportedAnalysis(nextId)

    store.addDataset(detail, analysis)
    return
  }

  const formData = new FormData()
  formData.append('file', input.file)
  formData.append('name', input.name)
  if (input.description) {
    formData.append('description', input.description)
  }
  if (input.household_id) {
    formData.append('household_id', input.household_id)
  }
  formData.append('unit', input.unit)
  if (input.column_mapping) {
    formData.append('column_mapping', JSON.stringify(input.column_mapping))
  }

  await http.post<ApiEnvelope<{ dataset: DatasetSummary }>>('/datasets/import', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
}

export async function askAssistant(input: {
  dataset_id: number
  question: string
  session_id?: number
  history?: Array<{ role: 'user' | 'assistant'; content: string }>
}): Promise<AssistantAnswer> {
  if (isMockMode) {
    await sleep(420)
    if (!input.session_id) {
      throw new Error('SESSION_REQUIRED')
    }
    return useMockStore
      .getState()
      .appendChatExchange(input.dataset_id, input.session_id, input.question)
  }

  const { data } = await http.post<ApiEnvelope<AssistantAnswer>>('/agent/ask', input, {
    timeout: 70000,
  })
  return data.data
}

export async function runClassification(datasetId: number): Promise<ClassificationResult> {
  if (isMockMode) {
    await sleep(320)
    const result = useMockStore.getState().classifications[datasetId]
    if (!result) {
      throw new Error('CLASSIFICATION_NOT_FOUND')
    }
    return result
  }

  const { data } = await http.post<ApiEnvelope<{ classification: ClassificationResult }>>(
    `/datasets/${datasetId}/classifications/predict`,
    {
      model_type: 'tcn',
      window_mode: 'full_dataset',
      force_refresh: true,
    },
  )
  return data.data.classification
}

export async function generateAdvices(datasetId: number): Promise<void> {
  if (isMockMode) {
    await sleep(280)
    return
  }

  await http.post(`/datasets/${datasetId}/advices/generate`)
}
