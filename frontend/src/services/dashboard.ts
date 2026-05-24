import axios from 'axios'
import type { ApiEnvelope, Pagination } from '@/types/api'
import type {
  AnalysisPayload,
  AssistantAnswer,
  ChatMessage,
  ChatSession,
  DatasetDetailPayload,
  DatasetListPayload,
  DatasetListQuery,
  DatasetSummary,
  ForecastDetail,
  ForecastPredictInput,
  ForecastRecord,
  HealthStatus,
  ImportDatasetInput,
  ReportRecord,
  ReportType,
  SystemConfig,
  SystemConfigPatchInput,
  ClassificationResult,
  DetectionResult,
} from '@/types/domain'

const http = axios.create({
  baseURL: import.meta.env.VITE_API_PREFIX || '/api/v1',
  timeout: 10000,
})

const assistantRequestTimeoutMs = 200000

export function extractApiErrorMessage(error: unknown, fallback: string): string {
  if (axios.isAxiosError(error)) {
    if (error.code === 'ECONNABORTED') {
      return '请求处理时间较长，前端已超时。请稍后重试或检查模型服务日志。'
    }

    const responseMessage = error.response?.data?.message
    if (typeof responseMessage === 'string' && responseMessage.trim()) {
      return responseMessage.trim()
    }

    const responseError = error.response?.data?.error
    if (typeof responseError === 'string' && responseError.trim()) {
      return responseError.trim()
    }

    if (typeof error.message === 'string' && error.message.trim()) {
      return error.message.trim()
    }
  }

  if (error instanceof Error && error.message.trim()) {
    return error.message.trim()
  }

  return fallback
}

export async function fetchHealth(): Promise<HealthStatus> {
  const { data } = await http.get<ApiEnvelope<HealthStatus>>('/health')
  return data.data
}

export async function fetchSystemConfig(): Promise<SystemConfig> {
  const { data } = await http.get<ApiEnvelope<SystemConfig>>('/system/config')
  return data.data
}

export async function updateSystemConfig(
  input: SystemConfigPatchInput,
): Promise<SystemConfig> {
  const { data } = await http.patch<ApiEnvelope<SystemConfig>>('/system/config', input)
  return data.data
}

export async function fetchDatasetList(
  query: DatasetListQuery,
): Promise<DatasetListPayload> {
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
  const { data } = await http.get<ApiEnvelope<DatasetDetailPayload>>(`/datasets/${datasetId}`)
  return data.data
}

export async function fetchDatasetAnalysis(datasetId: number): Promise<AnalysisPayload> {
  const { data } = await http.get<ApiEnvelope<AnalysisPayload>>(`/datasets/${datasetId}/analysis`)
  return data.data
}

export async function fetchLatestClassification(
  datasetId: number,
): Promise<ClassificationResult | null> {
  try {
    const { data } = await http.get<ApiEnvelope<{ classification: ClassificationResult }>>(
      `/datasets/${datasetId}/classifications/latest`,
      {
        params: { model_type: 'xgboost' },
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

export async function fetchClassifications(datasetId: number): Promise<ClassificationResult[]> {
  const { data } = await http.get<
    ApiEnvelope<{
      items: ClassificationResult[]
    }>
  >(`/datasets/${datasetId}/classifications`, {
    params: { model_type: 'xgboost', page: 1, page_size: 365 },
  })
  return data.data.items
}

export async function fetchCurrentDetection(
  datasetId: number,
): Promise<DetectionResult | null> {
  try {
    const { data } = await http.get<ApiEnvelope<{ detection: DetectionResult }>>(
      `/datasets/${datasetId}/detections/current`,
    )
    return data.data.detection
  } catch (error) {
    if (axios.isAxiosError(error) && error.response?.status === 404) {
      return null
    }
    throw error
  }
}

export async function runDetection(datasetId: number): Promise<DetectionResult> {
  const { data } = await http.post<ApiEnvelope<{ detection: DetectionResult }>>(
    `/datasets/${datasetId}/detections/detect`,
  )
  return data.data.detection
}

export async function fetchReports(datasetId: number): Promise<ReportRecord[]> {
  const { data } = await http.get<ApiEnvelope<{ items: ReportRecord[] }>>(
    `/datasets/${datasetId}/reports`,
  )
  return data.data.items
}

export async function exportDatasetReport(
  datasetId: number,
  reportType: ReportType,
): Promise<ReportRecord> {
  const { data } = await http.post<ApiEnvelope<{ report: ReportRecord }>>(
    `/datasets/${datasetId}/reports/export`,
    { report_type: reportType },
    { timeout: 360000 },
  )
  return data.data.report
}

export async function downloadReport(report: ReportRecord): Promise<void> {
  window.open(`/api/v1/reports/${report.id}/download`, '_blank', 'noopener,noreferrer')
}

export async function fetchForecasts(datasetId: number): Promise<ForecastRecord[]> {
  const { data } = await http.get<ApiEnvelope<{ items: ForecastRecord[]; pagination: Pagination }>>(
    `/datasets/${datasetId}/forecasts`,
    {
      params: { page: 1, page_size: 20 },
    },
  )
  return data.data.items
}

export async function fetchForecastDetail(forecastId: number): Promise<ForecastDetail> {
  const { data } = await http.get<ApiEnvelope<ForecastDetail>>(`/forecasts/${forecastId}`)
  return data.data
}

export async function runForecast(
  datasetId: number,
  input: ForecastPredictInput,
): Promise<ForecastRecord> {
  const { data } = await http.post<ApiEnvelope<{ forecast: ForecastRecord }>>(
    `/datasets/${datasetId}/forecasts/predict`,
    input,
  )
  return data.data.forecast
}

export async function fetchChatSessions(datasetId: number): Promise<ChatSession[]> {
  const { data } = await http.get<ApiEnvelope<{ items: ChatSession[] }>>('/chat/sessions', {
    params: { page: 1, page_size: 20, dataset_id: datasetId },
  })
  return data.data.items
}

export async function createChatSession(input: {
  dataset_id: number
  title: string
}): Promise<ChatSession> {
  const { data } = await http.post<ApiEnvelope<{ session: ChatSession }>>(
    '/chat/sessions',
    input,
  )
  return data.data.session
}

export async function fetchChatMessages(sessionId: number): Promise<ChatMessage[]> {
  const { data } = await http.get<ApiEnvelope<{ items: ChatMessage[] }>>(
    `/chat/sessions/${sessionId}/messages`,
    {
      params: { page: 1, page_size: 50 },
    },
  )
  return data.data.items
}

export async function importDataset(input: ImportDatasetInput): Promise<void> {
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
  const { data } = await http.post<ApiEnvelope<AssistantAnswer>>('/agent/ask', input, {
    timeout: assistantRequestTimeoutMs,
  })
  return data.data
}

export async function runClassification(datasetId: number): Promise<ClassificationResult> {
  const { data } = await http.post<ApiEnvelope<{ classification: ClassificationResult }>>(
    `/datasets/${datasetId}/classifications/predict`,
    {
      model_type: 'xgboost',
      window_mode: 'full_dataset',
      force_refresh: true,
    },
  )
  return data.data.classification
}
