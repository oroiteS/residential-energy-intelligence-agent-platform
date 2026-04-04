import { create } from 'zustand'
import {
  buildMockAssistantExchange,
  demoAdvices,
  demoAnalyses,
  demoBacktests,
  demoChatMessages,
  demoChatSessions,
  demoClassifications,
  demoDatasetDetails,
  demoDatasets,
  demoForecastDetails,
  demoForecasts,
  demoLlmConfigs,
  demoReports,
  demoSystemConfig,
} from '@/mocks/demoData'
import type {
  AnalysisPayload,
  ChatSession,
  DatasetDetailPayload,
  ForecastBacktest,
  ForecastDetail,
  LlmConfig,
  LlmConfigInput,
  ReportRecord,
  SystemConfig,
  SystemConfigPatchInput,
} from '@/types/domain'

type MockStore = {
  datasets: typeof demoDatasets
  datasetDetails: typeof demoDatasetDetails
  analyses: typeof demoAnalyses
  classifications: typeof demoClassifications
  advices: typeof demoAdvices
  reports: Record<number, ReportRecord[]>
  forecasts: Record<number, typeof demoForecasts[number]>
  forecastDetails: Record<number, ForecastDetail>
  backtests: Record<string, ForecastBacktest>
  systemConfig: SystemConfig
  llmConfigs: LlmConfig[]
  chatSessions: Record<number, ChatSession[]>
  chatMessages: typeof demoChatMessages
  addDataset: (detail: DatasetDetailPayload, analysis: AnalysisPayload) => void
  addForecast: (detail: ForecastDetail) => void
  upsertBacktest: (payload: ForecastBacktest) => void
  addReport: (report: ReportRecord) => void
  updateSystemConfig: (input: SystemConfigPatchInput) => SystemConfig
  createLlmConfig: (input: LlmConfigInput) => LlmConfig
  updateLlmConfig: (id: number, input: LlmConfigInput) => LlmConfig
  deleteLlmConfig: (id: number) => void
  setDefaultLlmConfig: (id: number) => void
  createChatSession: (datasetId: number, title: string) => ChatSession
  appendChatExchange: (
    datasetId: number,
    sessionId: number,
    question: string,
  ) => ReturnType<typeof buildMockAssistantExchange>['answer']
}

function getNextId(values: number[]) {
  return Math.max(0, ...values) + 1
}

export const useMockStore = create<MockStore>((set, get) => ({
  datasets: demoDatasets,
  datasetDetails: demoDatasetDetails,
  analyses: demoAnalyses,
  classifications: demoClassifications,
  advices: demoAdvices,
  reports: demoReports,
  forecasts: demoForecasts,
  forecastDetails: demoForecastDetails,
  backtests: demoBacktests,
  systemConfig: demoSystemConfig,
  llmConfigs: demoLlmConfigs,
  chatSessions: demoChatSessions,
  chatMessages: demoChatMessages,
  addDataset: (detail, analysis) =>
    set((state) => ({
      datasets: [detail.dataset, ...state.datasets],
      datasetDetails: {
        ...state.datasetDetails,
        [detail.dataset.id]: detail,
      },
      analyses: {
        ...state.analyses,
        [detail.dataset.id]: analysis,
      },
    })),
  addForecast: (detail) =>
    set((state) => ({
      forecasts: {
        ...state.forecasts,
        [detail.forecast.dataset_id]: [detail.forecast, ...(state.forecasts[detail.forecast.dataset_id] ?? [])],
      },
      forecastDetails: {
        ...state.forecastDetails,
        [detail.forecast.id]: detail,
      },
    })),
  upsertBacktest: (payload) =>
    set((state) => ({
      backtests: {
        ...state.backtests,
        [`${payload.backtest.dataset_id}:${payload.backtest.model_type}`]: payload,
      },
    })),
  addReport: (report) =>
    set((state) => ({
      reports: {
        ...state.reports,
        [report.dataset_id]: [report, ...(state.reports[report.dataset_id] ?? [])],
      },
    })),
  updateSystemConfig: (input) => {
    set((state) => {
      const nextConfig: SystemConfig = {
        ...state.systemConfig,
        ...input,
        peak_valley_config: input.peak_valley_config
          ? {
              peak: [...input.peak_valley_config.peak],
              valley: [...input.peak_valley_config.valley],
            }
          : state.systemConfig.peak_valley_config,
        model_history_window_config: input.model_history_window_config
          ? {
              ...state.systemConfig.model_history_window_config,
              ...input.model_history_window_config,
            }
          : state.systemConfig.model_history_window_config,
      }

      return {
        systemConfig: nextConfig,
      }
    })

    return get().systemConfig
  },
  createLlmConfig: (input) => {
    const nextId = getNextId(get().llmConfigs.map((item) => item.id))
    const timestamp = '2026-04-01T11:10:00+08:00'
    const record: LlmConfig = {
      id: nextId,
      name: input.name,
      base_url: input.base_url,
      model_name: input.model_name,
      temperature: input.temperature,
      timeout_seconds: input.timeout_seconds,
      is_default: input.is_default,
      created_at: timestamp,
      updated_at: timestamp,
    }

    set((state) => ({
      llmConfigs: state.llmConfigs.map((item) => ({
        ...item,
        is_default: input.is_default ? false : item.is_default,
      })).concat(record),
    }))

    return record
  },
  updateLlmConfig: (id, input) => {
    const timestamp = '2026-04-01T11:16:00+08:00'
    let updatedRecord: LlmConfig | null = null

    set((state) => ({
      llmConfigs: state.llmConfigs.map((item) => {
        if (item.id === id) {
          updatedRecord = {
            ...item,
            name: input.name,
            base_url: input.base_url,
            model_name: input.model_name,
            temperature: input.temperature,
            timeout_seconds: input.timeout_seconds,
            is_default: input.is_default,
            updated_at: timestamp,
          }
          return updatedRecord
        }

        return {
          ...item,
          is_default: input.is_default ? false : item.is_default,
        }
      }),
    }))

    if (!updatedRecord) {
      throw new Error('LLM_CONFIG_NOT_FOUND')
    }

    return updatedRecord
  },
  deleteLlmConfig: (id) =>
    set((state) => ({
      llmConfigs: state.llmConfigs.filter((item) => item.id !== id),
    })),
  setDefaultLlmConfig: (id) =>
    set((state) => ({
      llmConfigs: state.llmConfigs.map((item) => ({
        ...item,
        is_default: item.id === id,
      })),
    })),
  createChatSession: (datasetId, title) => {
    const nextId = getNextId(
      Object.values(get().chatSessions)
        .flat()
        .map((item) => item.id),
    )
    const timestamp = '2026-04-01T11:32:00+08:00'
    const session: ChatSession = {
      id: nextId,
      dataset_id: datasetId,
      title,
      created_at: timestamp,
      updated_at: timestamp,
    }

    set((state) => ({
      chatSessions: {
        ...state.chatSessions,
        [datasetId]: [session, ...(state.chatSessions[datasetId] ?? [])],
      },
      chatMessages: {
        ...state.chatMessages,
        [session.id]: [],
      },
    }))

    return session
  },
  appendChatExchange: (datasetId, sessionId, question) => {
    const existingMessageIds = Object.values(get().chatMessages)
      .flat()
      .map((item) => item.id)
    const nextMessageId = getNextId(existingMessageIds)
    const payload = buildMockAssistantExchange(datasetId, sessionId, question, nextMessageId)

    set((state) => ({
      chatMessages: {
        ...state.chatMessages,
        [sessionId]: [...(state.chatMessages[sessionId] ?? []), ...payload.messages],
      },
      chatSessions: {
        ...state.chatSessions,
        [datasetId]: (state.chatSessions[datasetId] ?? []).map((item) =>
          item.id === sessionId
            ? { ...item, updated_at: payload.answer.created_at }
            : item,
        ),
      },
    }))

    return payload.answer
  },
}))
