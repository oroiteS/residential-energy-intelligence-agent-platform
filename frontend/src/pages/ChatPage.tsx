import { useCallback, useEffect, useMemo, useState, useTransition } from 'react'
import {
  AimOutlined,
  ClockCircleOutlined,
  CompassOutlined,
  HistoryOutlined,
  MessageOutlined,
  PlusOutlined,
  RadarChartOutlined,
  SafetyCertificateOutlined,
  SendOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
import {
  Alert,
  Button,
  Col,
  Empty,
  Input,
  List,
  Row,
  Select,
  Space,
  Spin,
  Tag,
  Typography,
  message,
} from 'antd'
import { useSearchParams } from 'react-router-dom'
import { PageHero } from '@/components/common/PageHero'
import { SectionCard } from '@/components/common/SectionCard'
import {
  assistantConfidenceMap,
  assistantIntentMap,
} from '@/constants/display'
import {
  askAssistant,
  createChatSession,
  extractApiErrorMessage,
  fetchChatMessages,
  fetchChatSessions,
  fetchDatasets,
} from '@/services/dashboard'
import type {
  AssistantAnswer,
  ChatMessage,
  ChatSession,
  DatasetSummary,
} from '@/types/domain'
import { formatDateTime } from '@/utils/formatters'

const { TextArea } = Input

function renderCitationValue(value: AssistantAnswer['citations'][number]['value']) {
  if (Array.isArray(value)) {
    return value.map((item) => String(item)).join(' / ')
  }
  return String(value)
}

function isAssistantAnswerPayload(value: unknown): value is AssistantAnswer {
  return (
    typeof value === 'object' &&
    value !== null &&
    'answer' in value &&
    typeof (value as { answer?: unknown }).answer === 'string'
  )
}

function AnswerSignalTags({ answer }: { answer: AssistantAnswer }) {
  return (
    <Space wrap size={[8, 8]}>
      {answer.intent ? (
        <Tag className="tone-tag tone-tag--accent">
          <CompassOutlined />
          {assistantIntentMap[answer.intent] ?? answer.intent}
        </Tag>
      ) : null}
      {answer.confidence_level ? (
        <Tag color={assistantConfidenceMap[answer.confidence_level].color}>
          <SafetyCertificateOutlined />
          {assistantConfidenceMap[answer.confidence_level].label}
        </Tag>
      ) : null}
      {answer.degraded ? (
        <Tag color="warning">
          <RadarChartOutlined />
          降级回答
        </Tag>
      ) : null}
      {answer.created_at ? (
        <Tag className="tone-tag tone-tag--muted">
          <ClockCircleOutlined />
          {formatDateTime(answer.created_at)}
        </Tag>
      ) : null}
    </Space>
  )
}

function AssistantResponsePanel({
  answer,
}: {
  answer: AssistantAnswer | null
}) {
  if (!answer) {
    return (
      <div className="assistant-response assistant-response--empty">
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description="发送问题后，这里会展示 agent 的结构化回答、引用依据和建议动作。"
        />
      </div>
    )
  }

  return (
    <div className="assistant-response">
      <div className="assistant-response__header">
        <div>
          <Typography.Text className="assistant-response__eyebrow">
            Agent 输出
          </Typography.Text>
          <Typography.Title className="assistant-response__title" level={4}>
            本轮结构化回答
          </Typography.Title>
        </div>
        <AnswerSignalTags answer={answer} />
      </div>

      {answer.degraded ? (
        <Alert
          type="warning"
          showIcon
          message="当前回答为降级结果"
          description={
            answer.error_reason
              ? `后端返回降级标记：${answer.error_reason}`
              : '部分能力暂不可用，建议结合分析页与报告页交叉确认。'
          }
        />
      ) : null}

      <div className="assistant-response__body">
        <Typography.Paragraph className="assistant-response__answer">
          {answer.answer}
        </Typography.Paragraph>
      </div>

      {answer.missing_information?.length ? (
        <Alert
          type="info"
          showIcon
          message="Agent 仍需要更多信息"
          description={
            <div className="assistant-response__missing">
              {answer.missing_information.map((item) => (
                <div key={item.key} className="assistant-response__missing-item">
                  <Typography.Text strong>{item.question}</Typography.Text>
                  <Typography.Paragraph>
                    {item.reason}
                  </Typography.Paragraph>
                </div>
              ))}
            </div>
          }
        />
      ) : null}

      <div className="assistant-response__grid">
        <div className="assistant-response__panel">
          <Typography.Text className="assistant-response__panel-title">
            引用依据
          </Typography.Text>
          {answer.citations.length ? (
            <div className="assistant-citation-grid">
              {answer.citations.map((citation) => (
                <div key={citation.key} className="assistant-citation-card">
                  <Typography.Text className="assistant-citation-card__label">
                    {citation.label}
                  </Typography.Text>
                  <Typography.Paragraph className="assistant-citation-card__value">
                    {renderCitationValue(citation.value)}
                  </Typography.Paragraph>
                </div>
              ))}
            </div>
          ) : (
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description="当前没有返回引用依据"
            />
          )}
        </div>

        <div className="assistant-response__panel">
          <Typography.Text className="assistant-response__panel-title">
            建议动作
          </Typography.Text>
          <List
            className="assistant-action-list"
            size="small"
            dataSource={answer.actions}
            locale={{ emptyText: '当前没有返回建议动作' }}
            renderItem={(item, index) => (
              <List.Item>
                <div className="assistant-action-item">
                  <span className="assistant-action-item__index">
                    {String(index + 1).padStart(2, '0')}
                  </span>
                  <Typography.Text>{item}</Typography.Text>
                </div>
              </List.Item>
            )}
          />
        </div>
      </div>
    </div>
  )
}

function AssistantMessageHistoryPanel({ answer }: { answer: AssistantAnswer }) {
  return (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      <AnswerSignalTags answer={answer} />

      {answer.missing_information?.length ? (
        <Alert
          type="info"
          showIcon
          message="仍需补充的信息"
          description={
            <div>
              {answer.missing_information.map((item) => (
                <Typography.Paragraph key={item.key} style={{ marginBottom: 8 }}>
                  {item.question}：{item.reason}
                </Typography.Paragraph>
              ))}
            </div>
          }
        />
      ) : null}

      {answer.citations.length ? (
        <div>
          <Typography.Text strong>引用依据</Typography.Text>
          <div className="assistant-citation-grid" style={{ marginTop: 8 }}>
            {answer.citations.map((citation) => (
              <div key={citation.key} className="assistant-citation-card">
                <Typography.Text className="assistant-citation-card__label">
                  {citation.label}
                </Typography.Text>
                <Typography.Paragraph className="assistant-citation-card__value">
                  {renderCitationValue(citation.value)}
                </Typography.Paragraph>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {answer.actions.length ? (
        <div>
          <Typography.Text strong>建议动作</Typography.Text>
          <List
            className="assistant-action-list"
            size="small"
            style={{ marginTop: 8 }}
            dataSource={answer.actions}
            renderItem={(item, index) => (
              <List.Item>
                <div className="assistant-action-item">
                  <span className="assistant-action-item__index">
                    {String(index + 1).padStart(2, '0')}
                  </span>
                  <Typography.Text>{item}</Typography.Text>
                </div>
              </List.Item>
            )}
          />
        </div>
      ) : null}
    </Space>
  )
}

export function ChatPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [datasets, setDatasets] = useState<DatasetSummary[]>([])
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [answerBySessionId, setAnswerBySessionId] = useState<
    Record<number, AssistantAnswer>
  >({})
  const [loading, setLoading] = useState(true)
  const [sessionsLoading, setSessionsLoading] = useState(false)
  const [messagesLoading, setMessagesLoading] = useState(false)
  const [asking, setAsking] = useState(false)
  const [question, setQuestion] = useState('')
  const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(null)
  const [activeSessionId, setActiveSessionId] = useState<number | null>(null)
  const [, startTransition] = useTransition()

  const readyDatasets = useMemo(
    () => datasets.filter((item) => item.status === 'ready'),
    [datasets],
  )

  const selectedDataset = useMemo(
    () => readyDatasets.find((item) => item.id === selectedDatasetId) ?? null,
    [readyDatasets, selectedDatasetId],
  )

  const activeAnswer = useMemo(
    () => (activeSessionId ? answerBySessionId[activeSessionId] ?? null : null),
    [activeSessionId, answerBySessionId],
  )

  const loadSessions = useCallback(async (datasetId: number) => {
    setSessionsLoading(true)
    try {
      const result = await fetchChatSessions(datasetId)
      startTransition(() => {
        setSessions(result)
        setActiveSessionId((current) => {
          if (current && result.some((item) => item.id === current)) {
            return current
          }
          return result[0]?.id ?? null
        })
      })
    } catch (error) {
      message.error(extractApiErrorMessage(error, '会话列表加载失败。'))
    } finally {
      setSessionsLoading(false)
    }
  }, [startTransition])

  useEffect(() => {
    let active = true

    const loadPage = async () => {
      setLoading(true)
      try {
        const datasetList = await fetchDatasets()
        if (!active) {
          return
        }

        const availableDatasets = datasetList.filter((item) => item.status === 'ready')
        const queryDatasetId = Number(searchParams.get('dataset'))
        const initialDatasetId =
          availableDatasets.find((item) => item.id === queryDatasetId)?.id ??
          availableDatasets[0]?.id ??
          null

        startTransition(() => {
          setDatasets(datasetList)
          setSelectedDatasetId(initialDatasetId)
        })
      } catch (error) {
        if (active) {
          message.error(extractApiErrorMessage(error, '聊天页初始化失败。'))
        }
      } finally {
        if (active) {
          setLoading(false)
        }
      }
    }

    void loadPage()

    return () => {
      active = false
    }
  }, [searchParams, startTransition])

  useEffect(() => {
    if (!selectedDatasetId) {
      setSessions([])
      setActiveSessionId(null)
      setMessages([])
      return
    }

    setSearchParams({ dataset: String(selectedDatasetId) }, { replace: true })
    void loadSessions(selectedDatasetId)
  }, [loadSessions, selectedDatasetId, setSearchParams])

  useEffect(() => {
    if (!activeSessionId) {
      setMessages([])
      return
    }

    let active = true

    const loadMessages = async () => {
      setMessagesLoading(true)
      try {
        const nextMessages = await fetchChatMessages(activeSessionId)
        if (active) {
          const latestAssistantPayload = [...nextMessages]
            .reverse()
            .map((item) => item.assistant_payload)
            .find((item) => isAssistantAnswerPayload(item)) ?? null
          startTransition(() => {
            setMessages(nextMessages)
            setAnswerBySessionId((current) => {
              const next = { ...current }
              if (latestAssistantPayload) {
                next[activeSessionId] = latestAssistantPayload
              } else {
                delete next[activeSessionId]
              }
              return next
            })
          })
        }
      } catch (error) {
        if (active) {
          message.error(extractApiErrorMessage(error, '消息记录加载失败。'))
        }
      } finally {
        if (active) {
          setMessagesLoading(false)
        }
      }
    }

    void loadMessages()

    return () => {
      active = false
    }
  }, [activeSessionId, startTransition])

  const handleCreateSession = async () => {
    if (!selectedDataset) {
      message.warning('请先选择可聊天的数据集。')
      return
    }

    setAsking(true)
    try {
      const session = await createChatSession({
        dataset_id: selectedDataset.id,
        title: `${selectedDataset.name} · 节能问答`,
      })
      await loadSessions(selectedDataset.id)
      startTransition(() => {
        setActiveSessionId(session.id)
        setMessages([])
      })
      message.success('已创建新会话。')
    } catch (error) {
      message.error(extractApiErrorMessage(error, '创建会话失败，请稍后重试。'))
    } finally {
      setAsking(false)
    }
  }

  const handleAsk = async () => {
    if (!selectedDataset) {
      message.warning('请先选择数据集。')
      return
    }

    if (!question.trim()) {
      message.warning('请输入问题后再发送。')
      return
    }

    let sessionId = activeSessionId
    if (!sessionId) {
      try {
        const session = await createChatSession({
          dataset_id: selectedDataset.id,
          title: `${selectedDataset.name} · 节能问答`,
        })
        sessionId = session.id
        setActiveSessionId(session.id)
        await loadSessions(selectedDataset.id)
      } catch (error) {
        message.error(extractApiErrorMessage(error, '无法创建聊天会话。'))
        return
      }
    }

    setAsking(true)
    try {
      const result = await askAssistant({
        dataset_id: selectedDataset.id,
        session_id: sessionId,
        question: question.trim(),
        history: messages.map((item) => ({
          role: item.role,
          content: item.content,
        })),
      })
      const nextMessages = await fetchChatMessages(sessionId)
      startTransition(() => {
        setAnswerBySessionId((current) => ({
          ...current,
          [sessionId]: result,
        }))
        setQuestion('')
        setMessages(nextMessages)
      })
      await loadSessions(selectedDataset.id)
    } catch (error) {
      message.error(extractApiErrorMessage(error, '问答请求失败，请稍后重试。'))
    } finally {
      setAsking(false)
    }
  }

  if (loading) {
    return (
      <div className="page-state">
        <Spin size="large" />
      </div>
    )
  }

  return (
    <div className="page-stack">
      <PageHero
        eyebrow="智能体问答"
        title="围绕真实用电证据与会话上下文协同决策"
        description="当前问答页不再只是对话窗口，而是直接承接后端 agent 的结构化输出。你可以持续追问、查看依据、识别信息缺口，并将动作建议转成下一步操作。"
        icon={<MessageOutlined />}
        extra={
          <div className="hero-side-card agent-hero-card">
            <div className="agent-hero-card__row">
              <Typography.Text className="agent-hero-card__label">
                当前数据集
              </Typography.Text>
              <Typography.Text className="agent-hero-card__value">
                {selectedDataset?.name ?? '未选择'}
              </Typography.Text>
            </div>
            <div className="agent-hero-card__row">
              <Typography.Text className="agent-hero-card__label">
                会话数量
              </Typography.Text>
              <Typography.Text className="agent-hero-card__value">
                {sessions.length}
              </Typography.Text>
            </div>
            <div className="agent-hero-card__row">
              <Typography.Text className="agent-hero-card__label">
                本轮意图
              </Typography.Text>
              <Typography.Text className="agent-hero-card__value">
                {activeAnswer?.intent
                  ? assistantIntentMap[activeAnswer.intent] ?? activeAnswer.intent
                  : '等待提问'}
              </Typography.Text>
            </div>
            <div className="agent-hero-card__row">
              <Typography.Text className="agent-hero-card__label">
                回答状态
              </Typography.Text>
              <Typography.Text className="agent-hero-card__value">
                {activeAnswer?.degraded ? '降级' : activeAnswer ? '正常' : '空闲'}
              </Typography.Text>
            </div>
            <div className="agent-hero-card__note">
              <ThunderboltOutlined />
              <span>后端会自动注入统计摘要、分类结果、预测摘要与规则建议。</span>
            </div>
          </div>
        }
      >
        <Space wrap size={[12, 12]}>
          <Select<number>
            value={selectedDatasetId ?? undefined}
            style={{ width: 280 }}
            placeholder="选择数据集"
            options={readyDatasets.map((item) => ({
              label: item.name,
              value: item.id,
            }))}
            onChange={(value) => {
              startTransition(() => {
                setSelectedDatasetId(value)
                setActiveSessionId(null)
                setMessages([])
                setQuestion('')
              })
            }}
          />
          <Button icon={<PlusOutlined />} onClick={() => void handleCreateSession()}>
            新建会话
          </Button>
        </Space>
      </PageHero>

      {!readyDatasets.length ? (
        <Alert
          type="info"
          showIcon
          message="当前没有可聊天的数据集"
          description="请先导入并等待数据集处理完成，再进入智能问答。"
        />
      ) : null}

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={8}>
          <SectionCard
            title="会话轨道"
            subtitle={selectedDataset ? `当前数据集：${selectedDataset.name}` : '请先选择数据集'}
          >
            <div className="assistant-rail">
              <div className="assistant-rail__meta">
                <div className="assistant-rail__chip">
                  <HistoryOutlined />
                  <span>{sessions.length} 条会话</span>
                </div>
                <div className="assistant-rail__chip">
                  <AimOutlined />
                  <span>{messages.length} 条消息</span>
                </div>
              </div>

              <List
                loading={sessionsLoading}
                className="assistant-session-list"
                dataSource={sessions}
                locale={{
                  emptyText: (
                    <Empty
                      image={Empty.PRESENTED_IMAGE_SIMPLE}
                      description="当前数据集还没有会话"
                    />
                  ),
                }}
                renderItem={(item) => (
                  <List.Item
                    className={
                      item.id === activeSessionId
                        ? 'assistant-session assistant-session--active'
                        : 'assistant-session'
                    }
                    onClick={() => {
                      startTransition(() => {
                        setActiveSessionId(item.id)
                      })
                    }}
                  >
                    <Space direction="vertical" size={6} style={{ width: '100%' }}>
                      <Typography.Text strong>{item.title}</Typography.Text>
                      <Typography.Text type="secondary">
                        {formatDateTime(item.updated_at)}
                      </Typography.Text>
                    </Space>
                  </List.Item>
                )}
              />

              <div className="assistant-rail__note">
                <Typography.Text strong>工作流提示</Typography.Text>
                <Typography.Paragraph>
                  适合直接问“今天是什么类型”“明天有什么风险”“我该优先改什么”，
                  agent 会自动结合后端上下文做结构化回答。
                </Typography.Paragraph>
              </div>
            </div>
          </SectionCard>
        </Col>

        <Col xs={24} xl={16}>
          <SectionCard
            title="Agent 控制台"
            subtitle="消息历史、结构化回答、信息缺口和建议动作统一在这里完成。"
          >
            <div className="assistant-console">
              <div className="assistant-console__messages">
                {messagesLoading ? (
                  <div className="page-state page-state--compact">
                    <Spin />
                  </div>
                ) : messages.length === 0 ? (
                  <Empty
                    image={Empty.PRESENTED_IMAGE_SIMPLE}
                    description="当前会话还没有消息"
                  />
                ) : (
                  <List
                    dataSource={messages}
                    renderItem={(item) => (
                      <List.Item>
                        <div
                          className={
                            item.role === 'assistant'
                              ? 'assistant-bubble assistant-bubble--assistant'
                              : 'assistant-bubble assistant-bubble--user'
                          }
                        >
                          <Space direction="vertical" size={8} style={{ width: '100%' }}>
                            <Space wrap>
                              <Tag className={item.role === 'assistant' ? 'tone-tag tone-tag--accent' : 'tone-tag'}>
                                {item.role === 'assistant' ? '助手' : '用户'}
                              </Tag>
                            <Typography.Text type="secondary">
                              {formatDateTime(item.created_at)}
                            </Typography.Text>
                          </Space>
                          <Typography.Paragraph style={{ marginBottom: 0 }}>
                            {item.content}
                          </Typography.Paragraph>
                          {item.role === 'assistant' && isAssistantAnswerPayload(item.assistant_payload) ? (
                            <AssistantMessageHistoryPanel answer={item.assistant_payload} />
                          ) : null}
                        </Space>
                      </div>
                    </List.Item>
                    )}
                  />
                )}
              </div>

              <div className="assistant-composer">
                <Typography.Text className="assistant-composer__label">
                  提问输入
                </Typography.Text>
                <TextArea
                  rows={4}
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  onKeyDown={(event) => {
                    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
                      event.preventDefault()
                      void handleAsk()
                    }
                  }}
                  placeholder="例如：为什么我家夜间负荷偏高？明天的高峰主要会出现在什么时候？我应该优先做哪一步？"
                />
                <div className="assistant-composer__footer">
                  <Typography.Text type="secondary">
                    支持连续追问。按 `Ctrl/Cmd + Enter` 可直接发送。
                  </Typography.Text>
                  <Space wrap>
                    <Button
                      type="primary"
                      icon={<SendOutlined />}
                      loading={asking}
                      onClick={() => void handleAsk()}
                    >
                      发送问题
                    </Button>
                    <Button
                      onClick={() => {
                        setQuestion('')
                      }}
                    >
                      清空输入
                    </Button>
                  </Space>
                </div>
              </div>

              <AssistantResponsePanel answer={activeAnswer} />
            </div>
          </SectionCard>
        </Col>
      </Row>
    </div>
  )
}
