import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  MessageOutlined,
  PlusOutlined,
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
  askAssistant,
  createChatSession,
  fetchChatMessages,
  fetchChatSessions,
  fetchDatasets,
} from '@/services/dashboard'
import type { AssistantAnswer, ChatMessage, ChatSession, DatasetSummary } from '@/types/domain'
import { formatDateTime } from '@/utils/formatters'

const { TextArea } = Input

export function ChatPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [datasets, setDatasets] = useState<DatasetSummary[]>([])
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [answer, setAnswer] = useState<AssistantAnswer | null>(null)
  const [loading, setLoading] = useState(true)
  const [sessionsLoading, setSessionsLoading] = useState(false)
  const [messagesLoading, setMessagesLoading] = useState(false)
  const [asking, setAsking] = useState(false)
  const [question, setQuestion] = useState('')
  const [selectedDatasetId, setSelectedDatasetId] = useState<number | null>(null)
  const [activeSessionId, setActiveSessionId] = useState<number | null>(null)

  const readyDatasets = useMemo(
    () => datasets.filter((item) => item.status === 'ready'),
    [datasets],
  )

  const selectedDataset = useMemo(
    () => readyDatasets.find((item) => item.id === selectedDatasetId) ?? null,
    [readyDatasets, selectedDatasetId],
  )

  const loadSessions = useCallback(async (datasetId: number) => {
    setSessionsLoading(true)
    try {
      const result = await fetchChatSessions(datasetId)
      setSessions(result)
      setActiveSessionId((current) => {
        if (current && result.some((item) => item.id === current)) {
          return current
        }
        return result[0]?.id ?? null
      })
    } catch {
      message.error('会话列表加载失败。')
    } finally {
      setSessionsLoading(false)
    }
  }, [])

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
        setDatasets(datasetList)

        const queryDatasetId = Number(searchParams.get('dataset'))
        const initialDatasetId =
          availableDatasets.find((item) => item.id === queryDatasetId)?.id ??
          availableDatasets[0]?.id ??
          null

        setSelectedDatasetId(initialDatasetId)
      } catch {
        if (active) {
          message.error('聊天页初始化失败。')
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
  }, [searchParams])

  useEffect(() => {
    if (!selectedDatasetId) {
      setSessions([])
      setActiveSessionId(null)
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
          setMessages(nextMessages)
        }
      } catch {
        if (active) {
          message.error('消息记录加载失败。')
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
  }, [activeSessionId])

  const handleCreateSession = async () => {
    if (!selectedDataset) {
      message.warning('请先选择可聊天的数据集。')
      return
    }

    setAsking(true)
    try {
      const session = await createChatSession({
        dataset_id: selectedDataset.id,
        title: `${selectedDataset.name} 节能问答`,
      })
      await loadSessions(selectedDataset.id)
      setActiveSessionId(session.id)
      setMessages([])
      setAnswer(null)
      message.success('已创建新会话。')
    } catch {
      message.error('创建会话失败，请稍后重试。')
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
          title: `${selectedDataset.name} 节能问答`,
        })
        sessionId = session.id
        setActiveSessionId(session.id)
        await loadSessions(selectedDataset.id)
      } catch {
        message.error('无法创建聊天会话。')
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
      setAnswer(result)
      setQuestion('')
      setMessages(await fetchChatMessages(sessionId))
      await loadSessions(selectedDataset.id)
    } catch {
      message.error('问答请求失败，请稍后重试。')
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
        eyebrow="智能问答"
        title="和节能助手直接对话"
        description="选择已处理完成的数据集后，可以连续查看历史会话、提问并接收带依据的回答与动作建议。"
        icon={<MessageOutlined />}
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
              setSelectedDatasetId(value)
              setAnswer(null)
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
        <Col xs={24} xl={7}>
          <SectionCard
            title="会话列表"
            subtitle={selectedDataset ? `当前数据集：${selectedDataset.name}` : '请先选择数据集'}
          >
            <List
              loading={sessionsLoading}
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
                  onClick={() => setActiveSessionId(item.id)}
                >
                  <Space direction="vertical" size={4} style={{ width: '100%' }}>
                    <Typography.Text strong>{item.title}</Typography.Text>
                    <Typography.Text type="secondary">
                      {formatDateTime(item.updated_at)}
                    </Typography.Text>
                  </Space>
                </List.Item>
              )}
            />
          </SectionCard>
        </Col>

        <Col xs={24} xl={17}>
          <SectionCard title="对话窗口" subtitle="消息历史、回答依据和建议动作会在这里统一展示。">
            <div className="chat-workspace">
              <div className="assistant-messages">
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
                          <Space direction="vertical" size={6} style={{ width: '100%' }}>
                            <Space>
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
                          </Space>
                        </div>
                      </List.Item>
                    )}
                  />
                )}
              </div>

              <TextArea
                rows={4}
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="例如：为什么我家夜间负荷偏高？未来一天还有哪些高负荷风险？"
              />

              <Space wrap>
                <Button
                  type="primary"
                  icon={<MessageOutlined />}
                  loading={asking}
                  onClick={() => void handleAsk()}
                >
                  发送问题
                </Button>
                <Button
                  onClick={() => {
                    setQuestion('')
                    setAnswer(null)
                  }}
                >
                  清空输入
                </Button>
              </Space>

              {answer ? (
                <div className="assistant-answer">
                  {answer.degraded ? (
                    <Alert
                      type="warning"
                      showIcon
                      message="当前回答来自降级模式"
                      description={answer.error_reason ?? '未返回具体原因'}
                    />
                  ) : null}

                  <Typography.Title level={5}>本次回答</Typography.Title>
                  <Typography.Paragraph>{answer.answer}</Typography.Paragraph>

                  <Typography.Title level={5}>引用依据</Typography.Title>
                  <Space wrap>
                    {answer.citations.map((citation) => (
                      <Tag key={citation.key} className="tone-tag tone-tag--accent">
                        {citation.label}：
                        {Array.isArray(citation.value)
                          ? citation.value.join(' / ')
                          : String(citation.value)}
                      </Tag>
                    ))}
                  </Space>

                  <Typography.Title level={5} style={{ marginTop: 16 }}>
                    建议动作
                  </Typography.Title>
                  <List
                    size="small"
                    bordered
                    dataSource={answer.actions}
                    locale={{ emptyText: '当前没有返回动作建议' }}
                    renderItem={(item) => <List.Item>{item}</List.Item>}
                  />
                </div>
              ) : null}
            </div>
          </SectionCard>
        </Col>
      </Row>
    </div>
  )
}
