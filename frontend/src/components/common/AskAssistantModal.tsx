import { useEffect, useState } from 'react'
import { MessageOutlined, PlusOutlined, RobotOutlined } from '@ant-design/icons'
import {
  Alert,
  Button,
  Empty,
  Input,
  List,
  Modal,
  Space,
  Tag,
  Typography,
  message,
} from 'antd'
import {
  askAssistant,
  createChatSession,
  fetchChatMessages,
  fetchChatSessions,
} from '@/services/dashboard'
import type { AssistantAnswer, ChatMessage, ChatSession } from '@/types/domain'
import { formatDateTime } from '@/utils/formatters'

const { TextArea } = Input

type AskAssistantModalProps = {
  datasetId: number
  datasetName: string
  disabled?: boolean
}

export function AskAssistantModal({
  datasetId,
  datasetName,
  disabled = false,
}: AskAssistantModalProps) {
  const [open, setOpen] = useState(false)
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [sessionsLoading, setSessionsLoading] = useState(false)
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [activeSessionId, setActiveSessionId] = useState<number | null>(null)
  const [answer, setAnswer] = useState<AssistantAnswer | null>(null)

  useEffect(() => {
    if (!open) {
      return
    }

    let active = true

    const loadSessions = async () => {
      setSessionsLoading(true)
      try {
        const sessionList = await fetchChatSessions(datasetId)
        if (!active) {
          return
        }

        setSessions(sessionList)
        setActiveSessionId((current) => current ?? sessionList[0]?.id ?? null)
      } catch {
        if (active) {
          message.error('会话列表加载失败。')
        }
      } finally {
        if (active) {
          setSessionsLoading(false)
        }
      }
    }

    void loadSessions()

    return () => {
      active = false
    }
  }, [datasetId, open])

  useEffect(() => {
    if (!open || !activeSessionId) {
      setMessages([])
      return
    }

    let active = true

    const loadMessages = async () => {
      try {
        const nextMessages = await fetchChatMessages(activeSessionId)
        if (active) {
          setMessages(nextMessages)
        }
      } catch {
        if (active) {
          message.error('会话消息加载失败。')
        }
      }
    }

    void loadMessages()

    return () => {
      active = false
    }
  }, [activeSessionId, open])

  const handleCreateSession = async () => {
    setLoading(true)
    try {
      const session = await createChatSession({
        dataset_id: datasetId,
        title: `${datasetName} 节能问答`,
      })
      const sessionList = await fetchChatSessions(datasetId)
      setSessions(sessionList)
      setActiveSessionId(session.id)
      setMessages([])
      message.success('已创建新会话。')
    } catch {
      message.error('创建会话失败，请稍后重试。')
    } finally {
      setLoading(false)
    }
  }

  const handleAsk = async () => {
    if (!question.trim()) {
      message.warning('请输入问题后再发送。')
      return
    }

    let sessionId = activeSessionId

    if (!sessionId) {
      try {
        const session = await createChatSession({
          dataset_id: datasetId,
          title: `${datasetName} 节能问答`,
        })
        sessionId = session.id
        setActiveSessionId(session.id)
        setSessions(await fetchChatSessions(datasetId))
      } catch {
        message.error('无法创建问答会话。')
        return
      }
    }

    setLoading(true)
    try {
      const result = await askAssistant({
        dataset_id: datasetId,
        session_id: sessionId,
        question: question.trim(),
        history: messages.map((item) => ({
          role: item.role,
          content: item.content,
        })),
      })
      setAnswer(result)
      setMessages(await fetchChatMessages(sessionId))
      setSessions(await fetchChatSessions(datasetId))
      setQuestion('')
    } catch {
      message.error('智能问答请求失败，请稍后重试。')
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <Button
        type="primary"
        icon={<RobotOutlined />}
        disabled={disabled}
        onClick={() => setOpen(true)}
      >
        智能问答
      </Button>
      <Modal
        title={`节能问答 · ${datasetName}`}
        open={open}
        onCancel={() => setOpen(false)}
        footer={null}
        width={860}
        destroyOnClose
      >
        <Space direction="vertical" size={16} style={{ width: '100%' }}>
          <Typography.Paragraph type="secondary" style={{ marginBottom: 0 }}>
            选择会话后即可继续追问，回答会保留上下文并同步展示依据与建议。
          </Typography.Paragraph>

          <div className="assistant-layout">
            <div className="assistant-layout__sidebar">
              <div className="assistant-layout__sidebar-header">
                <Typography.Text strong>聊天会话</Typography.Text>
                <Button
                  size="small"
                  icon={<PlusOutlined />}
                  loading={loading}
                  onClick={() => void handleCreateSession()}
                >
                  新建
                </Button>
              </div>

              <List
                loading={sessionsLoading}
                dataSource={sessions}
                locale={{ emptyText: <Empty description="暂无会话" image={Empty.PRESENTED_IMAGE_SIMPLE} /> }}
                renderItem={(item) => (
                  <List.Item
                    className={item.id === activeSessionId ? 'assistant-session assistant-session--active' : 'assistant-session'}
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
            </div>

            <div className="assistant-layout__main">
              <div className="assistant-messages">
                {messages.length === 0 ? (
                  <Empty description="当前会话还没有消息" image={Empty.PRESENTED_IMAGE_SIMPLE} />
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
                              <Tag color={item.role === 'assistant' ? 'blue' : 'default'}>
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
                placeholder="例如：为什么我家夜间负荷偏高？未来一天还有哪些高负荷风险？…"
              />

              <Space wrap>
                <Button
                  type="primary"
                  icon={<MessageOutlined />}
                  loading={loading}
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
            </div>
          </div>

          {answer ? (
            <div className="assistant-answer">
              {answer.degraded ? (
                <Alert
                  type="warning"
                  showIcon
                  message="当前回答为简化结果"
                  description="部分依据暂不可用，建议结合图表与报告交叉查看。"
                />
              ) : null}

              <Typography.Title level={5}>本次回答摘要</Typography.Title>
              <Typography.Paragraph>{answer.answer}</Typography.Paragraph>

              <Typography.Title level={5}>引用指标</Typography.Title>
              <Space wrap>
                {answer.citations.map((citation) => (
                  <Tag key={citation.key} color="blue">
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
                renderItem={(item) => <List.Item>{item}</List.Item>}
              />
            </div>
          ) : null}
        </Space>
      </Modal>
    </>
  )
}
