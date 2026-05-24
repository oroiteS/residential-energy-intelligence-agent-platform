import { RobotOutlined } from '@ant-design/icons'
import { Button, Col, Empty, Row, Space, Table, Tag, Tooltip, Typography } from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { SectionCard } from '@/components/common/SectionCard'
import { classificationLabelMap } from '@/constants/display'
import type { ClassificationResult } from '@/types/domain'
import { formatPercent } from '@/utils/formatters'
import {
  buildClassificationTableItems,
  formatClassificationDay,
  type ClassificationTableItem,
} from '../model/classificationViewModel'

type ClassificationTabProps = {
  classification: ClassificationResult | null
  classificationHistory: ClassificationResult[]
  classificationActionLoading: boolean
  onRunClassification: () => void
  onOpenChat: () => void
}

function renderClassificationConfidenceTooltip(item: ClassificationResult) {
  return (
    <div className="classification-confidence-tooltip">
      {Object.entries(classificationLabelMap).map(([label, displayName]) => {
        const probability = item.probabilities[label as keyof typeof classificationLabelMap] ?? 0
        return (
          <div key={label} className="classification-confidence-tooltip__row">
            <span>{displayName}</span>
            <strong>{formatPercent(probability)}</strong>
          </div>
        )
      })}
    </div>
  )
}

const classificationColumns: ColumnsType<ClassificationTableItem> = [
  {
    title: '分类窗口',
    dataIndex: 'dayLabel',
    key: 'dayLabel',
    width: 190,
    render: (value: string) => <Typography.Text strong>{value}</Typography.Text>,
  },
  {
    title: '行为类别',
    dataIndex: 'predicted_label',
    key: 'predicted_label',
    width: 150,
    render: (_value, item) => (
      <Tag className="tone-tag tone-tag--accent">
        {item.label_display_name ?? classificationLabelMap[item.predicted_label]}
      </Tag>
    ),
  },
  {
    title: '置信度',
    dataIndex: 'confidence',
    key: 'confidence',
    width: 110,
    render: (value: number, item) => (
      <Tooltip placement="topRight" title={renderClassificationConfidenceTooltip(item)}>
        <Typography.Text className="classification-confidence-value">
          {formatPercent(value)}
        </Typography.Text>
      </Tooltip>
    ),
  },
  {
    title: '分类摘要',
    dataIndex: 'explanation',
    key: 'explanation',
    render: (value: string | undefined) => (
      <Typography.Text type="secondary">{value || '暂无分类说明'}</Typography.Text>
    ),
  },
]

export function ClassificationTab({
  classification,
  classificationHistory,
  classificationActionLoading,
  onRunClassification,
  onOpenChat,
}: ClassificationTabProps) {
  const classificationItems = buildClassificationTableItems(classificationHistory)

  return (
    <div className="page-stack">
      <Row gutter={[16, 16]}>
        <Col xs={24} xl={16}>
          <SectionCard
            title="行为分类结果"
            subtitle="按自然周识别用电行为类型，便于观察近期模式是否稳定。"
            extra={
              <Button loading={classificationActionLoading} onClick={onRunClassification}>
                重新生成分类
              </Button>
            }
          >
            {classification ? (
              <Space direction="vertical" size={18} style={{ width: '100%' }}>
                <div className="classification-current">
                  <div>
                    <Tag className="tone-tag tone-tag--accent">
                      {classification.label_display_name ??
                        classificationLabelMap[classification.predicted_label]}
                    </Tag>
                    <Typography.Title level={4} className="classification-current__title">
                      最新分类窗口：{formatClassificationDay(classification)}
                    </Typography.Title>
                    <Typography.Paragraph className="classification-current__summary">
                      {classification.explanation ||
                        `模型判定为${
                          classification.label_display_name ??
                          classificationLabelMap[classification.predicted_label]
                        }。`}
                    </Typography.Paragraph>
                  </div>
                  <Tooltip
                    placement="topRight"
                    title={renderClassificationConfidenceTooltip(classification)}
                  >
                    <div className="classification-current__confidence">
                      <span>最高置信度</span>
                      <strong>{formatPercent(classification.confidence)}</strong>
                    </div>
                  </Tooltip>
                </div>
                <Table
                  className="classification-history-table"
                  columns={classificationColumns}
                  dataSource={classificationItems}
                  pagination={false}
                  rowKey="id"
                  scroll={{ x: 760 }}
                  size="middle"
                />
              </Space>
            ) : (
              <Empty description="暂无分类结果" />
            )}
          </SectionCard>
        </Col>
        <Col xs={24} xl={8}>
          <SectionCard
            title="智能体建议"
            subtitle="基于分类、预测与异常检测生成建议。"
            className="agent-advice-card"
            extra={
              <Button type="primary" icon={<RobotOutlined />} onClick={onOpenChat}>
                进入智能问答
              </Button>
            }
          >
            <Space direction="vertical" size={14} className="agent-advice-entry">
              <Space wrap>
                <Tag className="tone-tag tone-tag--accent">分类上下文</Tag>
                <Tag className="tone-tag tone-tag--warm">未来预测</Tag>
                <Tag className="tone-tag tone-tag--muted">异常复核</Tag>
              </Space>
            </Space>
          </SectionCard>
        </Col>
      </Row>
    </div>
  )
}
