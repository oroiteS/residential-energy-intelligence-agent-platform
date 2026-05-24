import { Button, Descriptions, Empty, List, Space, Tag, Tooltip, Typography } from 'antd'
import { SectionCard } from '@/components/common/SectionCard'
import {
  detectionFeatureDescriptionMap,
  detectionFeatureLabelMap,
  detectionMethodLabelMap,
} from '@/constants/display'
import type { DetectionResult } from '@/types/domain'
import { formatDateTime, formatNumber, formatPercent } from '@/utils/formatters'

type DetectionTabProps = {
  detection: DetectionResult | null
  detectionActionLoading: boolean
  onRunDetection: () => void
}

export function DetectionTab({
  detection,
  detectionActionLoading,
  onRunDetection,
}: DetectionTabProps) {
  return (
    <SectionCard
      title="异常检测结果"
      subtitle="识别最近 7 天窗口是否明显偏离历史用电规律。"
      extra={
        <Button loading={detectionActionLoading} onClick={onRunDetection}>
          重新检测异常
        </Button>
      }
    >
      {detection ? (
        <Space direction="vertical" size={18} style={{ width: '100%' }}>
          <Space wrap>
            <Tag color={detection.is_anomaly ? 'error' : 'success'}>
              {detection.is_anomaly ? '检测到异常' : '未检测到异常'}
            </Tag>
            <Typography.Text type="secondary">
              异常分数 {formatNumber(detection.anomaly_score, 4)}
            </Typography.Text>
          </Space>
          <Descriptions column={{ xs: 1, lg: 2 }} size="small">
            <Descriptions.Item label="检测窗口开始">
              {formatDateTime(detection.window_start)}
            </Descriptions.Item>
            <Descriptions.Item label="检测窗口结束">
              {formatDateTime(detection.window_end)}
            </Descriptions.Item>
            <Descriptions.Item label="辅助分类">
              {detection.classification_hint ?? '--'}
            </Descriptions.Item>
            <Descriptions.Item label="结果时间">
              {formatDateTime(detection.created_at)}
            </Descriptions.Item>
          </Descriptions>
          <List
            size="small"
            header={<Typography.Text strong>异常原因</Typography.Text>}
            dataSource={detection.reasons}
            locale={{ emptyText: '当前窗口未触发异常规则。' }}
            renderItem={(item, index) => (
              <List.Item key={`${item.rule}-${index}`}>
                <Space direction="vertical" size={4} style={{ width: '100%' }}>
                  <Space wrap>
                    <Tooltip
                      placement="topLeft"
                      title={
                        detectionFeatureDescriptionMap[item.feature] ??
                        '该指标用于辅助解释当前窗口的用电行为特征。'
                      }
                    >
                      <Tag className="feature-reason-tag">
                        {detectionFeatureLabelMap[item.feature] ?? item.feature}
                      </Tag>
                    </Tooltip>
                    <Tag color="gold">
                      {detectionMethodLabelMap[item.method] ?? item.method}
                    </Tag>
                    <Typography.Text type="secondary">
                      触发强度 {formatPercent(item.severity)}
                    </Typography.Text>
                  </Space>
                  <Typography.Text>{item.reason}</Typography.Text>
                </Space>
              </List.Item>
            )}
          />
          <Descriptions column={{ xs: 1, lg: 3 }} size="small" title="关键特征摘要">
            {Object.entries(detection.feature_summary).map(([key, value]) => (
              <Descriptions.Item
                key={key}
                label={
                  <Tooltip
                    placement="topLeft"
                    title={
                      detectionFeatureDescriptionMap[key] ??
                      '该指标用于辅助解释当前窗口的用电行为特征。'
                    }
                  >
                    <span className="feature-label-chip">
                      {detectionFeatureLabelMap[key] ?? key}
                    </span>
                  </Tooltip>
                }
              >
                {formatNumber(value, 4)}
              </Descriptions.Item>
            ))}
          </Descriptions>
        </Space>
      ) : (
        <Empty description="暂无异常检测结果" />
      )}
    </SectionCard>
  )
}
