import { RocketOutlined } from '@ant-design/icons'
import { Button, Col, Descriptions, Empty, List, Row, Select, Space, Spin, Tag, Tooltip, Typography } from 'antd'
import { TrendChart } from '@/components/charts/TrendChart'
import { SectionCard } from '@/components/common/SectionCard'
import { MetricCard } from '@/components/sections/MetricCard'
import {
  classificationLabelMap,
  detectionFeatureDescriptionMap,
  detectionFeatureLabelMap,
  detectionMethodLabelMap,
  forecastModelMap,
  riskFlagMap,
} from '@/constants/display'
import type { ForecastDetail, ForecastModelType, ForecastRecord } from '@/types/domain'
import { formatNumber, formatPercent } from '@/utils/formatters'
import { buildForecastSeriesChartData } from '../model/chartMappers'
import {
  getForecastRangeLabel,
  getForecastWindowLabel,
  getPredictedTotalKwh,
} from '../model/forecastViewModel'

type ForecastTabProps = {
  visibleForecasts: ForecastRecord[]
  selectedForecast: ForecastRecord | null
  activeForecastId: number | null
  forecastDetail: ForecastDetail | null
  forecastDetailLoading: boolean
  forecastActionLoading: boolean
  selectedForecastModel: ForecastModelType
  onForecastModelChange: (model: ForecastModelType) => void
  onActiveForecastChange: (forecastId: number) => void
  onGenerateForecast: () => void
}

export function ForecastTab({
  visibleForecasts,
  selectedForecast,
  activeForecastId,
  forecastDetail,
  forecastDetailLoading,
  forecastActionLoading,
  selectedForecastModel,
  onForecastModelChange,
  onActiveForecastChange,
  onGenerateForecast,
}: ForecastTabProps) {
  return (
    <div className="page-stack">
      <SectionCard
        title="预测操作台"
        subtitle="基于最近 30 天日级窗口，一次生成未来 7 天总量、峰时和谷时预测。"
        extra={
          <Space wrap>
            <Select
              value={selectedForecastModel}
              style={{ width: 180 }}
              options={[{ label: forecastModelMap.lstm, value: 'lstm' }]}
              onChange={onForecastModelChange}
            />
            <Button
              type="primary"
              icon={<RocketOutlined />}
              loading={forecastActionLoading}
              onClick={onGenerateForecast}
            >
              生成预测
            </Button>
          </Space>
        }
      >
        {selectedForecast ? (
          <Row gutter={[16, 16]}>
            <Col xs={24} md={12} xl={6}>
              <MetricCard
                label="日均预测"
                value={`${formatNumber(selectedForecast.summary.predicted_avg_daily_kwh)} kWh`}
                accent="amber"
              />
            </Col>
            <Col xs={24} md={12} xl={6}>
              <MetricCard
                label="峰时预测"
                value={`${formatNumber(selectedForecast.summary.predicted_peak_kwh)} kWh`}
                accent="teal"
              />
            </Col>
            <Col xs={24} md={12} xl={6}>
              <MetricCard
                label="预测总量"
                value={`${formatNumber(getPredictedTotalKwh(selectedForecast.summary))} kWh`}
                accent="coral"
              />
            </Col>
            <Col xs={24} md={12} xl={6}>
              <MetricCard
                label="预测类别"
                value={
                  selectedForecast.summary.forecast_classification
                    ? selectedForecast.summary.forecast_classification.label_display_name ??
                      classificationLabelMap[
                        selectedForecast.summary.forecast_classification.predicted_label
                      ]
                    : '--'
                }
                hint={
                  selectedForecast.summary.forecast_classification
                    ? `置信度 ${formatPercent(
                        selectedForecast.summary.forecast_classification.confidence,
                      )}`
                    : '待生成未来分类'
                }
                accent="olive"
              />
            </Col>
          </Row>
        ) : (
          <Empty description="暂无预测结果，请先生成预测" />
        )}
      </SectionCard>

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={9}>
          <SectionCard title="预测结果列表" subtitle="保留最近结果，便于切换查看。">
            <List
              dataSource={visibleForecasts}
              locale={{ emptyText: '暂无预测记录' }}
              renderItem={(item) => (
                <List.Item
                  className={
                    item.id === activeForecastId
                      ? 'forecast-list-item forecast-list-item--active'
                      : 'forecast-list-item'
                  }
                  onClick={() => {
                    onActiveForecastChange(item.id)
                  }}
                >
                  <Space direction="vertical" size={6} style={{ width: '100%' }}>
                    <Space wrap>
                      <Tag className="tone-tag tone-tag--accent">
                        {forecastModelMap[item.model_type]}
                      </Tag>
                      <Tag className="tone-tag tone-tag--warm">
                        {getForecastWindowLabel(item.summary)}
                      </Tag>
                      <Typography.Text strong>
                        {getForecastRangeLabel(item.forecast_start, item.forecast_end)}
                      </Typography.Text>
                    </Space>
                    <Typography.Text type="secondary">
                      {formatNumber(item.summary.predicted_avg_daily_kwh)} kWh 日均 ·{' '}
                      {formatNumber(getPredictedTotalKwh(item.summary))} kWh 预测总量
                    </Typography.Text>
                  </Space>
                </List.Item>
              )}
            />
          </SectionCard>
        </Col>
        <Col xs={24} xl={15}>
          <SectionCard title="预测曲线" subtitle="展示所选预测记录的时间序列。">
            {forecastDetailLoading ? (
              <div className="page-state page-state--compact">
                <Spin />
              </div>
            ) : forecastDetail ? (
              <TrendChart
                data={buildForecastSeriesChartData(forecastDetail)}
                lineColor="#9b876d"
                unit="kWh"
              />
            ) : (
              <Empty description="暂无预测曲线" />
            )}
          </SectionCard>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col xs={24}>
          <SectionCard title="预测摘要" subtitle="汇总关键时段、未来分类与风险提示。">
            {selectedForecast ? (
              <Descriptions column={1} size="small">
                <Descriptions.Item label="预测天数">
                  {getForecastWindowLabel(selectedForecast.summary)}
                </Descriptions.Item>
                <Descriptions.Item label="模型类型">
                  {forecastModelMap[selectedForecast.model_type]}
                </Descriptions.Item>
                <Descriptions.Item label="预测区间">
                  {getForecastRangeLabel(
                    selectedForecast.forecast_start,
                    selectedForecast.forecast_end,
                  )}
                </Descriptions.Item>
                <Descriptions.Item label="预测总用电量">
                  {formatNumber(getPredictedTotalKwh(selectedForecast.summary))} kWh
                </Descriptions.Item>
                <Descriptions.Item label="预测分类">
                  {selectedForecast.summary.forecast_classification ? (
                    <Space wrap>
                      <Tag className="tone-tag tone-tag--accent">
                        {selectedForecast.summary.forecast_classification.label_display_name ??
                          classificationLabelMap[
                            selectedForecast.summary.forecast_classification.predicted_label
                          ]}
                      </Tag>
                      <Typography.Text type="secondary">
                        XGBoost 基于预测日曲线判断，置信度{' '}
                        {formatPercent(
                          selectedForecast.summary.forecast_classification.confidence,
                        )}
                      </Typography.Text>
                    </Space>
                  ) : (
                    '--'
                  )}
                </Descriptions.Item>
                <Descriptions.Item label="复核提示">
                  {selectedForecast.summary.risk_flags.length ? (
                    <Space wrap>
                      {selectedForecast.summary.risk_flags.map((flag) => (
                        <Tag key={flag} className="tone-tag tone-tag--muted">
                          {riskFlagMap[flag] ?? flag}
                        </Tag>
                      ))}
                    </Space>
                  ) : (
                    '--'
                  )}
                </Descriptions.Item>
                <Descriptions.Item label="未来异常检测">
                  {selectedForecast.summary.future_detection ? (
                    <div className="future-detection-panel">
                      <div className="future-detection-panel__head">
                        <Tag
                          color={
                            selectedForecast.summary.future_detection.is_anomaly
                              ? 'warning'
                              : 'success'
                          }
                        >
                          {selectedForecast.summary.future_detection.is_anomaly
                            ? '建议复核'
                            : '未触发复核'}
                        </Tag>
                        <Typography.Text type="secondary">
                          偏离分数{' '}
                          {formatNumber(
                            selectedForecast.summary.future_detection.anomaly_score,
                            4,
                          )}
                        </Typography.Text>
                      </div>
                      {selectedForecast.summary.future_detection.reasons.length ? (
                        <List
                          className="future-detection-reasons"
                          size="small"
                          dataSource={selectedForecast.summary.future_detection.reasons}
                          renderItem={(item, index) => (
                            <List.Item key={`${item.rule}-${index}`}>
                              <Space direction="vertical" size={6} style={{ width: '100%' }}>
                                <Space wrap>
                                  <Tooltip
                                    placement="topLeft"
                                    title={
                                      detectionFeatureDescriptionMap[item.feature] ??
                                      '该指标用于辅助解释预测窗口的用电行为特征。'
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
                      ) : (
                        <Typography.Text type="secondary">
                          预测窗口没有触发额外复核规则。
                        </Typography.Text>
                      )}
                    </div>
                  ) : (
                    '--'
                  )}
                </Descriptions.Item>
              </Descriptions>
            ) : (
              <Empty description="暂无预测摘要" />
            )}
          </SectionCard>
        </Col>
      </Row>
    </div>
  )
}
