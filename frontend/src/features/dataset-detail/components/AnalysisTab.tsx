import { Col, Descriptions, Empty, Row, Space, Tag, Typography } from 'antd'
import { PeakRatioChart } from '@/components/charts/PeakRatioChart'
import { TrendChart } from '@/components/charts/TrendChart'
import { SectionCard } from '@/components/common/SectionCard'
import type { AnalysisPayload, DatasetDetailPayload, PeakValleyConfig } from '@/types/domain'
import { formatPercent } from '@/utils/formatters'
import {
  buildDailyTrendChartData,
  buildTypicalDayChartData,
  buildWeeklyTrendChartData,
} from '../model/chartMappers'

type AnalysisTabProps = {
  detail: DatasetDetailPayload
  analysis: AnalysisPayload
  peakValleyConfig: PeakValleyConfig
}

export function AnalysisTab({ detail, analysis, peakValleyConfig }: AnalysisTabProps) {
  const weeklyTrendChartData = buildWeeklyTrendChartData(analysis)

  return (
    <div className="page-stack">
      <SectionCard title="数据质量摘要" subtitle="帮助快速判断样本完整性与清洗情况">
        <Descriptions column={{ xs: 1, lg: 2 }} size="small">
          <Descriptions.Item label="缺失率">
            {detail.quality_summary ? formatPercent(detail.quality_summary.missing_rate) : '--'}
          </Descriptions.Item>
          <Descriptions.Item label="重复时间戳">
            {detail.quality_summary?.duplicate_count ?? '--'}
          </Descriptions.Item>
          <Descriptions.Item label="采样间隔">
            {detail.quality_summary?.sampling_interval ?? '--'}
          </Descriptions.Item>
          <Descriptions.Item label="清洗策略" span={2}>
            {detail.quality_summary?.cleaning_strategies?.length ? (
              <Space wrap>
                {detail.quality_summary.cleaning_strategies.map((item) => (
                  <Tag key={item}>{item}</Tag>
                ))}
              </Space>
            ) : (
              '--'
            )}
          </Descriptions.Item>
        </Descriptions>
      </SectionCard>

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={12}>
          <SectionCard title="按日趋势" subtitle="每日总用电量趋势，单位 kWh">
            <TrendChart
              data={buildDailyTrendChartData(analysis)}
              lineColor="#9b876d"
              unit="kWh"
            />
          </SectionCard>
        </Col>
        <Col xs={24} xl={12}>
          <SectionCard title="按周趋势" subtitle="每周总用电量趋势，单位 kWh">
            {weeklyTrendChartData.length >= 2 ? (
              <TrendChart data={weeklyTrendChartData} lineColor="#5d6d5e" unit="kWh" />
            ) : (
              <Empty description="样本不足两周，暂不展示按周趋势" />
            )}
          </SectionCard>
        </Col>
        <Col xs={24} xl={14}>
          <SectionCard title="典型日曲线" subtitle="跨样本日按小时平均负荷曲线，单位 W">
            <TrendChart
              data={buildTypicalDayChartData(analysis)}
              lineColor="#8c7b62"
              unit="W"
            />
            <Typography.Text type="secondary">
              该曲线是所有日期在同一小时负荷的平均值；按小时聚合后会比 15 分钟原始曲线更平滑。
            </Typography.Text>
          </SectionCard>
        </Col>
        <Col xs={24} xl={10}>
          <SectionCard title="峰谷占比" subtitle="从峰时和谷时结构理解用电分布">
            <PeakRatioChart data={analysis.charts.peak_valley_pie} />
            <Descriptions column={1} size="small">
              <Descriptions.Item label="峰时">
                {peakValleyConfig.peak.join(' / ')}
              </Descriptions.Item>
              <Descriptions.Item label="谷时">
                {peakValleyConfig.valley.join(' / ')}
              </Descriptions.Item>
            </Descriptions>
            <Typography.Text type="secondary">
              本项目仅区分峰时和谷时，所有上传数据最终都会归入这两类时段。
            </Typography.Text>
          </SectionCard>
        </Col>
      </Row>
    </div>
  )
}
