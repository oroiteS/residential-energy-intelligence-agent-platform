import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import { Empty } from 'antd'
import type { PeakPieItem } from '@/types/domain'
import { formatPercent } from '@/utils/formatters'

type PeakRatioChartProps = {
  data: PeakPieItem[]
}

export function PeakRatioChart({ data }: PeakRatioChartProps) {
  if (data.length === 0) {
    return <Empty description="暂无峰谷占比数据" />
  }

  const option: EChartsOption = {
    tooltip: {
      trigger: 'item',
      valueFormatter: (value) => `${formatPercent(Number(value))}`,
    },
    color: ['#9b876d', '#5d6d5e', '#c7b7a0'],
    series: [
      {
        type: 'pie',
        radius: ['44%', '72%'],
        center: ['50%', '52%'],
        label: {
          formatter: '{b}\n{d}%',
          fontSize: 12,
          lineHeight: 18,
          color: '#3f4b42',
        },
        itemStyle: {
          borderColor: '#f7f3eb',
          borderWidth: 3,
        },
        data: data.map((item) => ({
          name: item.name,
          value: item.ratio,
        })),
      },
    ],
  }

  return <ReactECharts option={option} style={{ height: 280 }} />
}
