import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import { Empty } from 'antd'

type TrendPoint = {
  label: string
  value: number
}

type TrendChartProps = {
  data: TrendPoint[]
  lineColor?: string
  unit?: string
}

export function TrendChart({
  data,
  lineColor = '#5d6d5e',
  unit = 'kWh',
}: TrendChartProps) {
  if (data.length === 0) {
    return <Empty description="暂无曲线数据" />
  }

  const option: EChartsOption = {
    tooltip: {
      trigger: 'axis',
      valueFormatter: (value) => `${Number(value).toFixed(2)} ${unit}`,
    },
    grid: {
      left: 24,
      right: 20,
      top: 28,
      bottom: 28,
      containLabel: true,
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: data.map((item) => item.label),
      axisLabel: {
        color: '#6c7468',
      },
      axisLine: {
        lineStyle: {
          color: 'rgba(66, 78, 67, 0.18)',
        },
      },
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        color: '#6c7468',
      },
      axisLine: {
        show: false,
      },
      splitLine: {
        lineStyle: {
          color: 'rgba(66, 78, 67, 0.1)',
        },
      },
    },
    series: [
      {
        type: 'line',
        smooth: true,
        symbol: 'circle',
        symbolSize: 6,
        lineStyle: {
          color: lineColor,
          width: 3,
        },
        itemStyle: {
          color: lineColor,
        },
        areaStyle: {
          color: `${lineColor}18`,
        },
        data: data.map((item) => item.value),
      },
    ],
  }

  return <ReactECharts option={option} style={{ height: 300 }} />
}
