import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import { Empty } from 'antd'

type ComparePoint = {
  label: string
  actual: number
  predicted: number
}

type CompareTrendChartProps = {
  data: ComparePoint[]
  unit?: string
}

export function CompareTrendChart({
  data,
  unit = 'kWh',
}: CompareTrendChartProps) {
  if (data.length === 0) {
    return <Empty description="暂无对比曲线数据" />
  }

  const actualColor = '#5d6d5e'
  const predictedColor = '#9b876d'

  const option: EChartsOption = {
    color: [actualColor, predictedColor],
    tooltip: {
      trigger: 'axis',
      valueFormatter: (value) => `${Number(value).toFixed(2)} ${unit}`,
    },
    legend: {
      top: 0,
      textStyle: {
        color: '#6c7468',
      },
    },
    grid: {
      left: 24,
      right: 20,
      top: 48,
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
        name: '实际值',
        type: 'line',
        color: actualColor,
        smooth: true,
        symbol: 'none',
        lineStyle: {
          color: actualColor,
          width: 3,
        },
        itemStyle: {
          color: actualColor,
        },
        data: data.map((item) => item.actual),
      },
      {
        name: '预测值',
        type: 'line',
        color: predictedColor,
        smooth: true,
        symbol: 'none',
        lineStyle: {
          color: predictedColor,
          width: 3,
        },
        areaStyle: {
          color: `${predictedColor}14`,
        },
        itemStyle: {
          color: predictedColor,
        },
        data: data.map((item) => item.predicted),
      },
    ],
  }

  return <ReactECharts option={option} style={{ height: 320 }} />
}
