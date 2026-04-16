const stateElements = {
  statusText: document.getElementById('status-text'),
  virtualTime: document.getElementById('virtual-time'),
  weekLoop: document.getElementById('week-loop'),
  houseId: document.getElementById('house-id'),
  dataFile: document.getElementById('data-file'),
  todayProgress: document.getElementById('today-progress'),
  metricCurrent: document.getElementById('metric-current'),
  metricKwh: document.getElementById('metric-kwh'),
  metricPeak: document.getElementById('metric-peak'),
  metricBase: document.getElementById('metric-base'),
  classificationDate: document.getElementById('classification-date'),
  classificationLabel: document.getElementById('classification-label'),
  classificationModel: document.getElementById('classification-model'),
  classificationExplanation: document.getElementById('classification-explanation'),
  forecastDate: document.getElementById('forecast-date'),
  forecastModel: document.getElementById('forecast-model'),
  forecastAvg: document.getElementById('forecast-avg'),
  forecastPeak: document.getElementById('forecast-peak'),
  forecastExplanation: document.getElementById('forecast-explanation'),
  forecastRisks: document.getElementById('forecast-risks'),
  chatForm: document.getElementById('chat-form'),
  chatInput: document.getElementById('chat-input'),
  chatAnswer: document.getElementById('chat-answer'),
}

const chartContainer = document.getElementById('trend-chart')
const chart =
  typeof window !== 'undefined' &&
  typeof window.echarts !== 'undefined' &&
  chartContainer
    ? window.echarts.init(chartContainer)
    : null

const labelMap = {
  daytime_active: '白天活跃型',
  daytime_peak_strong: '白天尖峰明显型',
  flat_stable: '平稳基线型',
  night_dominant: '夜间主导型',
}

const modelMap = {
  xgboost: 'XGBoost',
  tft: 'Temporal Fusion Transformer',
}

const riskFlagMap = {
  evening_peak: '晚高峰风险',
  daytime_peak: '白天高峰风险',
  high_baseload: '基线负荷偏高',
  abnormal_rise: '异常抬升风险',
  peak_overlap_risk: '峰时叠加风险',
}

function formatNumber(value, digits = 2) {
  return Number(value || 0).toFixed(digits)
}

function formatModelLabel(value) {
  return modelMap[value] || value || '--'
}

function render(snapshot) {
  const latestClassification = snapshot.latestClassification || {}
  const nextDayForecast = snapshot.nextDayForecast || snapshot.activeForecast || {}
  const metrics = snapshot.metrics || {}
  const source = snapshot.source || {}

  stateElements.statusText.textContent = '实时流已连接'
  stateElements.virtualTime.textContent = snapshot.virtualTime
  stateElements.weekLoop.textContent = String(snapshot.weekLoop)
  stateElements.houseId.textContent = source.houseId || '--'
  stateElements.dataFile.textContent = source.dataFile || '--'
  stateElements.todayProgress.textContent = `${formatNumber(metrics.progressPercent)}%`
  stateElements.metricCurrent.textContent = `${formatNumber(metrics.currentPowerW)} W`
  stateElements.metricKwh.textContent = `${formatNumber(metrics.todayCumulativeKWH, 3)} kWh`
  stateElements.metricPeak.textContent = `${formatNumber(metrics.todayPeakLoadW)} W`
  stateElements.metricBase.textContent = `${formatNumber(metrics.todayBaseLoadW)} W`

  stateElements.classificationDate.textContent = latestClassification.date || '--'
  stateElements.classificationLabel.textContent =
    labelMap[latestClassification.label] || latestClassification.label || '--'
  stateElements.classificationModel.textContent =
    `模型 ${formatModelLabel(latestClassification.modelType)}`
  stateElements.classificationExplanation.textContent =
    latestClassification.error || latestClassification.explanation || '等待完整日分类结果。'

  stateElements.forecastDate.textContent = nextDayForecast.date || '--'
  stateElements.forecastModel.textContent =
    `模型 ${formatModelLabel(nextDayForecast.modelType)}`
  stateElements.forecastAvg.textContent = `${formatNumber(nextDayForecast.avgLoadW)} W`
  stateElements.forecastPeak.textContent = `${formatNumber(nextDayForecast.peakLoadW)} W`
  stateElements.forecastExplanation.textContent =
    nextDayForecast.error || nextDayForecast.explanation || '等待预测结果。'
  renderRiskFlags(nextDayForecast.riskFlags || [])

  renderChart(snapshot)
}

function renderRiskFlags(flags) {
  if (!Array.isArray(flags)) {
    stateElements.forecastRisks.innerHTML = '<span class="soft-tag">暂无风险提示</span>'
    return
  }
  if (flags.length === 0) {
    stateElements.forecastRisks.innerHTML = '<span class="soft-tag">暂无风险提示</span>'
    return
  }
  stateElements.forecastRisks.innerHTML = flags
    .map((flag) => `<span class="soft-tag">${riskFlagMap[flag] || flag}</span>`)
    .join('')
}

function renderChart(snapshot) {
  if (!chart) {
    return
  }
  const todayForecastPoints = Array.isArray(snapshot.todayForecast?.points)
    ? snapshot.todayForecast.points
    : []
  const nextDayForecastPoints = Array.isArray(snapshot.nextDayForecast?.points)
    ? snapshot.nextDayForecast.points
    : Array.isArray(snapshot.activeForecast?.points)
      ? snapshot.activeForecast.points
      : []
  const todayPoints = Array.isArray(snapshot.todayPoints) ? snapshot.todayPoints : []
  const labels =
    todayForecastPoints.length > 0
      ? todayForecastPoints.map((point) => point.timeLabel)
      : nextDayForecastPoints.map((point) => point.timeLabel)
  if (labels.length === 0) {
    chart.clear()
    return
  }
  const actualSeries = new Array(labels.length).fill(null)
  todayPoints.forEach((point) => {
    actualSeries[point.slotIndex] = point.aggregate
  })
  const todayForecastSeries = labels.map((_, index) => {
    const point = todayForecastPoints[index]
    return point ? point.predicted : null
  })
  const nextDayForecastSeries = labels.map((_, index) => {
    const point = nextDayForecastPoints[index]
    return point ? point.predicted : null
  })

  chart.setOption({
    animationDuration: 300,
    backgroundColor: 'transparent',
    color: ['#f1b875', '#7dd3c7', '#ef8a73'],
    tooltip: {
      trigger: 'axis',
      valueFormatter: (value) => `${formatNumber(value)} W`,
    },
    legend: {
      top: 8,
      textStyle: {
        color: '#d8d3c4',
        fontFamily: 'PingFang SC, Hiragino Sans GB, STSong, SimSun, Microsoft YaHei, sans-serif',
      },
      data: ['今日实测', '今日预测', '次日预测'],
    },
    grid: {
      left: 28,
      right: 20,
      top: 54,
      bottom: 30,
      containLabel: true,
    },
    xAxis: {
      type: 'category',
      data: labels,
      axisLabel: {
        color: '#b8b3a4',
        interval: 7,
      },
      axisLine: {
        lineStyle: {
          color: 'rgba(255,255,255,0.15)',
        },
      },
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        color: '#b8b3a4',
      },
      splitLine: {
        lineStyle: {
          color: 'rgba(255,255,255,0.08)',
        },
      },
    },
    series: [
      {
        name: '今日实测',
        type: 'line',
        smooth: true,
        showSymbol: false,
        data: actualSeries,
        itemStyle: {
          color: '#f1b875',
        },
        lineStyle: {
          width: 3,
          color: '#f1b875',
        },
        areaStyle: {
          color: 'rgba(241, 184, 117, 0.15)',
        },
      },
      {
        name: '今日预测',
        type: 'line',
        smooth: true,
        showSymbol: false,
        data: todayForecastSeries,
        itemStyle: {
          color: '#7dd3c7',
        },
        lineStyle: {
          width: 2,
          color: '#7dd3c7',
        },
      },
      {
        name: '次日预测',
        type: 'line',
        smooth: true,
        showSymbol: false,
        data: nextDayForecastSeries,
        itemStyle: {
          color: '#ef8a73',
        },
        lineStyle: {
          width: 2,
          type: 'dashed',
          color: '#ef8a73',
        },
      },
    ],
  })
}

async function loadInitialState() {
  const response = await fetch('/api/state')
  if (!response.ok) {
    throw new Error('初始化状态加载失败')
  }
  const snapshot = await response.json()
  render(snapshot)
}

function connectStream() {
  const source = new EventSource('/api/stream')
  source.addEventListener('tick', (event) => {
    render(JSON.parse(event.data))
  })
  source.onerror = () => {
    stateElements.statusText.textContent = '实时流断开，正在重连…'
  }
}

stateElements.chatForm.addEventListener('submit', async (event) => {
  event.preventDefault()
  const question = stateElements.chatInput.value.trim()
  if (!question) {
    stateElements.chatAnswer.textContent = '先输入一个问题，例如“明天的峰值大概几点？”'
    return
  }

  stateElements.chatAnswer.textContent = '正在根据当前实时态势生成回答…'
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question }),
  })

  if (!response.ok) {
    stateElements.chatAnswer.textContent = '提问失败，请稍后重试。'
    return
  }

  const result = await response.json()
  stateElements.chatAnswer.textContent = result.answer
})

window.addEventListener('resize', () => {
  if (chart) {
    chart.resize()
  }
})

loadInitialState()
  .then(connectStream)
  .catch((error) => {
    stateElements.statusText.textContent = error.message
  })

if (!chart) {
  stateElements.statusText.textContent = '图表库未加载，已回退为文本数据展示'
}
