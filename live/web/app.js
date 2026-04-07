const stateElements = {
  statusText: document.getElementById('status-text'),
  virtualTime: document.getElementById('virtual-time'),
  weekLoop: document.getElementById('week-loop'),
  houseId: document.getElementById('house-id'),
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

const chart = echarts.init(document.getElementById('trend-chart'))

const labelMap = {
  day_high_night_low: '白天高晚上低型',
  day_low_night_high: '白天低晚上高型',
  all_day_high: '全天高负载型',
  all_day_low: '全天低负载型',
}

const modelMap = {
  tcn: 'TCN',
  lstm: 'Seq2Seq LSTM',
  transformer: 'Transformer',
}

function formatNumber(value, digits = 2) {
  return Number(value || 0).toFixed(digits)
}

function formatModelLabel(value) {
  return modelMap[value] || value || '--'
}

function render(snapshot) {
  stateElements.statusText.textContent = '实时流已连接'
  stateElements.virtualTime.textContent = snapshot.virtualTime
  stateElements.weekLoop.textContent = String(snapshot.weekLoop)
  stateElements.houseId.textContent = snapshot.source.houseId || '--'
  stateElements.todayProgress.textContent = `${formatNumber(snapshot.metrics.progressPercent)}%`
  stateElements.metricCurrent.textContent = `${formatNumber(snapshot.metrics.currentPowerW)} W`
  stateElements.metricKwh.textContent = `${formatNumber(snapshot.metrics.todayCumulativeKWH, 3)} kWh`
  stateElements.metricPeak.textContent = `${formatNumber(snapshot.metrics.todayPeakLoadW)} W`
  stateElements.metricBase.textContent = `${formatNumber(snapshot.metrics.todayBaseLoadW)} W`

  stateElements.classificationDate.textContent = snapshot.latestClassification.date || '--'
  stateElements.classificationLabel.textContent =
    labelMap[snapshot.latestClassification.label] || snapshot.latestClassification.label || '--'
  stateElements.classificationModel.textContent =
    `模型 ${formatModelLabel(snapshot.latestClassification.modelType)}`
  stateElements.classificationExplanation.textContent =
    snapshot.latestClassification.error || snapshot.latestClassification.explanation || '等待完整日分类结果。'

  stateElements.forecastDate.textContent = snapshot.activeForecast.date || '--'
  stateElements.forecastModel.textContent =
    `模型 ${formatModelLabel(snapshot.activeForecast.modelType)}`
  stateElements.forecastAvg.textContent = `${formatNumber(snapshot.activeForecast.avgLoadW)} W`
  stateElements.forecastPeak.textContent = `${formatNumber(snapshot.activeForecast.peakLoadW)} W`
  stateElements.forecastExplanation.textContent =
    snapshot.activeForecast.error || snapshot.activeForecast.explanation || '等待预测结果。'
  renderRiskFlags(snapshot.activeForecast.riskFlags || [])

  renderChart(snapshot)
}

function renderRiskFlags(flags) {
  if (flags.length === 0) {
    stateElements.forecastRisks.innerHTML = '<span class="soft-tag">暂无风险提示</span>'
    return
  }
  stateElements.forecastRisks.innerHTML = flags
    .map((flag) => `<span class="soft-tag">${flag}</span>`)
    .join('')
}

function renderChart(snapshot) {
  const labels = snapshot.activeForecast.points.map((point) => point.timeLabel)
  const actualSeries = new Array(labels.length).fill(null)
  snapshot.todayPoints.forEach((point) => {
    actualSeries[point.slotIndex] = point.aggregate
  })
  const forecastSeries = snapshot.activeForecast.points.map((point) => point.predicted)

  chart.setOption({
    animationDuration: 300,
    backgroundColor: 'transparent',
    color: ['#f1b875', '#7dd3c7'],
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
      data: ['今日实测', '下一日预测'],
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
        name: '下一日预测',
        type: 'line',
        smooth: true,
        showSymbol: false,
        data: forecastSeries,
        itemStyle: {
          color: '#7dd3c7',
        },
        lineStyle: {
          width: 2,
          type: 'dashed',
          color: '#7dd3c7',
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
  chart.resize()
})

loadInitialState()
  .then(connectStream)
  .catch((error) => {
    stateElements.statusText.textContent = error.message
  })
