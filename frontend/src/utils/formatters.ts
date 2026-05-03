export function formatNumber(value: number | null | undefined, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--'
  }

  return new Intl.NumberFormat('zh-CN', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value)
}

export function formatPercent(value: number | null | undefined) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--'
  }

  return `${(value * 100).toFixed(1)}%`
}

export function formatDateTime(value: string | null | undefined) {
  if (!value) {
    return '--'
  }

  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value))
}

function isValidDate(value: Date) {
  return !Number.isNaN(value.getTime())
}

export function formatTime(value: string | null | undefined) {
  if (!value) {
    return '--'
  }

  return new Intl.DateTimeFormat('zh-CN', {
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value))
}

export function formatPeriodRange(value: string) {
  const [start, end] = value.split('/')
  if (!start || !end) {
    return value
  }

  const startDate = new Date(start)
  const endDate = new Date(end)
  if (!isValidDate(startDate) || !isValidDate(endDate)) {
    return value
  }

  return `${formatDateTime(start)} - ${formatTime(end)}`
}

export function humanizeDateRanges(value: string) {
  const isoRangePattern =
    /\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})\/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})/g

  return value.replace(isoRangePattern, (range) => formatPeriodRange(range))
}

export function formatFileSize(value: number) {
  if (value < 1024) {
    return `${value} B`
  }

  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`
  }

  return `${(value / (1024 * 1024)).toFixed(1)} MB`
}

export function formatFileLabel(value: string | null | undefined) {
  if (!value) {
    return '--'
  }

  const normalized = value.replace(/\\/g, '/')
  const segments = normalized.split('/').filter(Boolean)
  return segments.at(-1) ?? value
}
