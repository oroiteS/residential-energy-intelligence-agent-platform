import type { ClassificationResult } from '@/types/domain'
import { formatDateTime } from '@/utils/formatters'

export type ClassificationTableItem = ClassificationResult & {
  dayLabel: string
}

// 将后端返回的分类窗口转换成页面可直接展示的日期文案。
// 优先使用 window_start/window_end；缺失时回退到结果创建时间，保证表格不会空白。
export function formatClassificationDay(item: ClassificationResult) {
  if (item.window_start && item.window_end) {
    return `${formatDateTime(item.window_start).slice(0, 10)} ~ ${formatDateTime(item.window_end).slice(5, 10)}`
  }
  const source = item.window_start || item.created_at
  return source ? formatDateTime(source).slice(0, 10) : '未知窗口'
}

export function buildClassificationTableItems(
  classificationHistory: ClassificationResult[],
): ClassificationTableItem[] {
  return classificationHistory.map((item) => ({
    ...item,
    dayLabel: formatClassificationDay(item),
  }))
}
