import { Tag } from 'antd'
import { datasetStatusMap } from '@/constants/display'
import type { DatasetStatus } from '@/types/domain'

type StatusTagProps = {
  status: DatasetStatus
}

export function StatusTag({ status }: StatusTagProps) {
  const meta = datasetStatusMap[status]

  return (
    <Tag className={`status-tag status-tag--${status}`} bordered={false}>
      {meta.label}
    </Tag>
  )
}
