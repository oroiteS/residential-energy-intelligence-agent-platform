import type { ReactNode } from 'react'
import { Card, Typography } from 'antd'

type MetricCardProps = {
  label: string
  value: string
  hint?: string
  accent?: 'amber' | 'teal' | 'olive' | 'coral'
  icon?: ReactNode
}

export function MetricCard({
  label,
  value,
  hint,
  accent = 'amber',
  icon,
}: MetricCardProps) {
  return (
    <Card className={`metric-card metric-card--${accent}`} bordered={false}>
      <div className="metric-card__head">
        <Typography.Text className="metric-card__label">{label}</Typography.Text>
        {icon ? <span className="metric-card__icon">{icon}</span> : null}
      </div>
      <Typography.Title className="metric-card__value" level={3}>
        {value}
      </Typography.Title>
      {hint ? (
        <Typography.Paragraph className="metric-card__hint">
          {hint}
        </Typography.Paragraph>
      ) : null}
    </Card>
  )
}
