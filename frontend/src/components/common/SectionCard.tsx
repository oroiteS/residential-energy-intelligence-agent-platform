import type { PropsWithChildren, ReactNode } from 'react'
import { Card, Typography } from 'antd'

type SectionCardProps = PropsWithChildren<{
  title: string
  subtitle?: string
  extra?: ReactNode
  className?: string
}>

export function SectionCard({
  title,
  subtitle,
  extra,
  className,
  children,
}: SectionCardProps) {
  return (
    <Card
      className={['section-card', className].filter(Boolean).join(' ')}
      title={
        <div className="section-card__header">
          <Typography.Text className="section-card__title" strong>
            {title}
          </Typography.Text>
          {subtitle ? (
            <Typography.Paragraph className="section-card__subtitle">
              {subtitle}
            </Typography.Paragraph>
          ) : null}
        </div>
      }
      extra={extra}
      bordered={false}
    >
      {children}
    </Card>
  )
}
