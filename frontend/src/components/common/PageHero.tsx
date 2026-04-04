import type { PropsWithChildren, ReactNode } from 'react'
import { Typography } from 'antd'

type PageHeroProps = PropsWithChildren<{
  eyebrow: string
  title: string
  description: string
  extra?: ReactNode
  icon?: ReactNode
}>

export function PageHero({
  eyebrow,
  title,
  description,
  extra,
  icon,
  children,
}: PageHeroProps) {
  return (
    <section className="page-hero">
      <div className="page-hero__main">
        <div className="page-hero__eyebrow">
          {icon ? <span className="page-hero__eyebrow-icon">{icon}</span> : null}
          <span>{eyebrow}</span>
        </div>
        <Typography.Title level={2} className="page-hero__title">
          {title}
        </Typography.Title>
        <Typography.Paragraph className="page-hero__desc">
          {description}
        </Typography.Paragraph>
        {children ? <div className="page-hero__content">{children}</div> : null}
      </div>
      {extra ? <div className="page-hero__side">{extra}</div> : null}
    </section>
  )
}
