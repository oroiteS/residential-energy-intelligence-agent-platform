import { DownloadOutlined, FilePdfOutlined } from '@ant-design/icons'
import { Button, List, Space } from 'antd'
import { SectionCard } from '@/components/common/SectionCard'
import { reportTypeMap } from '@/constants/display'
import { downloadReport } from '@/services/dashboard'
import type { ReportRecord, ReportType } from '@/types/domain'
import { formatDateTime, formatFileLabel, formatFileSize } from '@/utils/formatters'

type ReportsTabProps = {
  reports: ReportRecord[]
  reportActionLoading: boolean
  onExportReport: (reportType: ReportType) => void
}

export function ReportsTab({ reports, reportActionLoading, onExportReport }: ReportsTabProps) {
  return (
    <SectionCard
      title="导出报告"
      subtitle="仅保留 PDF 报告导出，便于统一归档与展示。"
      extra={
        <Space wrap>
          <Button
            type="primary"
            icon={<FilePdfOutlined />}
            loading={reportActionLoading}
            onClick={() => onExportReport('pdf')}
          >
            导出 PDF
          </Button>
        </Space>
      }
    >
      <List
        dataSource={reports}
        locale={{ emptyText: '暂无报告记录' }}
        renderItem={(report) => (
          <List.Item
            actions={[
              <Button
                key="download"
                icon={<DownloadOutlined />}
                type="link"
                onClick={() => void downloadReport(report)}
              >
                下载
              </Button>,
            ]}
          >
            <List.Item.Meta
              avatar={
                report.report_type === 'pdf' ? (
                  <FilePdfOutlined className="report-icon" />
                ) : (
                  <DownloadOutlined className="report-icon" />
                )
              }
              title={`${reportTypeMap[report.report_type]} · #${report.id}`}
              description={`${formatFileLabel(report.file_path)} · ${formatFileSize(report.file_size)} · ${formatDateTime(report.created_at)}`}
            />
          </List.Item>
        )}
      />
    </SectionCard>
  )
}
