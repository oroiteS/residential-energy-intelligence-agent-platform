// 前端页面查找表。
// 答辩或维护时可以先从这里确认 URL、页面入口文件和对应业务模块，
// 再进入 pages 或 features 查看具体实现。
export const pageRouteMap = [
  {
    path: '/',
    title: '默认入口',
    pageFile: 'src/pages/DatasetsPage.tsx',
    description: '进入系统后默认跳转到数据集中心。',
  },
  {
    path: '/datasets',
    title: '数据集中心',
    pageFile: 'src/pages/DatasetsPage.tsx',
    description: '负责数据集列表、筛选、上传和概览指标。',
  },
  {
    path: '/datasets/:datasetId',
    title: '数据集详情',
    pageFile: 'src/pages/DatasetDetailPage.tsx',
    featureDir: 'src/features/dataset-detail/',
    description: '负责单个数据集的分析图表、分类、异常检测、预测和报告下载。',
  },
  {
    path: '/chat',
    title: '节能问答',
    pageFile: 'src/pages/ChatPage.tsx',
    description: '负责按数据集进行智能体问答和历史会话查看。',
  },
  {
    path: '/overview',
    title: '服务概览',
    pageFile: 'src/pages/SettingsPage.tsx',
    description: '负责系统健康状态和服务配置概览。',
  },
  {
    path: '/reports',
    title: '报告中心',
    pageFile: 'src/pages/ReportsPage.tsx',
    description: '负责统一查看和下载导出报告。',
  },
] as const
