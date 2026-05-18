# 前端工作台

`frontend/` 是居民用电分析系统的 React 前端，负责承载数据集浏览、用电分析、
节能问答、实时演示入口和报告中心等页面。

## 技术栈

- `React 19`
- `TypeScript`
- `Vite`
- `Ant Design`
- `React Router`
- `Zustand`
- `ECharts`

## 页面结构

当前主要页面如下：

- `/datasets`：数据集中心
- `/datasets/:datasetId`：单个数据集详情
- `/chat`：节能问答
- `/overview`：服务概览与系统配置
- `/live`：实时演示页
- `/reports`：报告中心

## 启动方式

```bash
cd frontend
pnpm install
pnpm dev
```

默认开发地址：

```text
http://127.0.0.1:3000
```

## 构建命令

```bash
pnpm build
pnpm preview
```

## 环境变量

可选环境变量示例见 `frontend/.env.example`。

- `VITE_USE_MOCK`
  - 开发环境下不等于 `false` 时启用本地 mock 数据
  - 设为 `false` 后调用真实后端接口
- `VITE_BACKEND_BASE_URL`
  - Vite 开发代理转发到的后端服务地址
  - 默认值：`http://127.0.0.1:5000`
- `VITE_API_PREFIX`
  - 前端请求 API 的路径前缀
  - 默认值：`/api/v1`

## 联调说明

- 开发服务器会将 `/api` 代理到 `VITE_BACKEND_BASE_URL`
- 如果后端运行在其它端口，只需要修改 `frontend/.env.development`

## 目录说明

- `src/app/`：应用外壳与布局
- `src/pages/`：页面级组件
- `src/components/`：图表与通用组件
- `src/services/`：后端接口封装
- `src/store/`：本地状态管理
- `src/mocks/`：mock 数据
- `src/types/`：领域类型定义

## 开发建议

- 纯前端联调阶段可先保持 `VITE_USE_MOCK=true`
- 接真实服务时，优先保证 `backend/` 端口和 `VITE_BACKEND_BASE_URL` 保持一致
