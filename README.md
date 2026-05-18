# 居民用电分析与节能建议系统

毕业设计项目：基于电力用户日冻结数据的用电行为分类、负荷预测、异常检测与节能建议系统。

## 项目结构

```text
.
├── backend/            Flask Python 后端（API + 模型推理 + 智能体）
├── frontend/           React + Vite + TypeScript 前端
├── models/             离线数据预处理与模型训练
├── doc/                项目文档与需求规格
├── docker-compose.yml  本地 MySQL 开发环境
├── schema.sql          数据库建表脚本
└── README.md
```

## 模块关系

```
models/ (离线训练)
  │  分类模型权重、预测模型权重
  ▼
backend/models/artifacts/ (权重同步)
  │
backend/ (Flask API + 推理)
  │  /api/v1/*
  ▼
frontend/ (React 工作台)
```

1. `models/` 完成数据预处理、特征工程与模型训练
2. 训练权重同步到 `backend/models/artifacts/`
3. `backend/` 提供 REST API，ML 模型（XGBoost 分类、LSTM/Transformer 预测、Isolation Forest 异常检测）和智能体（LangChain）同进程运行
4. `frontend/` 通过 `/api/v1` 调用后端接口

## 快速启动

### 0. 启动 MySQL

```bash
docker compose up -d
```

首次启动会自动执行 `schema.sql` 建表。

### 1. 启动 Flask 后端

```bash
cd backend
cp .env.example .env   # 按需修改数据库连接等配置
uv sync
uv run python main.py
```

默认地址：`http://127.0.0.1:5000`

### 2. 启动前端

```bash
cd frontend
pnpm install
pnpm dev
```

默认地址：`http://127.0.0.1:3000`

开发环境下 Vite 将 `/api` 代理到 `VITE_BACKEND_BASE_URL`（默认 `http://127.0.0.1:5000`）。

## 各模块说明

### `frontend/`

- 页面：数据集中心、数据集详情、节能问答、服务概览、实时演示、报告中心
- 技术栈：React 19、TypeScript、Vite、Ant Design、React Router、Zustand、ECharts
- 支持 mock 模式和真实后端联调，由 `VITE_USE_MOCK` 环境变量控制
- 详见 `frontend/README.md`

### `backend/`

- Flask REST API，管理数据集导入、统计分析、分类预测、负荷预测、问答会话、节能建议和报告导出
- 依赖 MySQL 持久化业务数据
- ML 模型（XGBoost 分类、LSTM/Transformer 预测、Isolation Forest 异常检测）以 Python 模块同进程运行
- 智能体基于 LangChain 实现，在 Flask 进程内调用 LLM
- 使用 `uv` 管理依赖，Python >= 3.10

### `models/`

- 基于 1000 户 × 365 天电力日冻结数据（总/峰/谷三类）
- 分类任务：KMeans 无监督伪标签 + XGBoost 监督分类（16 维行为特征）
- 预测任务：30 天历史 → 7 天预测，XGBoost/LSTM/Transformer 五种模型对比
- 异常检测：Isolation Forest + 统计规则引擎
- 每个任务独立预处理、配置和输出目录
- 使用 `uv` 管理依赖，PyTorch 训练需在 CUDA 服务器上运行
- 详见 `models/CLAUDE.md`

## 离线建模常用命令

```bash
cd models
uv sync

# 数据预处理
uv run python data/classification/preprocess_classification.py
uv run python data/forecast/preprocess_forecast.py
uv run python data/detection/preprocess_detection.py

# 分类模型训练
uv run python classification/kmeans/train.py
uv run python classification/xgboost/train.py

# 预测模型训练（需 CUDA）
uv run python forecast/xgboost/train_xgboost.py
uv run python forecast/lstm/train.py
uv run python forecast/transformer/train.py
uv run python forecast/lstm_baseline/train.py
uv run python forecast/transformer_baseline/train.py

# 异常检测
uv run python detection/isolation_forest/train.py
uv run python detection/statistical_rules/run.py
```

## 关键端口

| 服务   | 端口  |
|--------|-------|
| 前端   | 3000  |
| 后端   | 5000  |
| MySQL  | 3306  |

## 参考文件

- `backend/.env.example`
- `frontend/.env.example`
- `schema.sql`
- `models/CLAUDE.md`
