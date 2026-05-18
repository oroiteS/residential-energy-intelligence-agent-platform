# 居民用电分析与节能建议系统

这是一个面向毕业设计场景的本地化居民用电分析项目，覆盖数据预处理、
分类与预测训练、在线推理、节能问答、报告导出和实时演示。

当前仓库中可直接关注的核心目录：

- `frontend/`：React + Vite 前端工作台
- `backend/`：Flask Python 后端（含模型推理与智能体）
- `models/`：离线数据处理、分类训练、预测训练

说明：

- `doc/`、`doc-example/`、`mysql_data/`、`models/data/raw/`、
  `models/data/processed/` 等目录已被 `.gitignore` 排除，不作为当前 README
  的维护范围
- 仓库根目录当前没有统一的 `.env.example`，环境变量示例分散在各子模块内

## 项目结构

```text
.
├── backend/        Flask Python 后端（API + 模型推理 + 智能体）
├── frontend/       React 前端
├── models/         离线建模与数据流水线
├── docker-compose.yml
├── schema.sql
└── README.md
```

## 模块关系

典型联调链路如下：

1. `models/` 生成训练数据并训练分类/预测模型
2. 训练得到的权重同步到 `backend/models/artifacts/`
3. `backend/` 负责数据导入、分析编排、结果存储、报告导出，ML 模型（XGBoost/LSTM/Isolation Forest）和智能体（LangChain）以 Python 模块形式同进程运行
4. `frontend/` 调用 `backend/` 的 `/api/v1` 接口

## 快速启动

### 1. 启动 Flask 后端

```bash
cd backend
cp .env.example .env
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

开发环境下，Vite 会将 `/api` 代理到 `VITE_BACKEND_BASE_URL`（默认 `http://127.0.0.1:5001`）。

## 各模块说明

### `frontend/`

- 提供数据集中心、节能问答、服务概览、实时演示、报告中心等页面
- 使用 `Ant Design`、`React Router`、`Zustand`、`ECharts`
- 支持本地 mock 模式和真实后端联调

### `backend/`

- 管理数据集导入、统计分析、分类预测、负荷预测、问答会话、节能建议和报告导出
- 依赖 `MySQL` 持久化业务数据
- ML 模型（XGBoost 分类、LSTM 预测、Isolation Forest 异常检测）以 Python 模块同进程运行
- 智能体基于 `LangChain` 实现，在 Flask 进程内调用 LLM
- 使用 `uv` 管理 Python 依赖

### `models/`

- 管理 15 分钟粒度数据预处理、分类数据集构建、预测数据集构建、分类训练、预测训练
- 使用 `uv` 管理 Python 依赖
- 原始数据与生成数据集目录默认被 `.gitignore` 忽略

## 离线建模常用命令

```bash
cd models
uv sync
uv run python main.py preprocess-base
uv run python main.py build-classification
uv run python main.py build-forecast
uv run python classification/xgboost/main.py train
uv run python forecast/lstm/train.py
```

补充说明：

- 数据预处理主入口为 `models/main.py`
- 分类训练主入口为 `models/classification/xgboost/main.py`
- 预测训练主入口为 `models/forecast/lstm/train.py`（LSTM-Direct 多步直接预测）
- 设备检测脚本可使用 `models/cuda_test.py` 与 `models/mps_test.py`

## 关键端口

- 前端：`3000`
- 后端：`5000`
- MySQL：`3306`

## 参考文件

- `backend/.env.example`
- `frontend/.env.example`
- `schema.sql`
- `models/data/process.md`
