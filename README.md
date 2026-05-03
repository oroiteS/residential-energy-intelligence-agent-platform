# 居民用电分析与节能建议系统

这是一个面向毕业设计场景的本地化居民用电分析项目，覆盖数据预处理、
分类与预测训练、在线推理、节能问答、报告导出和实时演示。

当前仓库中可直接关注的核心目录：

- `frontend/`：React + Vite 前端工作台
- `backend/`：Go + Gin 主 API 服务
- `models_agent/`：Python 推理与智能体服务
- `models/`：离线数据处理、分类训练、预测训练
- `live/`：独立实时演示模块

说明：

- `doc/`、`doc-example/`、`mysql_data/`、`models/data/raw/`、
  `models/data/processed/` 等目录已被 `.gitignore` 排除，不作为当前 README
  的维护范围
- 仓库根目录当前没有统一的 `.env.example`，环境变量示例分散在各子模块内

## 项目结构

```text
.
├── backend/        Go 主服务
├── frontend/       React 前端
├── live/           实时演示服务
├── models/         离线建模与数据流水线
├── models_agent/   Python 推理与问答服务
├── docker-compose.yml
├── schema.sql
└── README.md
```

## 模块关系

典型联调链路如下：

1. `models/` 生成训练数据并训练分类/预测模型
2. 训练得到的权重同步到 `models_agent/checkpoints/`
3. `models_agent/` 提供分类、预测、问答和 PDF 渲染接口
4. `backend/` 负责数据导入、分析编排、结果存储、报告导出
5. `frontend/` 调用 `backend/`，并单独连接 `live/` 做实时展示

## 快速启动

### 1. 启动 Python 推理服务

```bash
cd models_agent
cp .env.example .env
uv sync
uv run python main.py
```

默认地址：`http://127.0.0.1:8001`

### 2. 启动 Go 后端

```bash
cd backend
cp .env.example .env
go run ./cmd/api
```

默认推荐通过 `.env` 使用 `127.0.0.1:8080`。
如果未提供 `.env`，代码默认端口为 `8888`。

### 3. 启动前端

```bash
cd frontend
pnpm install
pnpm dev
```

默认地址：`http://127.0.0.1:3000`

开发环境下，Vite 会将 `/api` 代理到 `http://127.0.0.1:8080`。

### 4. 启动实时演示模块

```bash
cd live
go run ./cmd/server
```

默认地址：`http://127.0.0.1:8090`

## 各模块说明

### `frontend/`

- 提供数据集中心、节能问答、服务概览、实时演示、报告中心等页面
- 使用 `Ant Design`、`React Router`、`Zustand`、`ECharts`
- 支持本地 mock 模式和真实后端联调

### `backend/`

- 管理数据集导入、统计分析、分类预测、负荷预测、问答会话、节能建议和报告导出
- 依赖 `MySQL` 持久化业务数据
- 通过 HTTP 调用 `models_agent/` 暴露的模型与智能体接口

### `models_agent/`

- 基于 `Robyn` 提供内部模型接口与智能体接口
- 当前集成 `XGBoost` 分类推理、`TFT` 预测推理、问答工作流、Markdown 转 PDF
- 默认从 `models_agent/checkpoints/` 读取本地权重

### `models/`

- 管理 15 分钟粒度数据预处理、分类数据集构建、预测数据集构建、分类训练、预测训练
- 使用 `uv` 管理 Python 依赖
- 原始数据与生成数据集目录默认被 `.gitignore` 忽略

### `live/`

- 用连续窗口样本模拟实时数据流
- 提供网页展示、SSE 推送和当前状态问答入口
- 运行时直接调用 `models_agent/` 的模型和智能体接口

## 离线建模常用命令

```bash
cd models
uv sync
uv run python data/process/main.py preprocess-base
uv run python data/process/main.py build-classification
uv run python data/process/main.py build-forecast
uv run python classification/xgboost/main.py train
uv run python forecast/tft/train.py
```

补充说明：

- 数据预处理主入口为 `models/data/process/main.py`
- 分类训练主入口为 `models/classification/xgboost/main.py`
- 预测训练主入口为 `models/forecast/tft/train.py`
- 设备检测脚本可使用 `models/cuda_test.py` 与 `models/mps_test.py`

## 关键端口

- 前端：`3000`
- 后端：`8080`（或未加载 `.env` 时的 `8888`）
- Python 推理服务：`8001`
- 实时演示：`8090`
- MySQL：`3306`

## 参考文件

- `backend/.env.example`
- `frontend/.env.example`
- `models_agent/.env.example`
- `schema.sql`
- `models/data/process.md`
