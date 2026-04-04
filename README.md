# 居民用电分析和节能建议智能体系统

面向毕业设计的本地化居民用电分析系统，包含：

- `models/`：离线数据预处理、TCN 分类训练、LSTM/Transformer 预测训练
- `models_agent/`：基于 Robyn 的 Python 推理与节能问答服务
- `backend/`：Go + Gin 主服务，负责数据集管理、统计分析、报告导出和调用 Python 服务
- `frontend/`：React + Vite 前端仪表盘

系统目标是把一份居民用电数据，从导入、清洗、统计分析、行为分类、负荷预测，一路串到前端展示和节能建议生成。

## 当前技术栈

- 后端主服务：Go 1.25 + Gin + GORM + MySQL 8
- Python 建模：Python 3.10+ + PyTorch 2.7+ + `uv`
- Python 服务：Robyn + LangChain
- 前端：React + TypeScript + Vite + Ant Design + ECharts
- 包管理：`uv`、`pnpm`

## 仓库结构

```text
backend/       Go 主服务
doc/           需求、数据库、数据流水线设计文档
frontend/      React 前端
models/        预处理、分类训练、预测训练
models_agent/  Robyn 推理/智能体服务
```

## 完整拿到手流程

下面这套流程是“新机器、本地第一次跑通项目”的推荐顺序。

### 1. 准备环境

建议先安装这些基础依赖：

- Go 1.25+
- Python 3.10+
- `uv`
- Node.js 20+
- `pnpm`
- Docker Desktop 或可用的 Docker Engine

如果你有 GPU，可以使用 CUDA；没有也没关系，`models/` 里的训练和推理会自动按 `cuda -> mps -> cpu` 选择设备。

### 2. 克隆项目

```bash
git clone <your-repo-url>
cd gp
```

### 3. 启动 MySQL

仓库根目录已经提供了 MySQL 8 的 `docker-compose.yml`，首次启动时会自动执行 [doc/schema.sql](/Users/syn/my/projects/gp/doc/schema.sql) 建表。

```bash
docker compose up -d db
```

确认数据库可用：

```bash
docker compose ps
```

默认数据库信息：

- host: `127.0.0.1`
- port: `3306`
- database: `resident`
- user: `root`
- password: `root`

### 4. 安装 `models/` 依赖并生成训练数据

`models/data/raw/` 目录下放的是离线训练用的 REFIT 数据。第一次拿到项目后，先把 15 分钟基础时序、分类数据集、预测数据集全部生成出来。

说明：

- 由于 REFIT 原始 CSV 体积过大，仓库默认不提交 `models/data/raw/*.csv`
- 请自行下载并放入 `models/data/raw/`
- 官方数据集地址：<https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned/>

使用 `uv`：

```bash
cd models
uv sync
uv run python data/process/main.py run-all
```

使用标准 `pip`：

```bash
cd models
python -m venv .venv
source .venv/bin/activate  # Windows 改成 .venv\Scripts\activate
pip install -r requirements.txt
python data/process/main.py run-all
```

如果只想单独执行某一步，也可以：

```bash
uv run python data/process/main.py preprocess-base
uv run python data/process/main.py build-classification
uv run python data/process/main.py build-forecast
```

### 5. 训练分类和预测模型

先训练分类 TCN，再训练预测 LSTM。Transformer 预测是实验对照项，可选。

```bash
cd models
uv run python classification/TCN/main.py train
uv run python forecast/LSTM/main.py train
```

可选的 Transformer 对照实验：

```bash
cd models
uv run python forecast/GPT/main.py train
```

训练产物默认会写到：

- `models/classification/TCN/output/`
- `models/forecast/LSTM/output/`
- `models/forecast/GPT/output/`

### 6. 让 `models_agent` 能读取训练好的权重

`models_agent` 的预测配置默认直接读取 `models/forecast/LSTM/output/best_model.pt`，所以 LSTM 训练完即可直接用。

分类服务默认读取 `models_agent/checkpoints/classification/best_model.pt`，因此第一次跑通时需要把 TCN 最优权重复制过去：

```bash
mkdir -p models_agent/checkpoints/classification
cp models/classification/TCN/output/best_model.pt models_agent/checkpoints/classification/best_model.pt
```

如果你不想复制文件，也可以直接修改 [models_agent/configs/classification.yaml](/Users/syn/my/projects/gp/models_agent/configs/classification.yaml) 的 `predict.checkpoint_path`。

### 7. 启动 Python 推理 / 智能体服务

使用 `uv`：

```bash
cd models_agent
uv sync
uv run python main.py
```

使用标准 `pip`：

```bash
cd models_agent
python -m venv .venv
source .venv/bin/activate  # Windows 改成 .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

默认地址：

- `http://127.0.0.1:8001`

说明：

- 分类接口依赖第 6 步准备好的 TCN checkpoint
- 预测接口默认走 LSTM checkpoint
- 如果未配置 LLM，问答接口会自动降级为规则建议

如需自定义 LLM，可修改 [models_agent/configs/agent.yaml](/Users/syn/my/projects/gp/models_agent/configs/agent.yaml) 或设置环境变量：

- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL`
- `LLM_TEMPERATURE`
- `LLM_TIMEOUT_SECONDS`

### 8. 启动 Go 后端

先复制环境变量模板：

```bash
cd backend
cp .env.example .env
```

默认配置已经对齐本地开发链路：

- 后端端口 `8080`
- MySQL 连接 `root:root@tcp(127.0.0.1:3306)/resident`
- Python 服务地址 `http://127.0.0.1:8001`

启动：

```bash
go run ./cmd/api
```

### 9. 启动前端

前端开发服务器默认跑在 `3000`，并通过 Vite 代理把 `/api` 转发到 `http://localhost:8080`。

```bash
cd frontend
pnpm install
cp .env.example .env.development
pnpm dev
```

默认情况下：

- `VITE_USE_MOCK=false`：走真实后端
- `VITE_USE_MOCK=true`：只走前端 mock 数据

### 10. 验证整条链路

先检查后端和 Python 服务健康状态：

```bash
curl http://127.0.0.1:8001/internal/model/v1/health
curl http://127.0.0.1:8001/internal/agent/v1/health
curl http://127.0.0.1:8080/api/v1/health
```

再打开前端：

```text
http://127.0.0.1:3000
```

推荐的实际使用顺序：

1. 在前端“数据集管理”页面导入一份 CSV
2. 查看数据集详情和基础统计分析
3. 触发行为分类
4. 触发负荷预测或回测
5. 生成节能建议或进入问答页
6. 导出报告

## 建模相关常用命令

### 数据预处理

```bash
cd models
uv run python data/process/main.py run-all
```

如果使用 `pip` 环境：

```bash
cd models
python data/process/main.py run-all
```

### TCN 分类

```bash
cd models
uv run python classification/TCN/main.py train
uv run python classification/TCN/main.py test
uv run python classification/TCN/main.py predict --input data/processed/classification/classification_day_features.csv
```

如果使用 `pip` 环境：

```bash
cd models
python classification/TCN/main.py train
python classification/TCN/main.py test
python classification/TCN/main.py predict --input data/processed/classification/classification_day_features.csv
```

### LSTM 预测

```bash
cd models
uv run python forecast/LSTM/main.py train
uv run python forecast/LSTM/main.py test
```

如果使用 `pip` 环境：

```bash
cd models
python forecast/LSTM/main.py train
python forecast/LSTM/main.py test
```

### Transformer 预测

```bash
cd models
uv run python forecast/GPT/main.py train
uv run python forecast/GPT/main.py test
```

如果使用 `pip` 环境：

```bash
cd models
python forecast/GPT/main.py train
python forecast/GPT/main.py test
```

## 前后端开发说明

### 后端接口前缀

主服务统一走：

```text
/api/v1
```

例如：

- `GET /api/v1/health`
- `POST /api/v1/datasets/import`
- `POST /api/v1/datasets/:id/classifications/predict`
- `POST /api/v1/datasets/:id/forecasts/predict`
- `POST /api/v1/agent/ask`

### Python 服务接口

`models_agent` 提供：

- `GET /internal/model/v1/health`
- `GET /internal/model/v1/model/info`
- `POST /internal/model/v1/predict`
- `POST /internal/model/v1/forecast`
- `POST /internal/model/v1/backtest`
- `GET /internal/agent/v1/health`
- `POST /internal/agent/v1/ask`

## 文档入口

- 需求与系统设计：[doc/chinese_project.md](/Users/syn/my/projects/gp/doc/chinese_project.md)
- 模型数据流水线：[doc/model_data_pipeline.md](/Users/syn/my/projects/gp/doc/model_data_pipeline.md)
- 数据库结构：[doc/schema.sql](/Users/syn/my/projects/gp/doc/schema.sql)

## 常见问题

### 1. 没有 GPU 能跑吗？

可以。训练和推理会自动在 `cuda -> mps -> cpu` 中选择可用设备。

### 2. 如果我只想先看前端页面？

可以直接在 `frontend/.env.development` 里把 `VITE_USE_MOCK=true`，然后执行：

```bash
cd frontend
pnpm install
pnpm dev
```

### 3. 为什么前端起了但接口报错？

优先检查这三项：

- `models_agent` 是否在 `8001` 端口正常启动
- `backend/.env` 中的 `APP_PORT` 是否为 `8080`
- MySQL 是否已启动且 `resident` 数据库建表成功

## 许可证

本项目采用 [MIT License](LICENSE)。
