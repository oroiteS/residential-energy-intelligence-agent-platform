# 居民用电分析与节能建议智能体系统

面向毕业设计的本地化居民用电分析系统，覆盖数据导入、统计分析、
行为分类、负荷预测、节能建议与问答展示。

当前仓库已经包含可运行的五个核心部分：

- `frontend/`：React + Vite 前端仪表盘
- `backend/`：Go + Gin 主服务
- `models_agent/`：Robyn 推理与节能问答服务
- `models/`：离线数据处理、分类训练、预测训练
- `live/`：独立实时演示模块，循环播放测试集连续窗口样本并实时推送展示

当前报告导出支持：

- `PDF`：由智能体先整理摘要正文，再由后端生成正式报告文件

## 文档导航

`doc/` 已按“总览 / 接口 / 数据库 / 建模”收敛，建议按下面顺序阅读：

1. `doc/README.md`：文档索引与边界说明
2. `doc/chinese_project.md`：项目范围、模块职责、部署口径
3. `doc/api_design.md`：前后端与模型服务接口契约
4. `doc/database_design.md`：数据库设计
5. `doc/model_data_pipeline.md`：训练数据流水线

说明：

- `doc/schema.sql` 为 MySQL 初始化脚本

## 仓库结构

```text
backend/       Go 主服务
doc/           项目文档
frontend/      React 前端
live/          独立实时演示模块
models/        离线训练与数据处理
models_agent/  Python 推理与智能体服务
docker-compose.yml
docker-compose.cuda.yml
scripts/deploy/
```

## 脚本化启动

现在默认推荐纯本地进程启动，不再依赖 `docker compose` 拉起
`backend / frontend / models_agent / live`。你只需要自行准备一个可用的
`MySQL`，其余服务可由脚本统一启动。

启动前建议先准备根目录环境变量：

```bash
cp .env.example .env
```

至少确认以下内容：

- `MYSQL_DSN` 指向你本机已启动的 MySQL
- `LLM_*` 已按需配置
- 如需修改端口，可同步调整对应服务环境变量

### Apple Silicon + MPS

```bash
./scripts/deploy/start-apple-mps.sh
```

说明：

- 等价于执行 `./scripts/deploy/start-local.sh`
- 会本地启动 `models_agent / backend / frontend / live`
- 所有日志写入 `.run/logs/`

### Linux + CUDA

```bash
./scripts/deploy/start-linux-cuda.sh
```

说明：

- 等价于执行 `./scripts/deploy/start-local.sh`
- 需要宿主机本地 Python / Go / pnpm 环境已可用

### 通用 CPU

```bash
./scripts/deploy/start-cpu.sh
```

### 统一本地启动

```bash
./scripts/deploy/start-local.sh
```

### 统一停止

```bash
./scripts/deploy/stop.sh
```

说明：

- 等价于执行 `./scripts/deploy/stop-local.sh`
- 会停止 `frontend / live / backend / models_agent`

## Docker 直接命令

如果你仍然想直接使用 `docker compose`，仓库中仍保留了相关配置；
但当前默认推荐使用上面的本地脚本模式。

默认编排会启动完整 CPU 栈：

- `db`：MySQL 8
- `models-agent`：Python 推理与智能体服务
- `backend`：Go API
- `frontend`：Nginx 托管前端并反向代理 `/api`
- `live`：独立实时演示服务，提供循环连续窗口样本播放页面与 SSE 实时流

如需配置 LLM 或切换 Python 服务地址，先准备根目录环境变量文件：

```bash
cp .env.example .env
```

如果你的网络环境无法直接访问 Docker Hub，建议同时在 `.env` 中设置镜像前缀：

```bash
DOCKER_IMAGE_PREFIX=docker.m.daocloud.io/
MYSQL_IMAGE=docker.m.daocloud.io/mysql:8.0
```

说明：

- `DOCKER_IMAGE_PREFIX` 会作用于 `golang / node / nginx / python / alpine` 等构建基础镜像
- `MYSQL_IMAGE` 单独覆盖数据库镜像
- 如果你有自己的企业镜像仓库，也可以替换为对应地址

启动：

```bash
docker compose up -d --build
```

访问地址：

- 前端：<http://127.0.0.1:3000>
- 后端健康检查：<http://127.0.0.1:8080/api/v1/health>
- Python 服务健康检查：<http://127.0.0.1:8001/internal/model/v1/health>
- 实时演示：<http://127.0.0.1:8090>
- MySQL：`127.0.0.1:3306`

停止：

```bash
docker compose down
```

清理并删除数据卷：

```bash
docker compose down -v
```

## CUDA 部署

如果宿主机已安装 `NVIDIA Container Toolkit`，可以叠加 CUDA 覆盖配置：

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d --build
```

该模式会为 `models-agent` 申请 `NVIDIA GPU`，推理设备优先级为：

```text
cuda -> mps -> cpu
```

## MPS 说明

`models_agent` 代码已经支持 `MPS` 检测与推理，但 `MPS` 属于 macOS 的
Metal 能力，标准 Linux 容器无法直接透传该能力。因此：

- `Docker` 全栈部署支持 `CPU` 与 `CUDA`
- `Apple Silicon + MPS` 需要将 `models_agent` 在宿主机本地启动
- 其余 `db / backend / frontend / live` 仍可继续使用 Docker

宿主机 MPS 启动方式：

```bash
cd models_agent
uv sync
APP_HOST=0.0.0.0 APP_PORT=8001 uv run python main.py
```

随后单独启动其余服务：

```bash
PYTHON_SERVICE_BASE_URL=http://host.docker.internal:8001 \
docker compose up -d db backend frontend live
```

如果这里依然报 `auth.docker.io/token`、`failed to authorize` 或 `EOF`，优先检查 `.env` 中是否已经配置镜像前缀。

## LLM 配置

如果需要启用节能问答大模型，可在启动前设置环境变量：

- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL`
- `LLM_TEMPERATURE`
- `LLM_TIMEOUT_SECONDS`

未配置时，智能体接口会自动降级为规则建议，不影响主流程。

## 权重与数据

- `models_agent/checkpoints/`：推理服务默认读取的分类与预测权重
- `models/data/raw/`：离线训练原始数据目录
- `models/data/processed/`：离线训练生成的数据集目录
- `live/data/live_sample.csv`：独立实时演示使用的连续窗口循环样本

仓库当前默认使用：

- 分类权重：`models_agent/checkpoints/classification/tcn/best_model.pt`
- LSTM 权重：`models_agent/checkpoints/forecast/lstm/best_model.pt`
- Transformer 权重：`models_agent/checkpoints/forecast/transformer/best_model.pt`

## 本地开发

### 前端

```bash
cd frontend
pnpm install
pnpm dev
```

### 后端

```bash
cd backend
go run ./cmd/api
```

### Python 推理服务

```bash
cd models_agent
uv sync
uv run python main.py
```

### 离线训练

```bash
cd models
uv sync
uv run python data/process/main.py run-all
uv run python data/process/main.py export-live-sample
uv run python classification/TCN/main.py train
uv run python forecast/LSTM/main.py train
uv run python forecast/transformer/main.py train
```

### 实时演示模块

```bash
cd live
go run ./cmd/server
```

说明：

- 默认读取 `live/data/live_sample.csv`
- 每 `1s` 推进一个虚拟 `15min` 点
- 页面会实时刷新今日负荷、上一完整日分类、下一日预测与问答结果
