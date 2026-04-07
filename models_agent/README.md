# models_agent

基于 `Robyn` 的 Python 后端，统一承载：

- `classification`：TCN 分类推理
- `forecast`：LSTM / Transformer 预测
- `agent`：LangChain 节能问答
- `pdf`：Markdown 转 PDF

## 目录说明

- `main.py`：服务启动入口
- `app/config.py`：环境变量与路径配置
- `app/bootstrap.py`：Robyn 路由注册
- `configs/classification.yaml`：分类模型推理配置
- `configs/forecast.yaml`：预测模型推理配置，支持在 `lstm` 与 `transformer` 之间切换
- `app/services/classification_service.py`：分类推理封装
- `app/services/forecast_service.py`：预测封装
- `app/services/agent_service.py`：智能体问答与降级逻辑
- `app/services/pdf_service.py`：Markdown 转 PDF 封装
- `app/tools/md2pdf.py`：内置 vendored PDF 转换脚本

## 启动方式

在 `models_agent` 目录下执行：

```bash
./.venv/bin/python main.py
```

默认监听：

- `127.0.0.1:8001`

## 关键环境变量

- `APP_HOST`
- `APP_PORT`
- `ENV_FILE_PATH`
- `CLASSIFICATION_CONFIG_PATH`
- `FORECAST_CONFIG_PATH`
- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL`
- `LLM_TEMPERATURE`
- `LLM_TIMEOUT_SECONDS`
- `LLM_REPORT_TIMEOUT_SECONDS`
- `PDF_RENDER_TIMEOUT_SECONDS`
- `PDF_THEME`
- `PDF_COVER`
- `PDF_TOC`

## 配置读取

默认会先读取：

- `models_agent/.env`

可选模型配置文件：

- `models_agent/configs/classification.yaml`
- `models_agent/configs/forecast.yaml`

`.env` 示例：

```bash
APP_HOST=127.0.0.1
APP_PORT=8001
CLASSIFICATION_CONFIG_PATH=./configs/classification.yaml
FORECAST_CONFIG_PATH=./configs/forecast.yaml
LLM_BASE_URL=https://example.com/v1
LLM_API_KEY=sk-xxxx
LLM_MODEL=deepseek-chat
LLM_TEMPERATURE=0.2
LLM_TIMEOUT_SECONDS=60
LLM_REPORT_TIMEOUT_SECONDS=180
PDF_RENDER_TIMEOUT_SECONDS=180
PDF_THEME=github-light
PDF_COVER=false
PDF_TOC=false
```

读取优先级：

1. 进程环境变量
2. `models_agent/.env`
3. 代码默认值

说明：

- `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` 未配置时，智能体接口会自动降级
- `LLM_REPORT_TIMEOUT_SECONDS` 仅用于报告摘要生成，未配置时回退到 `LLM_TIMEOUT_SECONDS`
- `PDF_RENDER_TIMEOUT_SECONDS` 控制 Markdown 转 PDF 的超时时间
- `PDF_THEME` / `PDF_COVER` / `PDF_TOC` 控制内部 PDF 渲染默认行为
- 如需切换 `.env` 路径，可设置 `ENV_FILE_PATH`
- 如需切换模型配置路径，可设置 `CLASSIFICATION_CONFIG_PATH`、`FORECAST_CONFIG_PATH`

## 已提供接口

### 模型接口

- `GET /internal/model/v1/health`
- `GET /internal/model/v1/model/info`
- `POST /internal/model/v1/predict`
- `POST /internal/model/v1/forecast`

### 智能体接口

- `GET /internal/agent/v1/health`
- `POST /internal/agent/v1/ask`
- `POST /internal/agent/v1/report-summary`
- `POST /internal/agent/v1/render-pdf`

## 说明

- 当前分类与预测已在 `models_agent/app/inference/` 内同步维护本地推理实现
- 默认只读取 `models_agent/checkpoints/` 下的本地权重，不再依赖外部 `models/` 目录
- 分类权重固定放在 `models_agent/checkpoints/classification/tcn/best_model.pt`
- LSTM 权重固定放在 `models_agent/checkpoints/forecast/lstm/best_model.pt`
- Transformer 权重固定放在 `models_agent/checkpoints/forecast/transformer/best_model.pt`
- 后续更新模型时，只需要覆盖对应位置的 `best_model.pt`，无需再改代码或配置
- 预测推理已兼容两种 checkpoint 归一化格式：旧版全局标准化，以及新版 `aggregate=input_window` 样本级归一化
- 若未配置 LangChain 依赖或 LLM 参数，智能体接口会自动降级为规则回答
- `render-pdf` 依赖 `reportlab`，需要在 `models_agent` 的 Python 环境中安装
- 目前保留 `models_agent` 目录名以减少改动；若后续想长期维护，更推荐改名为 `python_backend`
