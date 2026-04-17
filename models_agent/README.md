# models_agent

基于 `Robyn` 的 Python 后端，统一承载：

- `classification`：XGBoost 分类推理
- `forecast`：TFT 预测
- `agent`：LangChain 节能问答
- `pdf`：Markdown 转 PDF

## 目录说明

- `main.py`：服务启动入口
- `app/config.py`：环境变量与路径配置
- `app/bootstrap.py`：Robyn 路由注册
- `configs/classification.yaml`：分类模型推理配置
- `configs/forecast.yaml`：预测模型推理配置，仅保留 `tft`
- `app/services/classification_service.py`：分类推理封装
- `app/services/forecast_service.py`：预测封装
- `app/services/agent_service.py`：智能体服务门面，负责接入工作流与 LLM
- `app/agent/`：智能体内部模块，包含状态建模、意图路由、证据构建、建议规划、短期记忆、报告构建与工作流编排
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

`POST /internal/agent/v1/ask` 当前除了 `answer / citations / actions` 外，还会返回：

- `intent`：当前问题被路由到的意图
- `missing_information`：仍需补充的信息项
- `confidence_level`：当前回答的整体置信等级
- `degraded` / `error_reason`：是否降级及原因

`ask` 与 `report-summary` 的 `context` 当前建议按以下顶层结构传入，接口会做兼容性归一化与校验：

```json
{
  "dataset": {},
  "analysis_summary": {},
  "classification_result": {},
  "forecast_summary": {},
  "recent_history_summary": {},
  "rule_advices": [],
  "user_preferences": {},
  "conversation_state": {}
}
```

约束说明：

- 上述对象型字段必须是 JSON 对象
- `rule_advices` 必须是数组，数组元素只支持对象或字符串
- 未提供的标准字段会自动补默认值
- 非标准字段会继续透传，供后续扩展使用

其中两个核心结果块已收敛为 `v1 schema`：

`classification_result`

```json
{
  "schema_version": "v1",
  "model_type": "xgboost",
  "predicted_label": "day_low_night_high",
  "confidence": 0.88,
  "label_display_name": "晚上高峰型",
  "probabilities": {
    "afternoon_peak": 0.05,
    "day_low_night_high": 0.88,
    "morning_peak": 0.04,
    "all_day_low": 0.03
  }
}
```

说明：

- `schema_version` 当前仅支持 `v1`
- `model_type` 当前仅支持 `xgboost`
- `predicted_label` 仅支持：
  - `afternoon_peak`
  - `day_low_night_high`
  - `morning_peak`
  - `all_day_low`
- `confidence` 与 `probabilities` 取值必须在 `0~1`
- 未显式传入 `label_display_name` 时，会根据 `predicted_label` 自动补齐
- 未显式传入 `confidence` 且已提供 `probabilities` 时，会自动取主标签概率作为置信度

`forecast_summary`

```json
{
  "schema_version": "v1",
  "model_type": "tft",
  "forecast_horizon": "1d",
  "predicted_avg_load_w": 520.0,
  "predicted_peak_load_w": 1680.0,
  "predicted_total_kwh": 12.48,
  "peak_period": "19:00-22:00",
  "risk_flags": ["evening_peak"],
  "confidence_hint": "medium"
}
```

说明：

- `schema_version` 当前仅支持 `v1`
- `model_type` 当前仅支持 `tft`
- `forecast_horizon` 当前仅支持 `1d`
- 数值字段不能为负数
- `risk_flags` 当前仅支持：
  - `evening_peak`
  - `daytime_peak`
  - `high_baseload`
  - `abnormal_rise`
  - `peak_overlap_risk`
- `confidence_hint` 仅支持 `high / medium / low`

## 说明

- 当前分类与预测已在 `models_agent/app/inference/` 内同步维护本地推理实现
- 默认只读取 `models_agent/checkpoints/` 下的本地权重，不再依赖外部 `models/` 目录
- 分类权重固定放在 `models_agent/checkpoints/classification/xgboost/best_model.json`
- TFT 权重固定放在 `models_agent/checkpoints/forecast/tft/best.ckpt`
- 后续更新模型时，只需要覆盖对应位置的模型文件，无需再改代码或配置
- 分类模型更新时，只需要覆盖 `models_agent/checkpoints/classification/xgboost/best_model.json`
- 当前 `models_agent` 中，预测模型统一为 `7天 -> 1天` 的 TFT
- TFT 推理会基于历史 7 天序列自动补齐时间特征，并调用日级 XGBoost 分类器生成 4 维 profile 概率特征
- 当前 repo 中的 `models_agent/checkpoints/forecast/tft/best.ckpt` 可以直接指向离线训练完成的最佳权重
- 若未配置 LangChain 依赖或 LLM 参数，智能体接口会自动降级为规则回答
- 智能体内部已改为“结构化状态 + 工作流”模式：
  - `intent_router`：识别分类解释、预测解释、风险判断、节能建议、跟进问答等意图
  - `evidence_builder`：统一从分类结果、预测摘要、统计信息中提取证据
  - `advice_planner`：将规则建议、分类特征和预测风险整合为候选动作并排序
  - `memory_manager`：提供会话级短期记忆，仅保存在进程内存中
  - `report_builder`：复用同一套结构化分析结果生成报告摘要
- `render-pdf` 依赖 `reportlab`，需要在 `models_agent` 的 Python 环境中安装
- 目前保留 `models_agent` 目录名以减少改动；若后续想长期维护，更推荐改名为 `python_backend`
