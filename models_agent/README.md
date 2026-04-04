# models_agent

基于 `Robyn` 的 Python 后端，统一承载：

- `classification`：TCN 分类推理
- `forecast`：LSTM 预测与回测
- `agent`：LangChain 节能问答

## 目录说明

- `main.py`：服务启动入口
- `app/config.py`：环境变量与路径配置
- `app/bootstrap.py`：Robyn 路由注册
- `configs/classification.yaml`：分类模型推理配置
- `configs/forecast.yaml`：预测模型推理配置，默认指向 `models/forecast/LSTM/output`
- `configs/agent.yaml`：智能体配置
- `app/services/classification_service.py`：分类推理封装
- `app/services/forecast_service.py`：预测与回测封装
- `app/services/agent_service.py`：智能体问答与降级逻辑

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
- `AGENT_CONFIG_PATH`
- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL`
- `LLM_TEMPERATURE`
- `LLM_TIMEOUT_SECONDS`

## 智能体配置文件

默认配置文件：

- `models_agent/configs/agent.yaml`
- `models_agent/configs/classification.yaml`
- `models_agent/configs/forecast.yaml`

格式如下：

```yaml
llm:
  base_url: "https://example.com/v1"
  api_key: "sk-xxxx"
  model: "deepseek-chat"
  temperature: 0.2
  timeout_seconds: 60
```

读取优先级：

1. 环境变量
2. `models_agent/configs/*.yaml`
3. 代码默认值

说明：

- `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` 未配置时，智能体接口会自动降级
- 如需切换配置文件路径，可设置 `AGENT_CONFIG_PATH`
- 如需切换模型配置路径，可设置 `CLASSIFICATION_CONFIG_PATH`、`FORECAST_CONFIG_PATH`

## 已提供接口

### 模型接口

- `GET /internal/model/v1/health`
- `GET /internal/model/v1/model/info`
- `POST /internal/model/v1/predict`
- `POST /internal/model/v1/forecast`
- `POST /internal/model/v1/backtest`

### 智能体接口

- `GET /internal/agent/v1/health`
- `POST /internal/agent/v1/ask`

## 说明

- 当前分类与预测已在 `models_agent/app/inference/` 内提供最小本地推理实现
- 预测模块默认直接读取 `models/forecast/LSTM/output/best_model.pt`
- 预测推理已兼容两种 checkpoint 归一化格式：旧版全局标准化，以及新版 `aggregate=input_window` 样本级归一化
- 如需固定到独立副本，也可以自行改回 `models_agent/checkpoints/`
- 若未配置 LangChain 依赖或 LLM 参数，智能体接口会自动降级为规则回答
- 目前保留 `models_agent` 目录名以减少改动；若后续想长期维护，更推荐改名为 `python_backend`
