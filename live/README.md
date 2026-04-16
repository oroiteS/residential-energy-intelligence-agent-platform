# live

`live/` 是独立于主 `backend/`、`frontend/`、`models/` 代码目录运行的实时演示模块。
它不会直接 import 其他目录下的代码，但会在运行时通过 HTTP 调用模型服务与 LangChain 智能体服务。

它的职责只有三件事：

- 自动读取 `live/data/` 下最新复制进去的 `csv` 文件（或读取 `LIVE_DATA_PATH` 指定文件）
- 每 1 秒推进 1 个虚拟 15 分钟点
- 对外提供实时页面、SSE 推送和基于当前状态的问答接口
- 调用真实分类模型、预测模型与 LangChain 问答接口

## 启动

```bash
cd live
go run ./cmd/server
```

默认访问地址：

```text
http://127.0.0.1:8090
```


## 环境变量

- `LIVE_PORT`：服务端口，默认 `8090`
- `LIVE_DATA_PATH`：实时模拟数据文件或目录，默认 `data`
- `LIVE_WEB_PATH`：静态页面目录，默认 `web`
- `LIVE_MODEL_SERVICE_BASE_URL`：模型服务地址，默认 `http://127.0.0.1:8001`
- `LIVE_AGENT_SERVICE_BASE_URL`：智能体服务地址，默认与模型服务相同
- `LIVE_FORECAST_MODEL_TYPE`：预测模型类型，默认 `tft`
- `LIVE_REQUEST_TIMEOUT_SECONDS`：调用模型/智能体服务的超时时间，默认 `60`

## 数据说明

- 输入文件固定为 15 分钟粒度的连续窗口数据
- 每个虚拟日应包含 `96` 条记录
- 当前版本要求至少 `14` 个完整虚拟日；模拟器会从第 `8` 个虚拟日开始播放
- 模拟器会在虚拟 `00:00` 自动完成：
  - 通过 `xgboost` 分类模型刷新上一日分类
  - 通过配置的预测模型基于最近 `7` 个完整虚拟日刷新今日预测与下一日预测

## 接口

- `GET /api/state`：获取当前完整快照
- `GET /api/stream`：SSE 实时流
- `POST /api/chat`：基于当前状态直接提问，会调用 LangChain 智能体接口
