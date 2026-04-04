# API 设计文档

> 项目：居民用电分析与节能建议系统  
> 版本：v1  
> 适用范围：前端 React、Go 主服务、Python Robyn 后端联调  
> 关联文档：[chinese_project.md](./chinese_project.md) / [database_design.md](./database_design.md) / [schema.sql](./schema.sql)

---

## 1. 文档目标

本文档用于统一前后端并行开发时的接口契约，解决以下问题：

1. 前端明确页面需要调用哪些接口
2. 后端明确每个接口的请求与响应格式
3. Python Robyn 后端明确对外暴露的推理与智能体协议
4. 联调时减少字段名、状态值、错误码反复修改

本文档分为两层：

- 前端 ↔ Go 主服务 API
- Go 主服务 ↔ Python Robyn 后端 API

---

## 2. 设计原则

### 2.1 统一前缀与版本

- 前端访问 Go 主服务统一使用：`/api/v1`
- Go 主服务访问 Python Robyn 后端的模型路由统一使用：`/internal/model/v1`
- Go 主服务访问 Python Robyn 后端的智能体路由统一使用：`/internal/agent/v1`

### 2.2 JSON 优先

- 除文件上传和文件下载外，全部接口使用 `application/json`
- 时间字段统一返回 ISO 8601 格式，示例：`2026-03-23T15:30:00+08:00`

### 2.3 前端友好

- 列表接口统一支持分页
- 详情接口优先返回可直接渲染的数据结构
- 图表接口直接返回图表数据，不要求前端二次计算

### 2.4 错误可解释

- HTTP 状态码表达技术层结果
- `code` 字段表达业务语义
- 发生错误时返回清晰的可展示信息

### 2.5 降级优先

- LLM 不可用时，不影响统计分析、分类结果、报告导出等主流程
- 智能问答失败时，前端仍可展示规则建议

---

## 3. 通用约定

## 3.1 统一响应结构

除文件下载接口外，统一返回：

```json
{
  "code": "OK",
  "message": "success",
  "data": {},
  "request_id": "req_20260323_000001",
  "timestamp": "2026-03-23T15:30:00+08:00"
}
```

字段说明：

- `code`：业务码
- `message`：简要说明
- `data`：业务数据
- `request_id`：请求追踪 ID
- `timestamp`：服务端响应时间

## 3.2 统一错误结构

```json
{
  "code": "DATASET_NOT_FOUND",
  "message": "数据集不存在",
  "data": {
    "dataset_id": 123
  },
  "request_id": "req_20260323_000002",
  "timestamp": "2026-03-23T15:31:00+08:00"
}
```

## 3.3 HTTP 状态码约定

- `200 OK`：查询成功、普通操作成功
- `201 Created`：创建成功
- `202 Accepted`：异步任务已受理
- `400 Bad Request`：请求格式错误
- `404 Not Found`：资源不存在
- `409 Conflict`：状态冲突或重复操作
- `422 Unprocessable Entity`：业务校验失败
- `500 Internal Server Error`：服务内部错误
- `502 Bad Gateway`：主服务调用 Python 后端或其 LLM 能力失败

## 3.4 分页结构

列表接口统一返回：

```json
{
  "items": [],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total": 125
  }
}
```

## 3.5 枚举值约定

### 数据集状态

```text
uploaded
processing
ready
error
```

### 模型类型

```text
classification: tcn
forecasting: lstm
```

说明：

- 分类任务当前仅开放 `tcn`
- 预测任务当前开放 `lstm`
- `transformer` 作为预留扩展模型，当前不在 Python 后端启用列表中
- `model_type` 的合法值由接口所属任务决定

### 预测模型类型

```text
lstm
```

预留扩展值：

```text
transformer
```

### 行为分类标签

```text
day_high_night_low
day_low_night_high
all_day_high
all_day_low
```

标签语义约定：

- `day_high_night_low`：白天高晚上低型
- `day_low_night_high`：白天低晚上高型
- `all_day_high`：全天高负载型
- `all_day_low`：全天低负载型

说明：

- `predicted_label` 必须返回上述稳定英文标识
- 中文名称仅用于展示层映射，不写入接口枚举值
- 分类训练标签来源于规则标签生成器，而不是聚类结果

### 建议来源

```text
rule
llm
```

### 报告类型

```text
excel
html
pdf
```

## 3.6 通用请求参数格式

### 路径参数

- 路径中的 `{id}`、`{dataset_id}`、`{report_id}` 等均为正整数
- 若路径参数非法，返回 `400 INVALID_REQUEST`

示例：

```text
GET /api/v1/datasets/1
GET /api/v1/reports/6/download
```

### 查询参数

列表接口统一遵循以下约定：

- `page`：页码，从 1 开始，默认 1
- `page_size`：每页数量，默认 20，最大 100
- `keyword`：关键词模糊查询
- `status`：按状态过滤
- `model_type`：按模型类型过滤
- `sort_by`：排序字段
- `sort_order`：`asc` / `desc`

示例：

```text
GET /api/v1/datasets?page=1&page_size=20&status=ready&keyword=house
```

### JSON 请求体

- 字段统一使用 `snake_case`
- 布尔值必须使用 `true` / `false`
- 时间字段统一使用 ISO 8601
- 空值统一使用 `null`

### 文件上传

- 统一使用 `multipart/form-data`
- 文件字段固定命名为 `file`
- JSON 类型扩展字段通过字符串形式传递，后端负责反序列化

## 3.7 公共数据结构

以下结构在多个接口中重复出现，建议前后端共享同一份类型定义。

### `Pagination`

```json
{
  "page": 1,
  "page_size": 20,
  "total": 125
}
```

字段说明：

- `page`：当前页码
- `page_size`：当前页大小
- `total`：总记录数

### `PeakValleyConfig`

```json
{
  "peak": ["07:00-11:00", "18:00-23:00"],
  "valley": ["23:00-07:00"]
}
```

字段说明：

- `peak`：峰时段数组
- `valley`：谷时段数组

### `DatasetSummary`

```json
{
  "id": 1,
  "name": "REFIT House 1",
  "description": "训练用样本",
  "household_id": "house_1",
  "row_count": 8760,
  "time_start": "2014-01-01T00:00:00+08:00",
  "time_end": "2014-12-31T23:00:00+08:00",
  "status": "ready",
  "created_at": "2026-03-23T15:40:00+08:00",
  "updated_at": "2026-03-23T15:40:00+08:00"
}
```

### `DatasetDetail`

在 `DatasetSummary` 基础上补充：

- `raw_file_path`
- `processed_file_path`
- `feature_cols`
- `column_mapping`
- `quality_report_path`
- `error_message`

### `AnalysisSummary`

```json
{
  "total_kwh": 3650.42,
  "daily_avg_kwh": 10.01,
  "max_load_w": 5120.5,
  "max_load_time": "2014-07-12T20:00:00+08:00",
  "min_load_w": 52.4,
  "min_load_time": "2014-03-11T03:00:00+08:00",
  "peak_kwh": 1679.19,
  "valley_kwh": 693.58,
  "flat_kwh": 1277.65,
  "peak_ratio": 0.46,
  "valley_ratio": 0.19,
  "flat_ratio": 0.35
}
```

### `CitationItem`

用于表示智能体回答依据，不是外部链接，而是结构化指标证据。

```json
{
  "key": "peak_ratio",
  "label": "峰时占比",
  "value": 0.46
}
```

字段说明：

- `key`：机器可识别的依据键
- `label`：前端展示名称
- `value`：依据对应的实际值，可为数字、字符串、数组或对象

### `ClassificationResult`

```json
{
  "id": 8,
  "dataset_id": 1,
  "model_type": "tcn",
  "predicted_label": "day_low_night_high",
  "confidence": 0.83,
  "probabilities": {
    "day_high_night_low": 0.06,
    "day_low_night_high": 0.83,
    "all_day_high": 0.07,
    "all_day_low": 0.04
  },
  "explanation": "夜间均值显著高于白天均值，且 night_mean/day_mean = 1.46。",
  "window_start": null,
  "window_end": null,
  "created_at": "2026-03-23T15:50:00+08:00"
}
```

### `ForecastSummary`

```json
{
  "forecast_start": "2014-12-04T00:00:00+08:00",
  "forecast_end": "2014-12-04T23:45:00+08:00",
  "granularity": "15min",
  "predicted_total_kwh": 78.4,
  "predicted_daily_avg_kwh": 11.2,
  "forecast_peak_periods": [
    "2014-12-02T19:00:00+08:00/2014-12-02T22:00:00+08:00"
  ],
  "predicted_peak_ratio": 0.43,
  "predicted_valley_ratio": 0.21,
  "predicted_flat_ratio": 0.36,
  "risk_flags": [
    "evening_peak_risk",
    "night_load_risk"
  ]
}
```

字段说明：

- `forecast_start` / `forecast_end` / `granularity`：摘要对应的预测时间范围
- `predicted_total_kwh`：预测时间范围内总用电量估计
- `predicted_daily_avg_kwh`：预测日均用电量估计
- `forecast_peak_periods`：预测高负荷时段列表
- `predicted_peak_ratio` / `predicted_valley_ratio` / `predicted_flat_ratio`：预测峰谷平占比
- `risk_flags`：供前端和 LLM 共同使用的风险标签

补充说明：

- `ForecastSummary` 不是 Python 模型的原始返回结构
- `ForecastSummary` 由 Go 主服务根据模型返回的原始 `predictions` 序列、请求时间范围和峰谷规则生成
- `forecast_results.summary` 用于缓存该摘要，供列表页、详情页和智能体复用

### `AdviceSummary`

```json
{
  "id": 12,
  "dataset_id": 1,
  "classification_id": 8,
  "advice_type": "rule",
  "summary": "将洗衣与热水器运行时段调整到谷时段",
  "created_at": "2026-03-23T16:05:00+08:00"
}
```

### `ChatMessage`

```json
{
  "id": 21,
  "session_id": 3,
  "role": "assistant",
  "content": "你的夜间用电偏高，主要原因是持续基础负荷。",
  "content_path": null,
  "model_name": "deepseek-chat",
  "tokens_used": 512,
  "created_at": "2026-03-23T16:10:00+08:00"
}
```

### `ReportRecord`

```json
{
  "id": 6,
  "dataset_id": 1,
  "report_type": "excel",
  "file_path": "./outputs/reports/report_6.xlsx",
  "file_size": 1048576,
  "created_at": "2026-03-23T16:15:00+08:00"
}
```

## 3.8 字段类型约定

### ID 类字段

- 类型：整数
- 命名：`id` 或 `*_id`

### 比例类字段

- 类型：浮点数
- 取值范围：`0 ~ 1`
- 示例：`peak_ratio`、`confidence`

### 时间类字段

- 类型：字符串
- 格式：ISO 8601
- 示例：`2026-03-23T16:15:00+08:00`

### 文件路径字段

- 类型：字符串
- 命名：统一以 `*_path` 结尾
- 示例：`detail_path`、`file_path`、`content_path`

### JSON 扩展字段

- 类型：对象
- 用途：保存半结构化业务数据
- 示例：`column_mapping`、`metrics`、`probabilities`

## 3.9 接口描述模板

为了保证后续文档格式统一，新增接口时建议按照以下顺序书写：

1. 接口名称
2. 请求方法与路径
3. 接口用途
4. 请求头或 Content-Type
5. 路径参数说明
6. 查询参数说明
7. 请求体说明
8. 成功响应示例
9. 错误码与失败场景
10. 备注或状态流转说明

---

## 4. 前端 ↔ Go 主服务 API

## 4.1 健康检查

### `GET /api/v1/health`

用于前端启动时判断后端可用性。

响应示例：

```json
{
  "code": "OK",
  "message": "success",
  "data": {
    "service": "go-api",
    "status": "up",
    "version": "v1"
  },
  "request_id": "req_xxx",
  "timestamp": "2026-03-23T15:30:00+08:00"
}
```

---

## 4.2 系统配置接口

## 4.2.1 获取系统配置

### `GET /api/v1/system/config`

返回前端需要展示的系统级配置。

响应 `data`：

```json
{
  "peak_valley_config": {
    "peak": ["07:00-11:00", "18:00-23:00"],
    "valley": ["23:00-07:00"]
  },
  "model_history_window_config": {
    "classification_days": 1,
    "forecast_history_days": 3
  },
  "energy_advice_prompt_template": "这是居民过去{{history_days}}天的实际用电情况、未来一段时间的预测用电情况，以及居民用电行为分类。请基于统计分析结果、历史用电摘要、未来预测摘要和分类结果，给出具体、可执行、可解释的节能建议，并指出关键依据。",
  "data_upload_dir": "./uploads/datasets",
  "report_output_dir": "./outputs/reports",
  "default_llm_id": 1
}
```

## 4.2.2 更新系统配置

### `PATCH /api/v1/system/config`

请求体：

```json
{
  "peak_valley_config": {
    "peak": ["07:00-11:00", "18:00-23:00"],
    "valley": ["23:00-07:00"]
  },
  "model_history_window_config": {
    "classification_days": 1,
    "forecast_history_days": 3
  },
  "energy_advice_prompt_template": "这是居民过去{{history_days}}天的实际用电情况、未来一段时间的预测用电情况，以及居民用电行为分类。请基于统计分析结果、历史用电摘要、未来预测摘要和分类结果，给出具体、可执行、可解释的节能建议，并指出关键依据。"
}
```

说明：

- 允许局部更新
- 仅支持文档中已定义的系统配置项
- `model_history_window_config` 用于配置分类与预测模型所需的历史窗口，当前默认值固定为 `classification_days = 1`、`forecast_history_days = 3`
- `energy_advice_prompt_template` 用于配置节能建议智能体的提示词模板
- 智能体模块实现语言固定为 Python，Go 主服务通过 HTTP/JSON 调用 Python Robyn 后端，不在系统配置中暴露运行时语言切换

---

## 4.3 LLM 配置接口

## 4.3.1 获取配置列表

### `GET /api/v1/llm-configs`

响应 `data`：

```json
{
  "items": [
    {
      "id": 1,
      "name": "deepseek-local",
      "base_url": "https://example.com/v1",
      "model_name": "deepseek-chat",
      "temperature": 0.2,
      "timeout_seconds": 60,
      "is_default": true,
      "created_at": "2026-03-23T15:30:00+08:00",
      "updated_at": "2026-03-23T15:30:00+08:00"
    }
  ]
}
```

## 4.3.2 创建配置

### `POST /api/v1/llm-configs`

请求体：

```json
{
  "name": "deepseek-local",
  "base_url": "https://example.com/v1",
  "api_key": "sk-xxxx",
  "model_name": "deepseek-chat",
  "temperature": 0.2,
  "timeout_seconds": 60,
  "is_default": true
}
```

## 4.3.3 更新配置

### `PUT /api/v1/llm-configs/{id}`

## 4.3.4 删除配置

### `DELETE /api/v1/llm-configs/{id}`

约束：

- 默认配置若被删除，需先切换默认项

## 4.3.5 设置默认配置

### `POST /api/v1/llm-configs/{id}/set-default`

---

## 4.4 数据集接口

## 4.4.1 上传并导入数据集

### `POST /api/v1/datasets/import`

`multipart/form-data`

表单字段：

- `file`：必填，CSV 或 xlsx 文件
- `name`：必填，数据集名称
- `description`：可选
- `household_id`：可选
- `unit`：可选，`kwh` / `wh` / `w`
- `column_mapping`：可选，JSON 字符串

`column_mapping` 示例：

```json
{
  "Time": "timestamp",
  "Aggregate": "aggregate",
  "Issues": "issues",
  "Appliance1": "appliance",
  "Appliance2": "appliance"
}
```

说明：

- 导入协议最少只要求时间戳列和总用电量列
- `column_mapping` 采用“原始列名 -> 标准语义”映射
- 标准语义当前固定为：
  - `timestamp`
  - `aggregate`
  - `issues`
  - `appliance`
- 标记为 `appliance` 的列允许出现任意次，不限制数量
- 若未提供任何 `appliance` 列，接口仍允许导入成功，但仅保证基础统计链路可用
- 若自动识别失败，后端返回 `COLUMN_MAPPING_REQUIRED`，由前端提示用户手动补充映射

成功响应（`202 Accepted`）：

```json
{
  "code": "ACCEPTED",
  "message": "导入任务已受理",
  "data": {
    "dataset": {
      "id": 1,
      "name": "REFIT House 1",
      "status": "processing",
      "row_count": 0,
      "time_start": null,
      "time_end": null,
      "quality_report_path": null,
      "created_at": "2026-03-23T15:40:00+08:00"
    }
  },
  "request_id": "req_xxx",
  "timestamp": "2026-03-23T15:40:00+08:00"
}
```

失败场景：

- 自动识别列失败，返回 `422 COLUMN_MAPPING_REQUIRED`
- 文件格式错误，返回 `422 UNSUPPORTED_FILE_TYPE`

## 4.4.2 获取数据集列表

### `GET /api/v1/datasets?page=1&page_size=20&status=ready&keyword=house`

响应 `data`：

```json
{
  "items": [
    {
      "id": 1,
      "name": "REFIT House 1",
      "description": "训练用样本",
      "household_id": "house_1",
      "row_count": 8760,
      "time_start": "2014-01-01T00:00:00+08:00",
      "time_end": "2014-12-31T23:00:00+08:00",
      "status": "ready",
      "created_at": "2026-03-23T15:40:00+08:00",
      "updated_at": "2026-03-23T15:40:00+08:00"
    }
  ],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total": 1
  }
}
```

## 4.4.3 获取数据集详情

### `GET /api/v1/datasets/{id}`

响应 `data`：

```json
{
  "dataset": {
    "id": 1,
    "name": "REFIT House 1",
    "description": "训练用样本",
    "household_id": "house_1",
    "raw_file_path": "./uploads/datasets/house1.csv",
    "processed_file_path": "./uploads/datasets/house1_15min.csv",
    "row_count": 8760,
    "time_start": "2014-01-01T00:00:00+08:00",
    "time_end": "2014-12-31T23:00:00+08:00",
    "feature_cols": ["Time", "Aggregate"],
    "column_mapping": {
      "Time": "timestamp",
      "Aggregate": "value"
    },
    "status": "ready",
    "quality_report_path": "./outputs/quality/dataset_1.json",
    "error_message": null,
    "created_at": "2026-03-23T15:40:00+08:00",
    "updated_at": "2026-03-23T15:40:00+08:00"
  },
  "quality_summary": {
    "missing_rate": 0.01,
    "duplicate_count": 0,
    "sampling_interval": "15min",
    "cleaning_strategies": [
      "缺失值插值",
      "异常值裁剪",
      "15 分钟粒度重采样"
    ]
  }
}
```

## 4.4.4 删除数据集

### `DELETE /api/v1/datasets/{id}`

说明：

- 需要同时清理关联结果数据与文件资源

---

## 4.5 统计分析接口

## 4.5.1 获取统计分析结果

### `GET /api/v1/datasets/{id}/analysis`

说明：

- 若分析结果不存在，返回 `404 ANALYSIS_NOT_FOUND`
- 该接口直接服务于详情页卡片和图表区

响应 `data`：

```json
{
  "summary": {
    "total_kwh": 3650.42,
    "daily_avg_kwh": 10.01,
    "max_load_w": 5120.5,
    "max_load_time": "2014-07-12T20:00:00+08:00",
    "min_load_w": 52.4,
    "min_load_time": "2014-03-11T03:00:00+08:00",
    "peak_kwh": 1679.19,
    "valley_kwh": 693.58,
    "flat_kwh": 1277.65,
    "peak_ratio": 0.46,
    "valley_ratio": 0.19,
    "flat_ratio": 0.35
  },
  "peak_valley_config": {
    "peak": ["07:00-11:00", "18:00-23:00"],
    "valley": ["23:00-07:00"]
  },
  "charts": {
    "daily_trend": [
      {
        "date": "2014-01-01",
        "kwh": 10.2
      }
    ],
    "weekly_trend": [
      {
        "week_start": "2014-01-01",
        "week_end": "2014-01-07",
        "kwh": 71.4
      }
    ],
    "typical_day_curve": [
      {
        "hour": 0,
        "avg_load_w": 120.5
      }
    ],
    "peak_valley_pie": [
      {
        "name": "峰时",
        "ratio": 0.46,
        "kwh": 1679.19
      }
    ]
  },
  "detail_path": "./outputs/analysis/dataset_1.json",
  "updated_at": "2026-03-23T15:45:00+08:00"
}
```

## 4.5.2 重新生成统计分析

### `POST /api/v1/datasets/{id}/analysis/recompute`

适用场景：

- 重新处理峰谷配置后重算
- 修正数据集后重算

返回：

- 若同步执行，返回 `200`
- 若未来改为异步，可返回 `202`

---

## 4.6 行为分类接口

## 4.6.1 触发分类推理

### `POST /api/v1/datasets/{id}/classifications/predict`

请求体：

```json
{
  "model_type": "tcn",
  "window_mode": "full_dataset",
  "window_start": null,
  "window_end": null,
  "force_refresh": false
}
```

字段说明：

- `model_type`：必填
- `window_mode`：`full_dataset` / `time_window`
- `force_refresh`：是否忽略已有结果重新推理

响应 `data`：

```json
{
  "classification": {
    "id": 8,
    "dataset_id": 1,
    "model_type": "tcn",
    "predicted_label": "day_low_night_high",
    "confidence": 0.83,
    "probabilities": {
      "day_high_night_low": 0.06,
      "day_low_night_high": 0.83,
      "all_day_high": 0.07,
      "all_day_low": 0.04
    },
    "explanation": "夜间均值显著高于白天均值，且 night_mean/day_mean = 1.46。",
    "window_start": null,
    "window_end": null,
    "created_at": "2026-03-23T15:50:00+08:00"
  }
}
```

## 4.6.2 获取最新分类结果

### `GET /api/v1/datasets/{id}/classifications/latest?model_type=tcn`

## 4.6.3 获取分类结果列表

### `GET /api/v1/datasets/{id}/classifications?page=1&page_size=20&model_type=tcn`

---

## 4.7 时序预测接口

## 4.7.1 触发时序预测

### `POST /api/v1/datasets/{id}/forecasts/predict`

请求体：

```json
{
  "model_type": "lstm",
  "granularity": "15min",
  "forecast_start": "2014-12-04T00:00:00+08:00",
  "forecast_end": "2014-12-04T23:45:00+08:00",
  "force_refresh": false
}
```

字段说明：

- `model_type`：必填，使用哪个模型进行预测
- `granularity`：预测粒度，当前建模默认 `15min`
- `forecast_start` / `forecast_end`：预测时间范围
- `force_refresh`：是否忽略已有结果重新预测
- 响应中的 `summary` 固定使用 `ForecastSummary` 结构，供前端和智能体复用
- 该摘要由 Go 主服务生成并写入 `forecast_results.summary`

响应 `data`：

```json
{
  "forecast": {
    "id": 3,
    "dataset_id": 1,
    "model_type": "lstm",
    "forecast_start": "2014-12-04T00:00:00+08:00",
    "forecast_end": "2014-12-04T23:45:00+08:00",
    "granularity": "15min",
    "summary": {
      "forecast_start": "2014-12-04T00:00:00+08:00",
      "forecast_end": "2014-12-04T23:45:00+08:00",
      "granularity": "15min",
      "predicted_total_kwh": 78.4,
      "predicted_daily_avg_kwh": 11.2,
      "forecast_peak_periods": [
        "2014-12-02T19:00:00+08:00/2014-12-02T22:00:00+08:00"
      ],
      "predicted_peak_ratio": 0.43,
      "predicted_valley_ratio": 0.21,
      "predicted_flat_ratio": 0.36,
      "risk_flags": [
        "evening_peak_risk",
        "night_load_risk"
      ]
    },
    "detail_path": "./outputs/forecasts/fc_3.json",
    "created_at": "2026-03-23T15:50:00+08:00"
  }
}
```

## 4.7.2 获取预测结果列表

### `GET /api/v1/datasets/{id}/forecasts?page=1&page_size=20&model_type=lstm`

响应 `data`：

```json
{
  "items": [
    {
      "id": 3,
      "dataset_id": 1,
      "model_type": "lstm",
      "forecast_start": "2014-12-04T00:00:00+08:00",
      "forecast_end": "2014-12-04T23:45:00+08:00",
      "granularity": "15min",
      "summary": {
        "forecast_start": "2014-12-04T00:00:00+08:00",
        "forecast_end": "2014-12-04T23:45:00+08:00",
        "granularity": "15min",
        "predicted_total_kwh": 78.4,
        "predicted_daily_avg_kwh": 11.2,
        "forecast_peak_periods": [
          "2014-12-02T19:00:00+08:00/2014-12-02T22:00:00+08:00"
        ],
        "predicted_peak_ratio": 0.43,
        "predicted_valley_ratio": 0.21,
        "predicted_flat_ratio": 0.36,
        "risk_flags": [
          "evening_peak_risk"
        ]
      },
      "created_at": "2026-03-23T15:50:00+08:00"
    }
  ],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total": 1
  }
}
```

## 4.7.3 获取单次预测详情

### `GET /api/v1/forecasts/{forecast_id}`

响应 `data`：

```json
{
  "forecast": {
    "id": 3,
    "dataset_id": 1,
    "model_type": "lstm",
    "forecast_start": "2014-12-04T00:00:00+08:00",
    "forecast_end": "2014-12-04T23:45:00+08:00",
    "granularity": "15min",
    "summary": {
      "forecast_start": "2014-12-04T00:00:00+08:00",
      "forecast_end": "2014-12-04T23:45:00+08:00",
      "granularity": "15min",
      "predicted_total_kwh": 78.4,
      "predicted_daily_avg_kwh": 11.2,
      "forecast_peak_periods": [
        "2014-12-02T19:00:00+08:00/2014-12-02T22:00:00+08:00"
      ],
      "predicted_peak_ratio": 0.43,
      "predicted_valley_ratio": 0.21,
      "predicted_flat_ratio": 0.36,
      "risk_flags": [
        "evening_peak_risk",
        "night_load_risk"
      ]
    },
    "detail_path": "./outputs/forecasts/fc_3.json",
    "created_at": "2026-03-23T15:50:00+08:00"
  },
  "series": [
    {
      "timestamp": "2014-12-01T00:00:00+08:00",
      "predicted": 0.48
    }
  ]
}
```

## 4.7.4 回测预测效果

### `POST /api/v1/datasets/{id}/forecasts/backtest`

请求体：

```json
{
  "model_type": "lstm",
  "granularity": "15min",
  "backtest_start": "2014-12-04T00:00:00+08:00",
  "backtest_end": "2014-12-04T23:45:00+08:00"
}
```

字段说明：

- `model_type`：必填，使用哪个模型进行回测
- `granularity`：回测粒度，当前建模默认 `15min`
- `backtest_start` / `backtest_end`：回测时间范围，服务端基于该范围抽取实际值并生成对比结果

响应 `data`：

```json
{
  "backtest": {
    "dataset_id": 1,
    "model_type": "lstm",
    "backtest_start": "2014-12-04T00:00:00+08:00",
    "backtest_end": "2014-12-04T23:45:00+08:00",
    "granularity": "15min",
    "metrics": {
      "mae": 0.15,
      "rmse": 0.23,
      "smape": 8.7,
      "wape": 10.2
    }
  },
  "predictions": [
    {
      "timestamp": "2014-12-04T00:00:00+08:00",
      "actual": 0.45,
      "predicted": 0.48
    }
  ]
}
```

---

## 4.9 节能建议接口

## 4.9.1 获取建议列表

### `GET /api/v1/datasets/{id}/advices?advice_type=rule`

响应 `data`：

```json
{
  "items": [
    {
      "id": 12,
      "dataset_id": 1,
      "classification_id": 8,
      "advice_type": "rule",
      "summary": "将洗衣与热水器运行时段调整到谷时段",
      "created_at": "2026-03-23T16:05:00+08:00"
    }
  ]
}
```

## 4.9.2 获取建议详情

### `GET /api/v1/advices/{id}`

响应 `data`：

```json
{
  "advice": {
    "id": 12,
    "dataset_id": 1,
    "classification_id": 8,
    "advice_type": "rule",
    "summary": "将洗衣与热水器运行时段调整到谷时段",
    "content_path": "./outputs/advices/advice_12.json",
    "created_at": "2026-03-23T16:05:00+08:00"
  },
  "content": {
    "items": [
      {
        "reason": "峰时占比 46%，高于建议阈值 40%",
        "action": "将洗衣、充电等任务改到谷时段执行"
      }
    ]
  }
}
```

## 4.9.3 重新生成规则建议

### `POST /api/v1/datasets/{id}/advices/generate`

请求体：

```json
{
  "source": "rule",
  "classification_id": 8,
  "force_refresh": true
}
```

---

## 4.10 智能问答接口

## 4.10.1 创建会话

### `POST /api/v1/chat/sessions`

请求体：

```json
{
  "dataset_id": 1,
  "title": "House 1 节能建议问答"
}
```

## 4.10.2 获取会话列表

### `GET /api/v1/chat/sessions?page=1&page_size=20&dataset_id=1`

## 4.10.3 删除会话

### `DELETE /api/v1/chat/sessions/{id}`

说明：

- 会话删除后，关联的所有消息一并级联删除

## 4.10.4 获取会话消息列表

### `GET /api/v1/chat/sessions/{id}/messages?page=1&page_size=50`

## 4.10.5 发起提问

### `POST /api/v1/agent/ask`

请求体：

```json
{
  "dataset_id": 1,
  "session_id": 3,
  "question": "为什么我家夜间用电这么高？",
  "history": [
    {
      "role": "user",
      "content": "给我看一下整体情况"
    },
    {
      "role": "assistant",
      "content": "你的峰时占比为 46%，夜间负荷偏高。"
    }
  ]
}
```

服务端上下文组装约定：

- 前端只传 `dataset_id`、`session_id`、`question`、`history`
- Go 主服务在调用智能体前，自动组装本次问答上下文
- Go 主服务通过 HTTP/JSON 调用 Python Robyn 后端中的 LangChain 智能体路由
- 问答上下文至少包含：
  - 统计分析结果
  - 最近历史窗口的实际用电摘要（窗口长度由 `model_history_window_config` 决定）
- 最新分类结果（`model_type`、`predicted_label`、`confidence`、`explanation`）
- 指定未来时间范围的预测结果摘要（结构固定为 `ForecastSummary`）
- 已生成的规则建议
- 智能体提示词模板默认读取 `energy_advice_prompt_template`
- 智能体提示词语义应明确为：
  - “这是居民过去{{history_days}}天的实际用电情况、未来一段时间的预测用电情况，以及居民用电行为分类。请给出具体、可执行、可解释的节能建议，并引用关键指标或时段作为依据。”

第一阶段智能体能力范围：

- 数据概览问答
- 分类结果解释
- 预测结果解读
- 规则建议重述
- 行动清单生成
- 会话记忆
- 降级问答

第一阶段固定函数调用范围：

- 查询数据集摘要
- 查询最新分类结果
- 查询最新预测摘要
- 查询规则建议
- 读取会话历史
- 读取系统配置

`citations` 约定：

- 固定使用 `CitationItem[]`
- 表示回答所依据的指标、分类结论、预测摘要或风险标签
- 用于前端展示“回答依据”，不是外部链接列表

成功响应 `data`：

```json
{
  "session_id": 3,
  "answer": "你的夜间用电明显高于白天，主要负荷集中在 18:00 以后，建议优先排查晚间持续运行设备。",
  "citations": [
    {
      "key": "peak_ratio",
      "label": "峰时占比",
      "value": 0.46
    },
    {
      "key": "predicted_label",
      "label": "行为类型",
      "value": "day_low_night_high"
    },
    {
      "key": "forecast_peak_period",
      "label": "预测高负荷时段",
      "value": "2014-12-02 19:00-22:00"
    },
    {
      "key": "risk_flags",
      "label": "预测风险标签",
      "value": ["evening_peak_risk", "night_load_risk"]
    }
  ],
  "actions": [
    "检查夜间持续运行设备",
    "将热水器改为定时运行"
  ],
  "degraded": false,
  "error_reason": null,
  "created_at": "2026-03-23T16:10:00+08:00"
}
```

LLM 降级响应 `data`：

```json
{
  "session_id": 3,
  "answer": "智能问答暂时不可用，以下为基于规则的建议。",
  "citations": [
    {
      "key": "predicted_label",
      "label": "行为类型",
      "value": "day_low_night_high"
    }
  ],
  "actions": [
    "优先检查晚间持续运行的设备",
    "将可延后任务尽量安排到更低负荷时段"
  ],
  "degraded": true,
  "error_reason": "LLM_TIMEOUT",
  "created_at": "2026-03-23T16:10:00+08:00"
}
```

---

## 4.11 报告接口

## 4.11.1 创建导出任务

### `POST /api/v1/datasets/{id}/reports/export`

请求体：

```json
{
  "report_type": "excel"
}
```

返回：

- 推荐 `202 Accepted`

```json
{
  "code": "ACCEPTED",
  "message": "导出任务已受理",
  "data": {
    "report": {
      "id": 6,
      "dataset_id": 1,
      "report_type": "excel",
      "file_path": "./outputs/reports/report_6.xlsx",
      "file_size": 0,
      "created_at": "2026-03-23T16:15:00+08:00"
    }
  },
  "request_id": "req_xxx",
  "timestamp": "2026-03-23T16:15:00+08:00"
}
```

## 4.11.2 获取报告列表

### `GET /api/v1/datasets/{id}/reports`

## 4.11.3 下载报告

### `GET /api/v1/reports/{id}/download`

说明：

- 该接口直接返回文件流
- 成功时不包统一 JSON

---

## 5. Go 主服务 ↔ Python Robyn 后端 API

说明：

- 这部分接口仅供内部服务调用
- Go 主服务负责组织数据、调用 Python 后端、存储结果
- Python Robyn 后端统一承载 `model` 与 `agent` 两组能力
- Go 侧可继续保留 `modelclient` 与 `agentclient` 两个逻辑客户端，但它们可以指向同一个服务地址

基准前缀：

```text
/internal/model/v1
```

## 5.1 健康检查

### `GET /internal/model/v1/health`

响应：

```json
{
  "status": "up",
  "service": "python-robyn-backend",
  "model_loaded": true
}
```

## 5.2 获取模型信息

### `GET /internal/model/v1/model/info`

响应：

```json
{
  "service_version": "v1",
  "supported_models": [
    "tcn",
    "lstm"
  ],
  "reserved_models": [
    "transformer"
  ],
  "classification": {
    "labels": [
      "day_high_night_low",
      "day_low_night_high",
      "all_day_high",
      "all_day_low"
    ],
    "label_definitions": [
      {
        "key": "day_high_night_low",
        "display_name": "白天高晚上低型"
      },
      {
        "key": "day_low_night_high",
        "display_name": "白天低晚上高型"
      },
      {
        "key": "all_day_high",
        "display_name": "全天高负载型"
      },
      {
        "key": "all_day_low",
        "display_name": "全天低负载型"
      }
    ],
    "input_spec": {
      "granularity": "15min",
      "unit": "kwh",
      "history_window": {
        "unit": "day",
        "value": 1,
        "config_key": "model_history_window_config.classification_days",
        "configurable": true
      },
      "min_history_length": 96,
      "feature_names": [
        "aggregate",
        "active_appliance_count",
        "burst_event_count"
      ]
    }
  },
  "forecasting": {
    "request_mode": "time_range",
    "supported_granularities": ["15min"],
    "summary_schema": "ForecastSummary",
    "raw_output_schema": "predictions[96]",
    "input_spec": {
      "granularity": "15min",
      "unit": "kwh",
      "history_window": {
        "unit": "day",
        "value": 3,
        "config_key": "model_history_window_config.forecast_history_days",
        "configurable": true
      },
      "min_history_length": 288,
      "target_length": 96,
      "feature_names": [
        "aggregate",
        "active_appliance_count",
        "burst_event_count"
      ]
    }
  }
}
```

## 5.3 分类推理

### `POST /internal/model/v1/predict`

请求体：

```json
{
  "model_type": "tcn",
  "dataset_id": 1,
  "window": {
    "start": "2014-01-01T00:00:00+08:00",
    "end": "2014-01-01T23:45:00+08:00"
  },
  "series": [
    {
      "timestamp": "2014-01-01T00:00:00+08:00",
      "aggregate": 0.42,
      "active_appliance_count": 2,
      "burst_event_count": 0
    }
  ],
  "metadata": {
    "granularity": "15min",
    "unit": "kwh"
  }
}
```

响应：

```json
{
  "model_type": "tcn",
  "sample_id": "1_2014-01-01",
  "house_id": "1",
  "date": "2014-01-01",
  "predicted_label": "day_low_night_high",
  "confidence": 0.83,
  "prob_day_high_night_low": 0.06,
  "prob_day_low_night_high": 0.83,
  "prob_all_day_high": 0.07,
  "prob_all_day_low": 0.04,
  "runtime_device": "cpu",
  "runtime_loss": "CrossEntropyLoss"
}
```

说明：

- Python Robyn 后端返回原始概率字段 `prob_*`
- Go 主服务负责将其收口为公共 API 中的 `probabilities` JSON 结构
- 公共 API 中的 `explanation` 由 Go 主服务根据窗口统计量和分类结果生成，不要求模型直接返回

### 错误响应

```json
{
  "code": "MODEL_NOT_LOADED",
  "message": "指定模型尚未加载"
}
```

## 5.4 预测推理

### `POST /internal/model/v1/forecast`

请求体：

```json
{
  "model_type": "lstm",
  "dataset_id": 1,
  "forecast_start": "2014-12-04T00:00:00+08:00",
  "forecast_end": "2014-12-04T23:45:00+08:00",
  "granularity": "15min",
  "series": [
    {
      "timestamp": "2014-12-01T00:00:00+08:00",
      "aggregate": 0.42,
      "active_appliance_count": 2,
      "burst_event_count": 0
    }
  ],
  "metadata": {
    "unit": "kwh"
  }
}
```

字段说明：

- `model_type`：必填，使用哪个模型
- `forecast_start` / `forecast_end`：预测时间范围，预测步数由时间范围和粒度共同决定
- `series`：历史数据序列，长度需满足当前模型历史窗口配置要求
- `granularity`：预测粒度

响应：

```json
{
  "model_type": "lstm",
  "sample_id": "1_2014-12-01_2014-12-04",
  "house_id": "1",
  "input_start": "2014-12-01T00:00:00+08:00",
  "input_end": "2014-12-03T23:45:00+08:00",
  "predictions": [0.48, 0.47, 0.45]
}
```

说明：

- Python Robyn 后端当前返回原始 `predictions` 数组，不直接返回 `ForecastSummary`
- Go 主服务负责根据请求中的 `forecast_start` / `forecast_end` / `granularity` 为预测点补齐时间戳
- Go 主服务负责生成公共 API 里的 `summary`、`series` 和 `detail_path`

### 错误响应

```json
{
  "code": "INSUFFICIENT_HISTORY",
  "message": "历史数据不足，至少需要满足当前模型配置的历史窗口长度"
}
```

## 5.5 回测推理

### `POST /internal/model/v1/backtest`

请求体：

```json
{
  "model_type": "lstm",
  "dataset_id": 1,
  "backtest_start": "2014-12-04T00:00:00+08:00",
  "backtest_end": "2014-12-04T23:45:00+08:00",
  "granularity": "15min",
  "series": [
    {
      "timestamp": "2014-12-01T00:00:00+08:00",
      "aggregate": 0.42,
      "active_appliance_count": 2,
      "burst_event_count": 0
    },
    {
      "timestamp": "2014-12-04T00:00:00+08:00",
      "aggregate": 0.45,
      "active_appliance_count": 3,
      "burst_event_count": 1
    }
  ],
  "metadata": {
    "unit": "kwh"
  }
}
```

字段说明：

- `model_type`：必填，使用哪个模型
- `backtest_start` / `backtest_end`：回测时间范围
- `series`：同时包含历史上下文与回测区间实际值，供模型生成预测并计算指标
- `granularity`：回测粒度

响应：

```json
{
  "model_type": "lstm",
  "backtest_start": "2014-12-04T00:00:00+08:00",
  "backtest_end": "2014-12-04T23:45:00+08:00",
  "granularity": "15min",
  "predictions": [
    {
      "timestamp": "2014-11-24T00:00:00+08:00",
      "actual": 0.45,
      "predicted": 0.48
    }
  ],
  "metrics": {
    "mae": 0.15,
    "rmse": 0.23,
    "smape": 8.7,
    "wape": 10.2
  }
}
```

## 5.6 智能体健康检查

### `GET /internal/agent/v1/health`

响应：

```json
{
  "status": "up",
  "service": "python-robyn-backend",
  "agent_ready": true
}
```

## 5.7 智能体问答

### `POST /internal/agent/v1/ask`

请求体：

```json
{
  "dataset_id": 1,
  "session_id": 3,
  "question": "为什么我家夜间用电这么高？",
  "history": [
    {
      "role": "user",
      "content": "给我看一下整体情况"
    }
  ],
  "context": {
    "analysis_summary": {
      "peak_ratio": 0.46
    },
    "classification_result": {
      "predicted_label": "day_low_night_high",
      "confidence": 0.83
    },
    "forecast_summary": {
      "peak_period": "2014-12-02 19:00-22:00"
    },
    "rule_advices": [
      {
        "summary": "将热水器改为定时运行"
      }
    ]
  }
}
```

响应：

```json
{
  "answer": "你的夜间用电明显高于白天，主要负荷集中在 18:00 以后，建议优先排查晚间持续运行设备。",
  "citations": [
    {
      "key": "peak_ratio",
      "label": "峰时占比",
      "value": 0.46
    },
    {
      "key": "predicted_label",
      "label": "行为类型",
      "value": "day_low_night_high"
    }
  ],
  "actions": [
    "检查夜间持续运行设备",
    "将热水器改为定时运行"
  ]
}
```

说明：

- Go 主服务负责组装 `context` 并控制降级
- Python Robyn 后端中的 LangChain 智能体模块负责生成 `answer/citations/actions`
- 该内部接口不直接负责数据库写入

---

## 6. 前端首批联调接口建议

为了支持前后端并行开发，建议第一批先约定并联调以下接口：

1. `GET /api/v1/health`
2. `GET /api/v1/system/config`
3. `POST /api/v1/datasets/import`
4. `GET /api/v1/datasets`
5. `GET /api/v1/datasets/{id}`
6. `GET /api/v1/datasets/{id}/analysis`
7. `GET /api/v1/datasets/{id}/classifications/latest`
8. `GET /api/v1/datasets/{id}/advices`
9. `POST /api/v1/agent/ask`
10. `GET /api/v1/datasets/{id}/reports`

这 10 个接口足够支撑首版页面：

- 数据集列表页
- 数据集详情页
- 图表展示区
- 建议展示区
- 问答弹窗
- 报告列表区

---

## 7. 业务状态流转

## 7.1 数据集状态流转

```text
uploaded -> processing -> ready
uploaded -> processing -> error
error -> processing -> ready
```

前端展示建议：

- `uploaded`：已接收文件
- `processing`：处理中，显示加载状态
- `ready`：可查看分析与建议
- `error`：显示错误信息与重试入口

## 8. 业务错误码建议

建议在 Go 主服务中统一维护错误码常量：

```text
OK
ACCEPTED
INVALID_REQUEST
UNSUPPORTED_FILE_TYPE
COLUMN_MAPPING_REQUIRED
DATASET_NOT_FOUND
DATASET_NOT_READY
ANALYSIS_NOT_FOUND
CLASSIFICATION_NOT_FOUND
FORECAST_NOT_FOUND
ADVICE_NOT_FOUND
CHAT_SESSION_NOT_FOUND
REPORT_NOT_FOUND
LLM_CONFIG_NOT_FOUND
LLM_TIMEOUT
LLM_UNAVAILABLE
MODEL_SERVICE_UNAVAILABLE
MODEL_NOT_LOADED
TRAINING_TASK_NOT_FOUND
EXPORT_FAILED
BACKTEST_FAILED
INTERNAL_ERROR
```

---

## 9. 字段命名建议

为减少联调摩擦，命名统一遵循以下规则：

- JSON 字段一律使用 `snake_case`
- ID 一律使用整数，不使用字符串 UUID
- 金额、比例、功率、用电量都返回数值类型
- 时间字段统一返回完整带时区字符串
- 文件路径字段统一以 `*_path` 命名

---

## 10. Mock 开发建议

前后端并行开发时，建议顺序如下：

1. 以后端 API 文档为准，先冻结字段名
2. 前端用 mock 数据开发页面
3. Go 主服务先返回静态 JSON 或伪数据
4. 再逐步接数据库与 Python Robyn 后端
5. 最后接 LLM 智能问答

不建议的做法：

- 前端先随意命名字段，再让后端适配
- 同一接口在联调中频繁改字段名
- 把图表计算逻辑大量放到前端

---

## 11. 后续扩展建议

如果后续项目范围扩大，可以在此文档基础上继续补充：

1. OpenAPI 3.0 标准描述文件
2. WebSocket 实时任务进度推送
3. 批量数据集导入接口
4. 模型训练日志流式查看接口
5. 报告导出任务状态轮询接口

当前阶段不建议一开始就设计过重，优先确保首版联调闭环。
