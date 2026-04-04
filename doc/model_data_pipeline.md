# 模型数据流水线设计

> 文档版本：v2.2  
> 最后更新：2026年04月03日  
> 适用范围：`models/` 目录下的数据预处理、规则标签生成、分类训练、预测训练

---

## 1. 目标

本项目将 `models/data/raw/` 下支持的数据源统一加工为三层产物：

1. `15min` 基础时序数据
2. 分类任务数据集
3. 预测任务数据集

其中：

- 分类任务使用 `规则标签 + TCN`
- 预测任务使用 `Seq2Seq LSTM + Transformer`

说明：

- 本文档约束的是 `models/` 目录下的训练与离线建模数据方案
- 当前基础预处理支持多数据源统一导入，训练阶段仍以 REFIT 风格字段和 15 分钟户级总负荷为统一口径
- 面向最终用户的导入协议由主服务定义，不要求用户上传文件固定只有 `Appliance1~Appliance9`

---

## 2. 原始数据约束

当前基础预处理默认会扫描 `models/data/raw/` 下的以下目录：

- `refit/`
- `ukdale/`
- `slovakia_households_1000/`
- `opensynth_tudelft_electricity_consumption_1_0/`

其中 REFIT 已清洗 CSV 存放在 `models/data/raw/refit/`，标准字段：

- `Time`
- `Unix`
- `Aggregate`
- `Appliance1` ~ `Appliance9`
- `Issues`

处理约束：

- 原始文件只读，不允许手工修改
- `Issues = 1` 的记录默认剔除
- 建模统一使用 `15min` 粒度
- 跨家庭训练时，不直接使用原始 `Appliance1-9` 作为统一语义特征
- 面向用户导入时，可先识别任意数量的电器子表列，再映射到统一辅助特征
- 非 REFIT 数据源若不存在电器级子表，则 `active_appliance_count` 与 `burst_event_count` 默认置为 `0`

原因：

- 不同家庭的 `Appliance1-9` 含义不一致
- 但可以从这些列中提取统一统计特征

---

## 3. 基础预处理

### 3.1 时间粒度

- 将原始约 8 秒级数据统一重采样到 `15min`
- 每天固定得到 `96` 个时间步

### 3.2 聚合规则

- `Aggregate`：对 15 分钟窗口取均值
- 已识别电器列集合：对 15 分钟窗口取均值

### 3.3 缺失与异常处理

- `Issues = 1` 的原始记录在重采样前剔除
- 缺失的 `15min` 时间步必须按真实缺失槽位统计，不允许用重采样后的行数代替
- 仅允许对短缺失段做时间插值；连续缺失超过 `8` 个 `15min` 槽位的长缺失段保留为空，并在后续按天过滤
- 每个自然日若插值点占比超过 `10%`，则不进入分类/预测训练样本
- `aggregate` 在家庭内按 `P99.9` 做上截尾，削弱极端尖峰对预测损失和验证波动的放大效应
- 基础时序需额外保留质量标记字段：
  - `is_observed_point`
  - `is_imputed_point`
  - `is_unresolved_point`
  - `is_clipped_point`

### 3.4 辅助特征

在每个 `15min` 时间步上额外生成：

1. `active_appliance_count`
   - 定义：已识别电器列中功率大于 `10W` 的电器个数

2. `burst_event_count`
   - 定义：某个已识别电器列相邻采样点功率增量大于 `300W`
   - 且高功率持续时间不超过 `2` 个 `15min` 窗口
   - 每个 `15min` 内统计突发事件次数

### 3.5 基础输出字段

基础时序数据每行字段建议固定为：

- `house_id`
- `source_dataset`
- `timestamp`
- `date`
- `slot_index`
- `aggregate`
- `active_appliance_count`
- `burst_event_count`
- `is_weekend`
- `is_observed_point`
- `is_imputed_point`
- `is_unresolved_point`
- `is_clipped_point`

其中：

- `slot_index` 取值范围为 `0-95`
- 当前 `models/` 训练链路统一保留为 `W`
- 若后端或前端需要展示 `kWh`，应在服务侧按 `15min` 粒度再换算，避免训练与推理口径漂移

---

## 4. 分类任务数据集

### 4.1 样本定义

- 以“天”为单位构造样本
- 每个样本由同一天的 `96` 个时间步组成

### 4.2 规则标签输入

每个日样本的规则判定输入固定为三条日曲线：

- `aggregate[96]`
- `active_appliance_count[96]`
- `burst_event_count[96]`

### 4.3 规则标签流程

1. 对所有家庭的日样本构造日曲线特征
2. 计算：
   - `day_mean`
   - `night_mean`
   - `full_mean`
3. 由全体日样本 `full_mean` 统计得到：
   - `high_threshold = P75`
   - `low_threshold = P25`
4. 设定固定比例阈值：
   - `ratio_threshold = 1.2`
5. 根据规则直接生成 `label_name`

规则标签固定为：

- `day_high_night_low`
- `day_low_night_high`
- `all_day_high`
- `all_day_low`

时间段固定为：

- 白天：`08:00-18:00`
- 夜间：`18:00-08:00`

### 4.4 分类数据集字段

建议输出两份文件：

1. `classification_day_features.*`
   - 保存日曲线特征与元信息

2. `classification_day_labels.*`
   - 在上一步基础上新增规则统计量和标签列

每条样本至少包含：

- `sample_id`
- `house_id`
- `date`
- `aggregate_000` ~ `aggregate_095`
- `active_count_000` ~ `active_count_095`
- `burst_count_000` ~ `burst_count_095`
- `day_mean`
- `night_mean`
- `full_mean`
- `label_name`

### 4.5 TCN 分类输入输出

- 输入形状：`[batch, 96, 3]`
- 通道顺序固定：
  - `aggregate`
  - `active_appliance_count`
  - `burst_event_count`
- 输出：
  - `num_classes` 维 softmax
  - `predicted_label`

当前单样本推理原始输出字段固定为：

- `sample_id`
- `house_id`
- `date`
- `predicted_label`
- `confidence`
- `prob_day_high_night_low`
- `prob_day_low_night_high`
- `prob_all_day_high`
- `prob_all_day_low`

说明：

- `runtime_device`、`runtime_loss` 属于运行时调试信息，可选保留
- 面向前端与数据库的 `probabilities`、`explanation` 建议由 Go 主服务二次整理

---

## 5. 预测任务数据集

### 5.1 样本定义

- 输入窗口：最近 `3` 天
- 输出窗口：下一天

因此：

- 输入长度：`3 * 96 = 288`
- 输出长度：`96`

### 5.2 输入特征

预测任务支持以下候选输入特征：

- `aggregate`
- `active_appliance_count`
- `burst_event_count`

当前默认先使用单通道 baseline：

- `aggregate`

说明：

- 若要做特征对照实验，可再切换到：
  - `aggregate + active_appliance_count`
  - `aggregate + active_appliance_count + burst_event_count`
- 对 Transformer 类预测模型，允许额外加入稳定时间特征：
  - `slot_sin`
  - `slot_cos`
  - `weekday_sin`
  - `weekday_cos`
- `feature_names` 的第一个特征必须是 `aggregate`

### 5.3 预测目标

第一版目标固定为：

- 预测下一天的 `aggregate[96]`

即：

- 输入：`[288, C]`
- 输出：`[96]`

其中：

- `C = len(feature_names)`

### 5.4 预测数据集字段

建议按样本维度保存为扁平化结构，至少包含：

- `sample_id`
- `house_id`
- `input_start`
- `input_end`
- `target_start`
- `target_end`
- `input_imputed_points`
- `input_imputed_ratio`
- `input_clipped_points`
- `target_imputed_points`
- `target_imputed_ratio`
- `target_clipped_points`
- `x_aggregate_000` ~ `x_aggregate_287`
- `x_active_count_000` ~ `x_active_count_287`
- `x_burst_count_000` ~ `x_burst_count_287`
- `y_aggregate_000` ~ `y_aggregate_095`

### 5.5 模型输入输出

#### LSTM

- 输入：`[batch, 288, C]`
- 输出：`[batch, 96]`

当前默认训练约定：

- `aggregate` 不再使用全局均值方差，而是按单个样本的输入窗口 `288` 点做样本级归一化
- `y_aggregate_000 ~ y_aggregate_095` 的监督目标复用同一组输入窗口统计量进行归一化，并在评估/推理后还原回真实 `W`
- `active_appliance_count` 与 `burst_event_count` 继续使用训练集统计量做标准化
- LSTM 默认损失函数改为 `HuberLoss(delta=1.0)`；如需对比实验，可切换回 `MSELoss`

当前单样本推理原始输出字段固定为：

- `sample_id`
- `house_id`
- `input_start`
- `input_end`
- `predictions`

说明：

- `predictions` 固定为长度 `96` 的浮点数组
- 预测摘要 `ForecastSummary` 不是模型原始输出，应由 Go 主服务根据预测序列生成

#### Transformer

- 输入：`[batch, 288, 3]`
- 输出：`[batch, 96]`

说明：

- Transformer 采用面向连续值时序预测的结构
- 不再使用 SASRec 风格的推荐式离散序列建模

---

## 6. 划分与评估建议

预测任务默认采用 `by_house` 划分。

原因：

- 本项目真实部署场景是“用 REFIT 已见家庭训练，再服务于未见过的新家庭”
- 因此核心目标不是同家庭未来预测，而是跨家庭泛化
- 对 `3day -> 1day` 滑动窗口样本使用 `random` 切分时，相邻样本会共享大量重叠日期，评估结果容易偏乐观

建议保留两套实验：

1. `by_house`
   - 默认与主报告口径
   - 用于评估模型对新家庭的泛化能力

2. `random`
   - 仅用于开发阶段快速冒烟或确认模型是否能学习基本模式
   - 不应作为最终效果结论

所有实验都应固定随机种子，并记录：

- 切分方案
- 规则阈值版本
- 模型超参数

当前预测任务评估指标统一为：

- `MAE`
- `RMSE`
- `SMAPE`
- `WAPE`

---

## 7. 实现顺序

建议按以下顺序实现：

1. 生成 `15min` 基础时序数据
2. 生成分类日级特征
3. 计算 `day_mean / night_mean / full_mean`
4. 根据规则直接生成分类标签
5. 生成分类训练集
6. 生成 `3day -> 1day` 预测训练集
7. 分别接入 `TCN`、`LSTM`、`Transformer`

---

## 8. 当前冻结规则

当前先冻结以下默认值：

- 重采样粒度：`15min`
- `active_appliance_count` 阈值：`10W`
- `burst_event_count` 阈值：相邻功率增量 `> 300W`
- 突发持续上限：`2` 个 `15min` 窗口
- 分类样本单位：`1天`
- 白天时段：`08:00-18:00`
- 夜间时段：`18:00-08:00`
- `high_threshold`：全体日样本 `full_mean` 的 `P75`
- `low_threshold`：全体日样本 `full_mean` 的 `P25`
- `ratio_threshold`：`1.2`
- 预测窗口：`3天 -> 1天`

如后续实验要改阈值，必须同步更新本文件与训练配置。
