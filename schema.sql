USE `resident`;

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

DROP TABLE IF EXISTS `reports`;
DROP TABLE IF EXISTS `chat_messages`;
DROP TABLE IF EXISTS `chat_sessions`;
DROP TABLE IF EXISTS `detection_results`;
DROP TABLE IF EXISTS `forecast_results`;
DROP TABLE IF EXISTS `classification_results`;
DROP TABLE IF EXISTS `analysis_results`;
DROP TABLE IF EXISTS `datasets`;

CREATE TABLE `datasets` (
    `id`                         BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `name`                       VARCHAR(128)    NOT NULL COMMENT '数据集名称',
    `description`                TEXT            DEFAULT NULL COMMENT '数据集描述',
    `household_id`               VARCHAR(64)     DEFAULT NULL COMMENT '住户标识',
    `source_file_name`           VARCHAR(255)    DEFAULT NULL COMMENT '用户上传的原始文件名',
    `raw_file_path`              VARCHAR(512)    NOT NULL COMMENT '原始上传文件路径',
    `normalized_file_path`       VARCHAR(512)    DEFAULT NULL COMMENT '标准化后的时序文件路径',
    `daily_aggregate_file_path`  VARCHAR(512)    DEFAULT NULL COMMENT '按日聚合后的文件路径',
    `row_count`                  INT UNSIGNED    NOT NULL DEFAULT 0 COMMENT '原始记录数',
    `time_start`                 DATETIME        DEFAULT NULL COMMENT '数据起始时间',
    `time_end`                   DATETIME        DEFAULT NULL COMMENT '数据结束时间',
    `source_granularity_minutes` SMALLINT UNSIGNED DEFAULT NULL COMMENT '检测到的最细时间粒度（分钟）',
    `column_mapping`             JSON            DEFAULT NULL COMMENT '原始列到标准字段的映射',
    `quality_summary`            JSON            DEFAULT NULL COMMENT '缺失率、重复数、粒度校验等质量摘要',
    `quality_report_path`        VARCHAR(512)    DEFAULT NULL COMMENT '质量报告文件路径',
    `status`                     ENUM('uploaded', 'processing', 'ready', 'error')
                                 NOT NULL DEFAULT 'uploaded' COMMENT '处理状态',
    `error_message`              TEXT            DEFAULT NULL COMMENT '处理失败信息',
    `created_at`                 DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updated_at`                 DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP
                                 ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`),
    KEY `idx_datasets_status` (`status`),
    KEY `idx_datasets_household` (`household_id`),
    KEY `idx_datasets_created_at` (`created_at`),
    CONSTRAINT `chk_datasets_time_range`
        CHECK (
            `time_start` IS NULL
            OR `time_end` IS NULL
            OR `time_end` >= `time_start`
        ),
    CONSTRAINT `chk_datasets_granularity`
        CHECK (
            `source_granularity_minutes` IS NULL
            OR `source_granularity_minutes` > 0
        )
) ENGINE=InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = '上传数据集及预处理产物';

CREATE TABLE `analysis_results` (
    `id`             BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `dataset_id`     BIGINT UNSIGNED NOT NULL COMMENT '关联数据集',
    `total_kwh`      DECIMAL(14, 4)  DEFAULT NULL COMMENT '总用电量',
    `daily_avg_kwh`  DECIMAL(12, 4)  DEFAULT NULL COMMENT '日均用电量',
    `max_load_w`     DECIMAL(14, 4)  DEFAULT NULL COMMENT '最大负荷功率',
    `max_load_time`  DATETIME        DEFAULT NULL COMMENT '最大负荷对应时间',
    `min_load_w`     DECIMAL(14, 4)  DEFAULT NULL COMMENT '最小负荷功率',
    `min_load_time`  DATETIME        DEFAULT NULL COMMENT '最小负荷对应时间',
    `peak_kwh`       DECIMAL(14, 4)  DEFAULT NULL COMMENT '峰时总电量',
    `valley_kwh`     DECIMAL(14, 4)  DEFAULT NULL COMMENT '谷时总电量',
    `peak_ratio`     DECIMAL(6, 4)   DEFAULT NULL COMMENT '峰时占比',
    `valley_ratio`   DECIMAL(6, 4)   DEFAULT NULL COMMENT '谷时占比',
    `summary_json`   JSON            DEFAULT NULL COMMENT '前端概览摘要',
    `detail_path`    VARCHAR(512)    DEFAULT NULL COMMENT '详细分析结果文件路径',
    `created_at`     DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_analysis_results_dataset` (`dataset_id`),
    CONSTRAINT `fk_analysis_results_dataset`
        FOREIGN KEY (`dataset_id`) REFERENCES `datasets` (`id`) ON DELETE CASCADE,
    CONSTRAINT `chk_analysis_ratio_peak`
        CHECK (`peak_ratio` IS NULL OR (`peak_ratio` >= 0 AND `peak_ratio` <= 1)),
    CONSTRAINT `chk_analysis_ratio_valley`
        CHECK (`valley_ratio` IS NULL OR (`valley_ratio` >= 0 AND `valley_ratio` <= 1))
) ENGINE=InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = '数据统计分析结果';

CREATE TABLE `classification_results` (
    `id`              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `dataset_id`      BIGINT UNSIGNED NOT NULL COMMENT '关联数据集',
    `forecast_id`     BIGINT UNSIGNED DEFAULT NULL COMMENT '若为未来窗口分类，则关联预测结果',
    `schema_version`  VARCHAR(16)     NOT NULL DEFAULT 'v1' COMMENT '结果协议版本',
    `model_type`      ENUM('xgboost') NOT NULL DEFAULT 'xgboost' COMMENT '分类模型类型',
    `window_role`     ENUM('current', 'future') NOT NULL DEFAULT 'current' COMMENT '窗口角色',
    `predicted_label` VARCHAR(32)     NOT NULL COMMENT '分类标签',
    `confidence`      DECIMAL(6, 4)   DEFAULT NULL COMMENT '最高置信度',
    `probabilities`   JSON            DEFAULT NULL COMMENT '各类别概率分布',
    `explanation`     TEXT            DEFAULT NULL COMMENT '分类解释',
    `sample_id`       VARCHAR(128)    DEFAULT NULL COMMENT '样本标识',
    `runtime_library` VARCHAR(64)     DEFAULT NULL COMMENT '运行时库信息',
    `window_start`    DATETIME        DEFAULT NULL COMMENT '窗口开始日期',
    `window_end`      DATETIME        DEFAULT NULL COMMENT '窗口结束日期',
    `created_at`      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`),
    KEY `idx_classification_dataset` (`dataset_id`),
    KEY `idx_classification_forecast` (`forecast_id`),
    KEY `idx_classification_role` (`dataset_id`, `window_role`, `created_at`),
    CONSTRAINT `fk_classification_dataset`
        FOREIGN KEY (`dataset_id`) REFERENCES `datasets` (`id`) ON DELETE CASCADE,
    CONSTRAINT `chk_classification_confidence`
        CHECK (`confidence` IS NULL OR (`confidence` >= 0 AND `confidence` <= 1)),
    CONSTRAINT `chk_classification_window_range`
        CHECK (
            `window_start` IS NULL
            OR `window_end` IS NULL
            OR `window_end` >= `window_start`
        )
) ENGINE=InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = '行为分类结果';

CREATE TABLE `forecast_results` (
    `id`                  BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `dataset_id`          BIGINT UNSIGNED NOT NULL COMMENT '关联数据集',
    `schema_version`      VARCHAR(16)     NOT NULL DEFAULT 'v1' COMMENT '结果协议版本',
    `model_type`          ENUM('lstm')    NOT NULL DEFAULT 'lstm' COMMENT '预测模型类型',
    `history_days`        TINYINT UNSIGNED NOT NULL DEFAULT 30 COMMENT '历史窗口天数',
    `forecast_horizon_days` TINYINT UNSIGNED NOT NULL DEFAULT 7 COMMENT '预测天数',
    `forecast_start`      DATETIME        NOT NULL COMMENT '预测开始日期',
    `forecast_end`        DATETIME        NOT NULL COMMENT '预测结束日期',
    `granularity`         ENUM('daily')   NOT NULL DEFAULT 'daily' COMMENT '预测结果粒度',
    `summary`             JSON            DEFAULT NULL COMMENT '预测摘要',
    `detail_path`         VARCHAR(512)    NOT NULL COMMENT '预测明细文件路径',
    `metrics`             JSON            DEFAULT NULL COMMENT '预测评估指标',
    `created_at`          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`),
    KEY `idx_forecast_dataset` (`dataset_id`),
    KEY `idx_forecast_dataset_created` (`dataset_id`, `created_at`),
    CONSTRAINT `fk_forecast_dataset`
        FOREIGN KEY (`dataset_id`) REFERENCES `datasets` (`id`) ON DELETE CASCADE,
    CONSTRAINT `chk_forecast_range`
        CHECK (`forecast_end` >= `forecast_start`)
) ENGINE=InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = '未来 7 天日级预测结果';

ALTER TABLE `classification_results`
    ADD CONSTRAINT `fk_classification_forecast`
        FOREIGN KEY (`forecast_id`) REFERENCES `forecast_results` (`id`) ON DELETE SET NULL;

CREATE TABLE `detection_results` (
    `id`                  BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `dataset_id`          BIGINT UNSIGNED NOT NULL COMMENT '关联数据集',
    `forecast_id`         BIGINT UNSIGNED DEFAULT NULL COMMENT '若为未来窗口异常，则关联预测结果',
    `schema_version`      VARCHAR(16)     NOT NULL DEFAULT 'v1' COMMENT '结果协议版本',
    `model_type`          ENUM('iforest_rules') NOT NULL DEFAULT 'iforest_rules' COMMENT '异常检测模型',
    `window_role`         ENUM('current', 'future') NOT NULL DEFAULT 'current' COMMENT '窗口角色',
    `window_start`        DATETIME        DEFAULT NULL COMMENT '窗口开始日期',
    `window_end`          DATETIME        DEFAULT NULL COMMENT '窗口结束日期',
    `is_anomaly`          TINYINT(1)      NOT NULL DEFAULT 0 COMMENT '是否异常',
    `anomaly_score`       DECIMAL(14, 6)  DEFAULT NULL COMMENT '异常分数',
    `severity`            ENUM('low', 'medium', 'high') DEFAULT NULL COMMENT '异常等级',
    `reasons`             JSON            DEFAULT NULL COMMENT '异常原因列表',
    `feature_summary`     JSON            DEFAULT NULL COMMENT '触发异常的关键特征摘要',
    `classification_hint` VARCHAR(64)     DEFAULT NULL COMMENT '辅助分类标签',
    `created_at`          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`),
    KEY `idx_detection_dataset` (`dataset_id`),
    KEY `idx_detection_forecast` (`forecast_id`),
    KEY `idx_detection_role` (`dataset_id`, `window_role`, `created_at`),
    CONSTRAINT `fk_detection_dataset`
        FOREIGN KEY (`dataset_id`) REFERENCES `datasets` (`id`) ON DELETE CASCADE,
    CONSTRAINT `fk_detection_forecast`
        FOREIGN KEY (`forecast_id`) REFERENCES `forecast_results` (`id`) ON DELETE SET NULL,
    CONSTRAINT `chk_detection_window_range`
        CHECK (
            `window_start` IS NULL
            OR `window_end` IS NULL
            OR `window_end` >= `window_start`
        )
) ENGINE=InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = '异常检测结果';

CREATE TABLE `chat_sessions` (
    `id`          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `dataset_id`  BIGINT UNSIGNED DEFAULT NULL COMMENT '关联数据集，允许为空',
    `title`       VARCHAR(128)    DEFAULT NULL COMMENT '会话标题',
    `created_at`  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updated_at`  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP
                                  ON UPDATE CURRENT_TIMESTAMP COMMENT '最近更新时间',
    PRIMARY KEY (`id`),
    KEY `idx_chat_sessions_dataset` (`dataset_id`),
    KEY `idx_chat_sessions_updated` (`updated_at`),
    CONSTRAINT `fk_chat_sessions_dataset`
        FOREIGN KEY (`dataset_id`) REFERENCES `datasets` (`id`) ON DELETE SET NULL
) ENGINE=InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = '问答会话';

CREATE TABLE `chat_messages` (
    `id`                BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `session_id`        BIGINT UNSIGNED NOT NULL COMMENT '关联会话',
    `role`              ENUM('user', 'assistant', 'system') NOT NULL COMMENT '消息角色',
    `content`           TEXT            DEFAULT NULL COMMENT '消息正文',
    `assistant_payload` JSON            DEFAULT NULL COMMENT '智能体结构化回复载荷',
    `content_path`      VARCHAR(512)    DEFAULT NULL COMMENT '长文本落盘路径',
    `model_name`        VARCHAR(128)    DEFAULT NULL COMMENT '使用的模型名',
    `tokens_used`       INT UNSIGNED    DEFAULT NULL COMMENT 'token 消耗',
    `created_at`        DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`),
    KEY `idx_chat_messages_session` (`session_id`, `created_at`),
    CONSTRAINT `fk_chat_messages_session`
        FOREIGN KEY (`session_id`) REFERENCES `chat_sessions` (`id`) ON DELETE CASCADE,
    CONSTRAINT `chk_chat_messages_content`
        CHECK (`content` IS NOT NULL OR `content_path` IS NOT NULL OR `assistant_payload` IS NOT NULL)
) ENGINE=InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = '问答消息';

CREATE TABLE `reports` (
    `id`          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `dataset_id`  BIGINT UNSIGNED NOT NULL COMMENT '关联数据集',
    `report_type` ENUM('pdf')     NOT NULL DEFAULT 'pdf' COMMENT '当前仅保留 PDF 报告',
    `file_path`   VARCHAR(512)    NOT NULL COMMENT '报告文件路径',
    `file_size`   BIGINT UNSIGNED NOT NULL DEFAULT 0 COMMENT '报告文件大小（字节）',
    `created_at`  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`),
    KEY `idx_reports_dataset` (`dataset_id`),
    CONSTRAINT `fk_reports_dataset`
        FOREIGN KEY (`dataset_id`) REFERENCES `datasets` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = '导出报告记录';

SET FOREIGN_KEY_CHECKS = 1;
