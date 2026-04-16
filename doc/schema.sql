-- ============================================================
-- 居民用电分析与节能建议系统 — MySQL 8.0 建表脚本
-- 数据库：resident
-- 说明：
-- 1. 本脚本只负责数据库结构与基础系统配置初始化
-- 2. 采用“元数据入库、大对象落文件”的设计
-- ============================================================

USE `resident`;

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- -----------------------------------------------------------
-- 1. system_config — 系统配置（KV 存储）
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS `system_config` (
    `config_key`   VARCHAR(64)  NOT NULL COMMENT '配置键',
    `config_value` TEXT         NOT NULL COMMENT '配置值',
    `description`  VARCHAR(255) DEFAULT NULL COMMENT '配置说明',
    `updated_at`   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP COMMENT '最后更新时间',
    PRIMARY KEY (`config_key`)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='系统配置表';

INSERT INTO `system_config` (`config_key`, `config_value`, `description`)
VALUES
    ('peak_valley_config', '{"peak":["07:00-11:00","18:00-23:00"],"valley":["23:00-07:00"]}', '峰谷时段配置（JSON 格式）'),
    ('model_history_window_config', '{"classification_days":1,"forecast_history_days":3}', '模型历史窗口配置（分类/预测）'),
    ('energy_advice_prompt_template', '这是居民过去{{history_days}}天的实际用电情况、未来一段时间的预测用电情况，以及居民用电行为分类。请基于统计分析结果、历史用电摘要、未来预测摘要和分类结果，给出具体、可执行、可解释的节能建议，并指出关键依据。', '节能建议智能体提示词模板'),
    ('data_upload_dir', './uploads/datasets', '数据集上传目录'),
    ('report_output_dir', './outputs/reports', '报告输出目录')
ON DUPLICATE KEY UPDATE
    `config_value` = VALUES(`config_value`),
    `description` = VALUES(`description`);

-- -----------------------------------------------------------
-- 2. datasets — 数据集
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS `datasets` (
    `id`                  BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `name`                VARCHAR(128)    NOT NULL COMMENT '数据集名称',
    `description`         TEXT            DEFAULT NULL COMMENT '描述',
    `raw_file_path`       VARCHAR(512)    NOT NULL COMMENT '原始上传文件路径',
    `processed_file_path` VARCHAR(512)    DEFAULT NULL COMMENT '清洗聚合后文件路径',
    `household_id`        VARCHAR(64)     DEFAULT NULL COMMENT '家庭标识',
    `row_count`           INT UNSIGNED    NOT NULL DEFAULT 0 COMMENT '数据行数',
    `time_start`          DATETIME        DEFAULT NULL COMMENT '数据时间范围起点',
    `time_end`            DATETIME        DEFAULT NULL COMMENT '数据时间范围终点',
    `feature_cols`        JSON            DEFAULT NULL COMMENT '原始特征列名列表',
    `column_mapping`      JSON            DEFAULT NULL COMMENT '原始列名到标准特征的映射',
    `status`              ENUM('uploaded','processing','ready','error')
                      NOT NULL DEFAULT 'uploaded' COMMENT '处理状态',
    `quality_report_path` VARCHAR(512)    DEFAULT NULL COMMENT '数据质量报告文件路径',
    `error_message`       TEXT            DEFAULT NULL COMMENT '处理失败时的错误信息',
    `created_at`          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '上传时间',
    `updated_at`          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`id`),
    KEY `idx_datasets_status` (`status`),
    KEY `idx_datasets_created` (`created_at`),
    KEY `idx_datasets_household` (`household_id`),
    CONSTRAINT `chk_datasets_time_range`
        CHECK (
            `time_start` IS NULL
            OR `time_end` IS NULL
            OR `time_end` >= `time_start`
        )
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='数据集表';

-- -----------------------------------------------------------
-- 3. analysis_results — 统计分析结果
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS `analysis_results` (
    `id`            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `dataset_id`    BIGINT UNSIGNED NOT NULL COMMENT '关联数据集',
    `total_kwh`     DECIMAL(12,4)   DEFAULT NULL COMMENT '总用电量 kWh',
    `daily_avg_kwh` DECIMAL(10,4)   DEFAULT NULL COMMENT '日均用电量 kWh',
    `max_load_w`    DECIMAL(10,2)   DEFAULT NULL COMMENT '最高负荷 W',
    `max_load_time` DATETIME        DEFAULT NULL COMMENT '最高负荷时段',
    `min_load_w`    DECIMAL(10,2)   DEFAULT NULL COMMENT '最低负荷 W',
    `min_load_time` DATETIME        DEFAULT NULL COMMENT '最低负荷时段',
    `peak_kwh`      DECIMAL(12,4)   DEFAULT NULL COMMENT '峰时用电量 kWh',
    `valley_kwh`    DECIMAL(12,4)   DEFAULT NULL COMMENT '谷时用电量 kWh',
    `flat_kwh`      DECIMAL(12,4)   DEFAULT NULL COMMENT '平时用电量 kWh',
    `peak_ratio`    DECIMAL(5,4)    DEFAULT NULL COMMENT '峰时占比',
    `valley_ratio`  DECIMAL(5,4)    DEFAULT NULL COMMENT '谷时占比',
    `flat_ratio`    DECIMAL(5,4)    DEFAULT NULL COMMENT '平时占比',
    `detail_path`   VARCHAR(512)    DEFAULT NULL COMMENT '详细分析数据文件路径',
    `created_at`    DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_analysis_results_dataset` (`dataset_id`),
    CONSTRAINT `fk_analysis_dataset`
        FOREIGN KEY (`dataset_id`) REFERENCES `datasets`(`id`) ON DELETE CASCADE,
    CONSTRAINT `chk_analysis_total_kwh`
        CHECK (`total_kwh` IS NULL OR `total_kwh` >= 0),
    CONSTRAINT `chk_analysis_daily_avg_kwh`
        CHECK (`daily_avg_kwh` IS NULL OR `daily_avg_kwh` >= 0),
    CONSTRAINT `chk_analysis_max_load_w`
        CHECK (`max_load_w` IS NULL OR `max_load_w` >= 0),
    CONSTRAINT `chk_analysis_min_load_w`
        CHECK (`min_load_w` IS NULL OR `min_load_w` >= 0),
    CONSTRAINT `chk_analysis_peak_kwh`
        CHECK (`peak_kwh` IS NULL OR `peak_kwh` >= 0),
    CONSTRAINT `chk_analysis_valley_kwh`
        CHECK (`valley_kwh` IS NULL OR `valley_kwh` >= 0),
    CONSTRAINT `chk_analysis_flat_kwh`
        CHECK (`flat_kwh` IS NULL OR `flat_kwh` >= 0),
    CONSTRAINT `chk_analysis_peak_ratio`
        CHECK (`peak_ratio` IS NULL OR (`peak_ratio` >= 0 AND `peak_ratio` <= 1)),
    CONSTRAINT `chk_analysis_valley_ratio`
        CHECK (`valley_ratio` IS NULL OR (`valley_ratio` >= 0 AND `valley_ratio` <= 1)),
    CONSTRAINT `chk_analysis_flat_ratio`
        CHECK (`flat_ratio` IS NULL OR (`flat_ratio` >= 0 AND `flat_ratio` <= 1))
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='统计分析结果表';

-- -----------------------------------------------------------
-- 4. classification_results — 行为分类结果
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS `classification_results` (
    `id`              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `dataset_id`      BIGINT UNSIGNED NOT NULL COMMENT '关联数据集',
    `model_type`      ENUM('xgboost')
                      NOT NULL COMMENT '模型架构',
    `predicted_label` VARCHAR(32)     NOT NULL COMMENT '预测标签：day_high_night_low/day_low_night_high/all_day_high/all_day_low',
    `confidence`      DECIMAL(5,4)    DEFAULT NULL COMMENT '最高类别置信度',
    `probabilities`   JSON            DEFAULT NULL COMMENT '4 类稳定标签的概率分布',
    `explanation`     TEXT            DEFAULT NULL COMMENT '基于 day_mean/night_mean/full_mean 的分类依据说明',
    `window_start`    DATETIME        DEFAULT NULL COMMENT '窗口起始时间',
    `window_end`      DATETIME        DEFAULT NULL COMMENT '窗口结束时间',
    `created_at`      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`),
    KEY `idx_cls_dataset` (`dataset_id`),
    KEY `idx_cls_model` (`dataset_id`, `model_type`),
    KEY `idx_cls_dataset_created` (`dataset_id`, `created_at`),
    CONSTRAINT `fk_cls_dataset`
        FOREIGN KEY (`dataset_id`) REFERENCES `datasets`(`id`) ON DELETE CASCADE,
    CONSTRAINT `chk_cls_confidence`
        CHECK (`confidence` IS NULL OR (`confidence` >= 0 AND `confidence` <= 1)),
    CONSTRAINT `chk_cls_window_range`
        CHECK (
            `window_start` IS NULL
            OR `window_end` IS NULL
            OR `window_end` >= `window_start`
        )
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='行为分类结果表';

-- -----------------------------------------------------------
-- classification_results 历史库升级说明
-- 仅用于已经存在旧表且 model_type 仍为 tcn 的场景
-- -----------------------------------------------------------
-- ALTER TABLE `classification_results`
-- MODIFY COLUMN `model_type` ENUM('tcn','xgboost') NOT NULL COMMENT '模型架构';
--
-- UPDATE `classification_results`
-- SET `model_type` = 'xgboost'
-- WHERE `model_type` = 'tcn';
--
-- ALTER TABLE `classification_results`
-- MODIFY COLUMN `model_type` ENUM('xgboost') NOT NULL COMMENT '模型架构';

-- -----------------------------------------------------------
-- 5. forecast_results — 时序预测结果
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS `forecast_results` (
    `id`             BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `dataset_id`     BIGINT UNSIGNED NOT NULL COMMENT '关联数据集',
    `model_type`     VARCHAR(32)
                     NOT NULL COMMENT '预测模型类型（API 对外统一为 tft）',
    `forecast_start` DATETIME        NOT NULL COMMENT '预测起始时间',
    `forecast_end`   DATETIME        NOT NULL COMMENT '预测结束时间',
    `granularity`    ENUM('15min','hourly','daily')
                     NOT NULL DEFAULT '15min' COMMENT '预测粒度',
    `summary`        JSON            DEFAULT NULL COMMENT '预测摘要，供前端与智能体快速复用',
    `detail_path`    VARCHAR(512)    NOT NULL COMMENT '预测值序列文件路径',
    `metrics`        JSON            DEFAULT NULL COMMENT '评估指标（当前统一使用 mae/rmse/smape/wape）',
    `created_at`     DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`),
    KEY `idx_fc_dataset` (`dataset_id`),
    KEY `idx_fc_model` (`dataset_id`, `model_type`),
    KEY `idx_fc_dataset_created` (`dataset_id`, `created_at`),
    CONSTRAINT `fk_fc_dataset`
        FOREIGN KEY (`dataset_id`) REFERENCES `datasets`(`id`) ON DELETE CASCADE,
    CONSTRAINT `chk_fc_time_range`
        CHECK (`forecast_end` >= `forecast_start`)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='时序预测结果表';

-- -----------------------------------------------------------
-- 6. chat_sessions — 聊天会话
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS `chat_sessions` (
    `id`         BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `dataset_id` BIGINT UNSIGNED DEFAULT NULL COMMENT '关联数据集，空表示通用对话',
    `title`      VARCHAR(128)    DEFAULT NULL COMMENT '会话标题',
    `created_at` DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `updated_at` DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP COMMENT '最后活跃时间',
    PRIMARY KEY (`id`),
    KEY `idx_session_dataset` (`dataset_id`),
    KEY `idx_session_updated` (`updated_at`),
    CONSTRAINT `fk_session_dataset`
        FOREIGN KEY (`dataset_id`) REFERENCES `datasets`(`id`) ON DELETE SET NULL
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='聊天会话表';

-- -----------------------------------------------------------
-- 7. chat_messages — 聊天消息
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS `chat_messages` (
    `id`           BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `session_id`   BIGINT UNSIGNED NOT NULL COMMENT '关联会话',
    `role`         ENUM('user','assistant','system') NOT NULL COMMENT '消息角色',
    `content`      TEXT            DEFAULT NULL COMMENT '短消息内容',
    `content_path` VARCHAR(512)    DEFAULT NULL COMMENT '长消息文件路径',
    `model_name`   VARCHAR(128)    DEFAULT NULL COMMENT '使用的模型名称',
    `tokens_used`  INT UNSIGNED    NOT NULL DEFAULT 0 COMMENT 'token 消耗量',
    `created_at`   DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '发送时间',
    PRIMARY KEY (`id`),
    KEY `idx_msg_session` (`session_id`),
    KEY `idx_msg_created` (`session_id`, `created_at`),
    CONSTRAINT `fk_msg_session`
        FOREIGN KEY (`session_id`) REFERENCES `chat_sessions`(`id`) ON DELETE CASCADE,
    CONSTRAINT `chk_msg_content_exists`
        CHECK (`content` IS NOT NULL OR `content_path` IS NOT NULL)
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='聊天消息表';

-- -----------------------------------------------------------
-- 8. energy_advices — 节能建议
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS `energy_advices` (
    `id`                BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `dataset_id`        BIGINT UNSIGNED NOT NULL COMMENT '关联数据集',
    `classification_id` BIGINT UNSIGNED DEFAULT NULL COMMENT '关联分类结果',
    `advice_type`       ENUM('rule','llm') NOT NULL DEFAULT 'rule' COMMENT '建议来源',
    `content_path`      VARCHAR(512)    NOT NULL COMMENT '建议内容文件路径',
    `summary`           VARCHAR(512)    DEFAULT NULL COMMENT '建议摘要',
    `created_at`        DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`),
    KEY `idx_advice_dataset` (`dataset_id`),
    KEY `idx_advice_cls` (`classification_id`),
    KEY `idx_advice_type` (`advice_type`),
    CONSTRAINT `fk_advice_dataset`
        FOREIGN KEY (`dataset_id`) REFERENCES `datasets`(`id`) ON DELETE CASCADE,
    CONSTRAINT `fk_advice_cls`
        FOREIGN KEY (`classification_id`) REFERENCES `classification_results`(`id`) ON DELETE SET NULL
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='节能建议表';

-- -----------------------------------------------------------
-- 9. reports — 导出报告
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS `reports` (
    `id`          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `dataset_id`  BIGINT UNSIGNED NOT NULL COMMENT '关联数据集',
    `report_type` ENUM('excel','html','pdf') NOT NULL COMMENT '报告格式',
    `file_path`   VARCHAR(512)    NOT NULL COMMENT '报告文件路径',
    `file_size`   BIGINT UNSIGNED NOT NULL DEFAULT 0 COMMENT '文件大小（字节）',
    `created_at`  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    PRIMARY KEY (`id`),
    KEY `idx_report_dataset` (`dataset_id`),
    KEY `idx_report_type` (`report_type`),
    CONSTRAINT `fk_report_dataset`
        FOREIGN KEY (`dataset_id`) REFERENCES `datasets`(`id`) ON DELETE CASCADE
) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='导出报告表';

SET FOREIGN_KEY_CHECKS = 1;
