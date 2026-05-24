from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent

# 统一从 backend/.env 加载本地配置。
# 未配置的项目会使用下方默认值，保证开发环境可以快速启动。
load_dotenv(BASE_DIR / ".env")


def _as_bool(value: str | None, default: bool = False) -> bool:
    """将环境变量字符串转换为布尔值。

    环境变量天然是字符串，这里统一兼容 1/true/yes/on 等常见写法；
    当变量不存在时返回调用方传入的默认值。
    """

    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    """后端运行时配置中心。

    该类集中管理 Flask、数据库、文件存储、模型窗口、峰谷时段和 LLM 参数。
    应用创建时会通过 app.config.from_object(Config) 注入这些配置。
    """

    # Flask 基础配置。
    SECRET_KEY = os.getenv("SECRET_KEY", "resident-energy-dev")
    DEBUG = _as_bool(os.getenv("FLASK_DEBUG"), default=True)

    # 数据库连接配置。
    # 默认使用本地 MySQL；DATABASE_URL 可一次性覆盖完整连接串，便于部署环境接入。
    DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
    DB_PORT = int(os.getenv("DB_PORT", "3306"))
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_NAME = os.getenv("DB_NAME", "resident")
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,
        "pool_recycle": 3600,
    }

    # HTTP 服务配置。
    # API_PREFIX 统一给所有蓝图添加版本前缀，便于后续接口演进。
    API_PREFIX = os.getenv("API_PREFIX", "/api/v1")
    APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT = int(os.getenv("APP_PORT", "5000"))

    # 文件存储目录配置。
    # 后端会把上传文件、规范化数据、聚合结果、分析结果、预测结果和报告分目录保存。
    STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", BASE_DIR / "storage"))
    UPLOAD_DIR = STORAGE_ROOT / "uploads"
    NORMALIZED_DIR = STORAGE_ROOT / "normalized"
    DAILY_DIR = STORAGE_ROOT / "daily"
    QUALITY_DIR = STORAGE_ROOT / "quality"
    ANALYSIS_DIR = STORAGE_ROOT / "analysis"
    FORECAST_DIR = STORAGE_ROOT / "forecasts"
    REPORT_DIR = STORAGE_ROOT / "reports"
    REPORT_MARKDOWN_DIR = STORAGE_ROOT / "reports_md"
    MD2PDF_SCRIPT_PATH = Path(os.getenv("MD2PDF_SCRIPT_PATH", BASE_DIR / "app" / "tools" / "md2pdf.py"))
    PDF_RENDER_TIMEOUT_SECONDS = int(os.getenv("PDF_RENDER_TIMEOUT_SECONDS", "120"))
    PDF_THEME = os.getenv("PDF_THEME", "github-light")
    PDF_COVER = _as_bool(os.getenv("PDF_COVER"), default=True)
    PDF_TOC = _as_bool(os.getenv("PDF_TOC"), default=True)

    # 上传体积限制。
    # 默认 20MB，用于防止超大数据文件直接打满后端内存或磁盘。
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH_MB", "20")) * 1024 * 1024

    # 业务时间窗口配置。
    # 峰谷时段用于用电分析；分类、检测和预测窗口用于控制模型读取多少历史数据。
    PEAK_PERIODS = os.getenv("PEAK_PERIODS", "07:00-11:00,18:00-23:00").split(",")
    VALLEY_PERIODS = os.getenv("VALLEY_PERIODS", "23:00-07:00,11:00-18:00").split(",")
    CLASSIFICATION_DAYS = int(os.getenv("CLASSIFICATION_DAYS", "7"))
    DETECTION_DAYS = int(os.getenv("DETECTION_DAYS", "7"))
    FORECAST_HISTORY_DAYS = int(os.getenv("FORECAST_HISTORY_DAYS", "30"))
    FORECAST_HORIZON_DAYS = int(os.getenv("FORECAST_HORIZON_DAYS", "7"))

    # 大模型配置。
    # 聊天、智能体和报告生成会读取这些配置；为空时相关能力会降级为不可用状态。
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()
    LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
    LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
    LLM_REPORT_TIMEOUT_SECONDS = int(os.getenv("LLM_REPORT_TIMEOUT_SECONDS", str(LLM_TIMEOUT_SECONDS)))

    # 上传数据粒度约束。
    # 项目接受 1 到 60 分钟粒度的数据，后续会统一聚合到日级指标。
    ACCEPTED_MIN_GRANULARITY_MINUTES = int(os.getenv("ACCEPTED_MIN_GRANULARITY_MINUTES", "1"))
    ACCEPTED_MAX_GRANULARITY_MINUTES = int(os.getenv("ACCEPTED_MAX_GRANULARITY_MINUTES", "60"))
