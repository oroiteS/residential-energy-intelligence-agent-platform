from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "resident-energy-dev")
    DEBUG = _as_bool(os.getenv("FLASK_DEBUG"), default=True)

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

    API_PREFIX = os.getenv("API_PREFIX", "/api/v1")
    APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT = int(os.getenv("APP_PORT", "5000"))

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

    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH_MB", "20")) * 1024 * 1024

    PEAK_PERIODS = os.getenv("PEAK_PERIODS", "07:00-11:00,18:00-23:00").split(",")
    VALLEY_PERIODS = os.getenv("VALLEY_PERIODS", "23:00-07:00,11:00-18:00").split(",")
    CLASSIFICATION_DAYS = int(os.getenv("CLASSIFICATION_DAYS", "7"))
    DETECTION_DAYS = int(os.getenv("DETECTION_DAYS", "7"))
    FORECAST_HISTORY_DAYS = int(os.getenv("FORECAST_HISTORY_DAYS", "30"))
    FORECAST_HORIZON_DAYS = int(os.getenv("FORECAST_HORIZON_DAYS", "7"))

    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()
    LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
    LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
    LLM_REPORT_TIMEOUT_SECONDS = int(os.getenv("LLM_REPORT_TIMEOUT_SECONDS", str(LLM_TIMEOUT_SECONDS)))

    ACCEPTED_MIN_GRANULARITY_MINUTES = int(os.getenv("ACCEPTED_MIN_GRANULARITY_MINUTES", "1"))
    ACCEPTED_MAX_GRANULARITY_MINUTES = int(os.getenv("ACCEPTED_MAX_GRANULARITY_MINUTES", "60"))
