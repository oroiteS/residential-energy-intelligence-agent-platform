"""服务配置。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_AGENT_ROOT = PROJECT_ROOT / "models_agent"
DEFAULT_ENV_FILE_PATH = MODELS_AGENT_ROOT / ".env"
DEFAULT_CLASSIFICATION_CONFIG_PATH = MODELS_AGENT_ROOT / "configs" / "classification.yaml"
DEFAULT_FORECAST_CONFIG_PATH = MODELS_AGENT_ROOT / "configs" / "forecast.yaml"
DEFAULT_MD2PDF_SCRIPT_PATH = MODELS_AGENT_ROOT / "app" / "tools" / "md2pdf.py"


@dataclass(slots=True)
class Settings:
    host: str
    port: int
    classification_config_path: Path
    forecast_config_path: Path
    llm_base_url: str | None
    llm_api_key: str | None
    llm_model: str | None
    llm_temperature: float
    llm_timeout_seconds: int
    llm_report_timeout_seconds: int
    pdf_render_timeout_seconds: int
    pdf_theme: str
    pdf_cover: bool
    pdf_toc: bool
    md2pdf_script_path: Path


def _load_dotenv_file(env_file_path: Path) -> None:
    """从 .env 读取配置，但不覆盖已注入的系统环境变量。"""

    if not env_file_path.exists():
        return

    for raw_line in env_file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


def _env_or_default(env_key: str, default: object | None = None) -> object | None:
    env_value = os.getenv(env_key)
    if env_value is not None and env_value != "":
        return env_value
    return default


def _resolve_path(raw_path: str | Path, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_settings() -> Settings:
    env_file_path = _resolve_path(
        os.getenv("ENV_FILE_PATH", str(DEFAULT_ENV_FILE_PATH)),
        MODELS_AGENT_ROOT,
    )
    _load_dotenv_file(env_file_path)

    return Settings(
        host=os.getenv("APP_HOST", "127.0.0.1"),
        port=int(os.getenv("APP_PORT", "8001")),
        classification_config_path=_resolve_path(
            os.getenv(
                "CLASSIFICATION_CONFIG_PATH",
                str(DEFAULT_CLASSIFICATION_CONFIG_PATH),
            ),
            MODELS_AGENT_ROOT,
        ),
        forecast_config_path=_resolve_path(
            os.getenv(
                "FORECAST_CONFIG_PATH",
                str(DEFAULT_FORECAST_CONFIG_PATH),
            ),
            MODELS_AGENT_ROOT,
        ),
        llm_base_url=_env_or_default("LLM_BASE_URL"),
        llm_api_key=_env_or_default("LLM_API_KEY"),
        llm_model=_env_or_default("LLM_MODEL"),
        llm_temperature=float(_env_or_default("LLM_TEMPERATURE", 0.2)),
        llm_timeout_seconds=int(_env_or_default("LLM_TIMEOUT_SECONDS", 60)),
        llm_report_timeout_seconds=int(
            _env_or_default(
                "LLM_REPORT_TIMEOUT_SECONDS",
                _env_or_default("LLM_TIMEOUT_SECONDS", 60),
            )
        ),
        pdf_render_timeout_seconds=int(_env_or_default("PDF_RENDER_TIMEOUT_SECONDS", 180)),
        pdf_theme=str(_env_or_default("PDF_THEME", "github-light")),
        pdf_cover=str(_env_or_default("PDF_COVER", "false")).lower() in {"1", "true", "yes", "on"},
        pdf_toc=str(_env_or_default("PDF_TOC", "false")).lower() in {"1", "true", "yes", "on"},
        md2pdf_script_path=_resolve_path(
            os.getenv("MD2PDF_SCRIPT_PATH", str(DEFAULT_MD2PDF_SCRIPT_PATH)),
            MODELS_AGENT_ROOT,
        ),
    )
