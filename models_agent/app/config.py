"""服务配置。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_AGENT_ROOT = PROJECT_ROOT / "models_agent"
DEFAULT_AGENT_CONFIG_PATH = MODELS_AGENT_ROOT / "configs" / "agent.yaml"
DEFAULT_CLASSIFICATION_CONFIG_PATH = MODELS_AGENT_ROOT / "configs" / "classification.yaml"
DEFAULT_FORECAST_CONFIG_PATH = MODELS_AGENT_ROOT / "configs" / "forecast.yaml"


@dataclass(slots=True)
class Settings:
    host: str
    port: int
    classification_config_path: Path
    forecast_config_path: Path
    agent_config_path: Path
    llm_base_url: str | None
    llm_api_key: str | None
    llm_model: str | None
    llm_temperature: float
    llm_timeout_seconds: int


def _load_agent_yaml(config_path: Path) -> dict[str, object]:
    if not config_path.exists():
        return {}

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return {}

    llm_config = payload.get("llm", {})
    if not isinstance(llm_config, dict):
        return {}
    return llm_config


def _env_or_yaml(
    env_key: str,
    yaml_payload: dict[str, object],
    yaml_key: str,
    default: object | None = None,
) -> object | None:
    env_value = os.getenv(env_key)
    if env_value is not None and env_value != "":
        return env_value
    if yaml_key in yaml_payload and yaml_payload[yaml_key] not in {None, ""}:
        return yaml_payload[yaml_key]
    return default


def load_settings() -> Settings:
    agent_config_path = Path(
        os.getenv(
            "AGENT_CONFIG_PATH",
            str(DEFAULT_AGENT_CONFIG_PATH),
        )
    )
    agent_yaml = _load_agent_yaml(agent_config_path)

    return Settings(
        host=os.getenv("APP_HOST", "127.0.0.1"),
        port=int(os.getenv("APP_PORT", "8001")),
        classification_config_path=Path(
            os.getenv(
                "CLASSIFICATION_CONFIG_PATH",
                str(DEFAULT_CLASSIFICATION_CONFIG_PATH),
            )
        ),
        forecast_config_path=Path(
            os.getenv(
                "FORECAST_CONFIG_PATH",
                str(DEFAULT_FORECAST_CONFIG_PATH),
            )
        ),
        agent_config_path=agent_config_path,
        llm_base_url=_env_or_yaml("LLM_BASE_URL", agent_yaml, "base_url"),
        llm_api_key=_env_or_yaml("LLM_API_KEY", agent_yaml, "api_key"),
        llm_model=_env_or_yaml("LLM_MODEL", agent_yaml, "model"),
        llm_temperature=float(_env_or_yaml("LLM_TEMPERATURE", agent_yaml, "temperature", 0.2)),
        llm_timeout_seconds=int(
            _env_or_yaml("LLM_TIMEOUT_SECONDS", agent_yaml, "timeout_seconds", 60)
        ),
    )
