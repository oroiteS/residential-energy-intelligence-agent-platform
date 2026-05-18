from __future__ import annotations

from flask import current_app
from sqlalchemy import text

from app.extensions import db
from app.services.llm_client import can_use_llm, normalize_openai_base_url
from models.common import ARTIFACTS_DIR


def get_health_payload() -> dict:
    dependencies = {
        "database": "down",
        "model": _model_status(),
        "agent": _agent_status(),
    }
    try:
        db.session.execute(text("SELECT 1"))
        dependencies["database"] = "up"
    except Exception:
        dependencies["database"] = "down"

    if dependencies["database"] == "up":
        status = "degraded" if any(value != "up" for value in dependencies.values()) else "up"
    else:
        status = "down"

    return {
        "status": status,
        "service": "resident-energy-flask-backend",
        "version": "0.1.0",
        "dependencies": dependencies,
        "peak_valley_config": {
            "peak": current_app.config.get("PEAK_PERIODS", []),
            "valley": current_app.config.get("VALLEY_PERIODS", []),
        },
        "agent_config": {
            "base_url": normalize_openai_base_url(current_app.config.get("LLM_BASE_URL")),
            "model": current_app.config.get("LLM_MODEL") or "",
            "api_key_configured": bool(current_app.config.get("LLM_API_KEY")),
        },
    }


def _model_status() -> str:
    required_files = [
        ARTIFACTS_DIR / "classification" / "xgboost" / "xgboost_model.json",
        ARTIFACTS_DIR / "classification" / "xgboost" / "label_encoder.pkl",
        ARTIFACTS_DIR / "detection" / "isolation_forest" / "isolation_forest.pkl",
        ARTIFACTS_DIR / "detection" / "statistical_rules" / "rule_thresholds.json",
        ARTIFACTS_DIR / "forecast" / "lstm" / "checkpoints" / "best.ckpt",
        ARTIFACTS_DIR / "forecast" / "lstm" / "input_scalers.npz",
        ARTIFACTS_DIR / "forecast" / "lstm" / "feature_columns.json",
    ]
    return "up" if all(path.exists() for path in required_files) else "degraded"


def _agent_status() -> str:
    return "up" if can_use_llm() else "degraded"
