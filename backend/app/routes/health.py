from __future__ import annotations

from flask import Blueprint

from app.api import success
from app.services.health_service import get_health_payload


health_bp = Blueprint("health", __name__)


@health_bp.get("/health")
def get_health():
    return success(get_health_payload())

