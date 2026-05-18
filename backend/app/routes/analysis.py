from __future__ import annotations

from flask import Blueprint

from app.api import success
from app.errors import NotFoundError
from app.services.analysis_service import get_analysis_payload


analysis_bp = Blueprint("analysis", __name__)


@analysis_bp.get("/datasets/<int:dataset_id>/analysis")
def get_analysis(dataset_id: int):
    try:
        return success(get_analysis_payload(dataset_id))
    except ValueError as exc:
        raise NotFoundError(str(exc), code="ANALYSIS_NOT_FOUND") from exc


@analysis_bp.post("/datasets/<int:dataset_id>/analysis/recompute")
def recompute_analysis(dataset_id: int):
    return get_analysis(dataset_id)

