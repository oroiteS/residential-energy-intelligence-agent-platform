from __future__ import annotations

from flask import Blueprint, request

from app.api import success
from app.services.forecast_service import get_forecast_detail, list_forecasts, predict_forecast


forecasts_bp = Blueprint("forecasts", __name__)


@forecasts_bp.post("/datasets/<int:dataset_id>/forecasts/predict")
def post_forecast(dataset_id: int):
    payload = request.get_json(silent=True) or {}
    result = predict_forecast(
        dataset_id,
        forecast_start=payload["forecast_start"],
        forecast_end=payload["forecast_end"],
    )
    return success({"forecast": result})


@forecasts_bp.get("/datasets/<int:dataset_id>/forecasts")
def get_forecasts(dataset_id: int):
    return success(list_forecasts(dataset_id))


@forecasts_bp.get("/forecasts/<int:forecast_id>")
def get_forecast(forecast_id: int):
    return success(get_forecast_detail(forecast_id))

