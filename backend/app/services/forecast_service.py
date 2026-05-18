from __future__ import annotations

from datetime import datetime
from pathlib import Path

from flask import current_app

from app.errors import NotFoundError
from app.extensions import db
from app.models import ClassificationResult, DetectionResult, ForecastResult
from app.services.dataset_service import get_daily_rows, get_dataset_or_404
from app.services.common import read_json, to_iso, write_json
from models.forecast import forecast_daily_series


def predict_forecast(dataset_id: int, *, forecast_start: str, forecast_end: str) -> dict:
    dataset = get_dataset_or_404(dataset_id)
    daily_rows = get_daily_rows(dataset)
    if not daily_rows:
        raise NotFoundError("数据集缺少按日聚合结果", code="DAILY_DATA_NOT_FOUND")

    forecast_start_date = datetime.fromisoformat(forecast_start.replace("Z", "+00:00")).date()
    forecast_end_date = datetime.fromisoformat(forecast_end.replace("Z", "+00:00")).date()
    horizon_days = (forecast_end_date - forecast_start_date).days + 1

    history_days = current_app.config["FORECAST_HISTORY_DAYS"]
    history_rows = daily_rows[-history_days:] if len(daily_rows) >= history_days else daily_rows
    result = forecast_daily_series(
        history_rows,
        forecast_start=forecast_start_date,
        horizon_days=horizon_days,
    )

    detail_path = current_app.config["FORECAST_DIR"] / f"forecast_{dataset.id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    write_json(detail_path, result)

    record = ForecastResult(
        dataset_id=dataset.id,
        model_type="lstm",
        history_days=history_days,
        forecast_horizon_days=horizon_days,
        forecast_start=datetime.combine(forecast_start_date, datetime.min.time()),
        forecast_end=datetime.combine(forecast_end_date, datetime.min.time()),
        granularity="daily",
        summary=result["summary"],
        detail_path=str(detail_path),
        metrics={"method": "heuristic_baseline", "note": "当前为后端内置启发式预测实现"},
    )
    db.session.add(record)
    db.session.flush()

    classification_summary = result["summary"]["forecast_classification"]
    classification = ClassificationResult(
        dataset_id=dataset.id,
        forecast_id=record.id,
        model_type="xgboost",
        window_role="future",
        predicted_label=classification_summary["predicted_label"],
        confidence=classification_summary["confidence"],
        probabilities=classification_summary["probabilities"],
        explanation=result["classification"]["explanation"],
        sample_id=f"{dataset.id}-future-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        runtime_library="backend-heuristic",
        window_start=datetime.fromisoformat(classification_summary["window_start"]),
        window_end=datetime.fromisoformat(classification_summary["window_end"]),
    )
    detection_summary = result["summary"]["future_detection"]
    detection = DetectionResult(
        dataset_id=dataset.id,
        forecast_id=record.id,
        model_type="iforest_rules",
        window_role="future",
        window_start=datetime.fromisoformat(classification_summary["window_start"]),
        window_end=datetime.fromisoformat(classification_summary["window_end"]),
        is_anomaly=detection_summary["is_anomaly"],
        anomaly_score=detection_summary["anomaly_score"],
        severity=detection_summary["severity"],
        reasons=detection_summary["reasons"],
        feature_summary=detection_summary["feature_summary"],
        classification_hint=classification_summary["predicted_label"],
    )
    db.session.add_all([classification, detection])
    db.session.commit()
    return forecast_record_dto(record)


def list_forecasts(dataset_id: int) -> dict:
    records = (
        ForecastResult.query.filter_by(dataset_id=dataset_id)
        .order_by(ForecastResult.created_at.desc())
        .all()
    )
    return {
        "items": [forecast_record_dto(item) for item in records],
        "pagination": {"page": 1, "page_size": 20, "total": len(records)},
    }


def get_forecast_detail(forecast_id: int) -> dict:
    record = ForecastResult.query.get(forecast_id)
    if record is None:
        raise NotFoundError("预测结果不存在", code="FORECAST_NOT_FOUND")
    detail = read_json(record.detail_path, default={"series": []})
    return {
        "forecast": forecast_record_dto(record),
        "series": detail.get("series", []),
    }


def get_latest_forecast(dataset_id: int) -> ForecastResult | None:
    return (
        ForecastResult.query.filter_by(dataset_id=dataset_id)
        .order_by(ForecastResult.created_at.desc())
        .first()
    )


def forecast_record_dto(record: ForecastResult) -> dict:
    return {
        "id": record.id,
        "dataset_id": record.dataset_id,
        "model_type": record.model_type,
        "forecast_start": to_iso(record.forecast_start),
        "forecast_end": to_iso(record.forecast_end),
        "granularity": record.granularity,
        "summary": record.summary or {},
        "detail_path": record.detail_path,
        "created_at": to_iso(record.created_at),
    }

