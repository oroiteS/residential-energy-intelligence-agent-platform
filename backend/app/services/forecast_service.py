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
    """执行日级用电预测。

    输入为数据集 ID 和预测日期范围；
    输出包括预测记录摘要，并同时生成未来窗口的分类和异常检测结果。
    """

    dataset = get_dataset_or_404(dataset_id)
    daily_rows = get_daily_rows(dataset)
    if not daily_rows:
        raise NotFoundError("数据集缺少按日聚合结果", code="DAILY_DATA_NOT_FOUND")

    # 将前端传入的 ISO 日期转换为 date。
    # horizon_days 使用闭区间计算，因此开始日期和结束日期同一天时预测 1 天。
    forecast_start_date = datetime.fromisoformat(forecast_start.replace("Z", "+00:00")).date()
    forecast_end_date = datetime.fromisoformat(forecast_end.replace("Z", "+00:00")).date()
    horizon_days = (forecast_end_date - forecast_start_date).days + 1

    # 预测只读取最近的历史窗口。
    # 默认 30 天，既控制模型输入长度，也减少过旧数据对未来预测的影响。
    history_days = current_app.config["FORECAST_HISTORY_DAYS"]
    history_rows = daily_rows[-history_days:] if len(daily_rows) >= history_days else daily_rows
    result = forecast_daily_series(
        history_rows,
        forecast_start=forecast_start_date,
        horizon_days=horizon_days,
    )

    # 完整预测序列写入 JSON 文件。
    # 数据库 ForecastResult 只保存摘要和 detail_path，避免表字段承载大量序列点。
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

    # 预测完成后立即生成未来分类结果。
    # 这样前端一次预测即可同时展示未来用电类型。
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

    # 同一预测窗口也会生成未来异常检测结果。
    # forecast_id 将三类结果串联起来，便于报告或详情页关联展示。
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
    """分页样式返回某个数据集的预测记录列表。"""

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
    """读取单次预测的摘要和完整预测序列。"""

    record = ForecastResult.query.get(forecast_id)
    if record is None:
        raise NotFoundError("预测结果不存在", code="FORECAST_NOT_FOUND")
    detail = read_json(record.detail_path, default={"series": []})
    return {
        "forecast": forecast_record_dto(record),
        "series": detail.get("series", []),
    }


def get_latest_forecast(dataset_id: int) -> ForecastResult | None:
    """查询某个数据集最近一次预测记录。"""

    return (
        ForecastResult.query.filter_by(dataset_id=dataset_id)
        .order_by(ForecastResult.created_at.desc())
        .first()
    )


def forecast_record_dto(record: ForecastResult) -> dict:
    """转换预测记录为接口返回结构。"""

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
