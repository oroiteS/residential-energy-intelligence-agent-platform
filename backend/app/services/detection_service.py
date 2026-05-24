from __future__ import annotations

from datetime import datetime

from app.errors import ValidationError
from app.extensions import db
from app.models import DetectionResult
from app.services.classification_service import get_latest_classification
from app.services.dataset_service import get_daily_rows, get_dataset_or_404
from app.services.common import to_iso
from models.detection import detect_daily_window


def get_latest_detection_record(dataset_id: int, *, window_role: str = "current") -> DetectionResult | None:
    """查询某个数据集指定窗口角色下的最新检测记录。"""

    return (
        DetectionResult.query.filter_by(dataset_id=dataset_id, window_role=window_role)
        .order_by(DetectionResult.created_at.desc())
        .first()
    )


def ensure_current_detection(dataset_id: int) -> DetectionResult:
    """确保当前窗口存在异常检测结果。

    如果数据库已有结果则直接返回，否则触发一次当前窗口检测。
    """

    latest = get_latest_detection_record(dataset_id, window_role="current")
    if latest is not None:
        return latest

    return detect_current_window(dataset_id)


def detect_current_window(dataset_id: int) -> DetectionResult:
    """对最近一段日级数据执行当前异常检测。"""

    dataset = get_dataset_or_404(dataset_id)
    daily_rows = get_daily_rows(dataset)
    if not daily_rows:
        raise ValidationError("缺少按日聚合结果，无法执行异常检测", code="DAILY_DATA_NOT_FOUND")

    # 当前检测默认观察最近 7 天。
    # history 作为历史背景，用于规则或孤立森林类方法判断近期表现是否异常。
    window = daily_rows[-7:] if len(daily_rows) >= 7 else daily_rows
    history = daily_rows[:-7] if len(daily_rows) > 7 else daily_rows
    result = detect_daily_window(window, history, window_role="current")
    classification = get_latest_classification(dataset.id)

    # 检测结果不仅保存是否异常，还保存异常分数、严重程度、原因和分类提示。
    # classification_hint 可以帮助报告把“异常”和“用电类型”联系起来解释。
    record = DetectionResult(
        dataset_id=dataset.id,
        model_type="iforest_rules",
        window_role="current",
        window_start=datetime.combine(window[0]["date"], datetime.min.time()) if window else None,
        window_end=datetime.combine(window[-1]["date"], datetime.min.time()) if window else None,
        is_anomaly=result["is_anomaly"],
        anomaly_score=result["anomaly_score"],
        severity=result["severity"],
        reasons=result["reasons"],
        feature_summary=result["feature_summary"],
        classification_hint=classification["predicted_label"] if classification else None,
    )
    db.session.add(record)
    db.session.commit()
    return record


def get_current_detection(dataset_id: int) -> dict | None:
    """获取当前异常检测结果，不存在时自动生成。"""

    return detection_dto(ensure_current_detection(dataset_id))


def rerun_current_detection(dataset_id: int) -> dict | None:
    """重新执行当前窗口异常检测。"""

    record = detect_current_window(dataset_id)
    return detection_dto(record)


def detection_dto(record: DetectionResult | None) -> dict | None:
    """转换异常检测记录为接口返回结构。"""

    if record is None:
        return None
    return {
        "id": record.id,
        "dataset_id": record.dataset_id,
        "model_type": record.model_type,
        "window_start": to_iso(record.window_start),
        "window_end": to_iso(record.window_end),
        "window_role": record.window_role,
        "is_anomaly": record.is_anomaly,
        "anomaly_score": record.anomaly_score or 0,
        "severity": record.severity,
        "reasons": record.reasons or [],
        "feature_summary": record.feature_summary or {},
        "classification_hint": record.classification_hint,
        "created_at": to_iso(record.created_at),
    }
