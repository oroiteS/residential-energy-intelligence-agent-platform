from __future__ import annotations

from flask import Blueprint

from app.api import success
from app.services.detection_service import get_current_detection, rerun_current_detection


detections_bp = Blueprint("detections", __name__)


@detections_bp.get("/datasets/<int:dataset_id>/detections/current")
def get_detection(dataset_id: int):
    """获取当前窗口异常检测结果。"""

    return success({"detection": get_current_detection(dataset_id)})


@detections_bp.post("/datasets/<int:dataset_id>/detections/detect")
def post_detection(dataset_id: int):
    """重新执行当前窗口异常检测。"""

    return success({"detection": rerun_current_detection(dataset_id)})
