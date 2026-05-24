from __future__ import annotations

from flask import Blueprint, request

from app.api import success
from app.errors import NotFoundError
from app.services.classification_service import (
    get_latest_classification,
    list_classifications,
    predict_classification,
)


classifications_bp = Blueprint("classifications", __name__)


@classifications_bp.post("/datasets/<int:dataset_id>/classifications/predict")
def post_classification(dataset_id: int):
    """触发数据集用电类型分类。"""

    payload = request.get_json(silent=True) or {}
    result = predict_classification(dataset_id, window_role=payload.get("window_role", "current"))
    return success({"classification": result})


@classifications_bp.get("/datasets/<int:dataset_id>/classifications/latest")
def get_latest(dataset_id: int):
    """获取最近一次当前窗口分类结果。"""

    result = get_latest_classification(dataset_id)
    if result is None:
        raise NotFoundError("分类结果不存在", code="CLASSIFICATION_NOT_FOUND")
    return success({"classification": result})


@classifications_bp.get("/datasets/<int:dataset_id>/classifications")
def get_classifications(dataset_id: int):
    """列出数据集的历史分类窗口结果。"""

    return success({"items": list_classifications(dataset_id)})
