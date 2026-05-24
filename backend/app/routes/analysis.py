from __future__ import annotations

from flask import Blueprint

from app.api import success
from app.errors import NotFoundError
from app.services.analysis_service import get_analysis_payload


analysis_bp = Blueprint("analysis", __name__)


@analysis_bp.get("/datasets/<int:dataset_id>/analysis")
def get_analysis(dataset_id: int):
    """获取数据集用电分析结果。"""

    try:
        return success(get_analysis_payload(dataset_id))
    except ValueError as exc:
        raise NotFoundError(str(exc), code="ANALYSIS_NOT_FOUND") from exc


@analysis_bp.post("/datasets/<int:dataset_id>/analysis/recompute")
def recompute_analysis(dataset_id: int):
    """重新计算分析结果的接口占位。

    当前分析结果在数据集导入时已经生成，因此这里复用查询逻辑。
    """

    return get_analysis(dataset_id)
