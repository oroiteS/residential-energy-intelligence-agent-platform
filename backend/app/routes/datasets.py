from __future__ import annotations

import json

from flask import Blueprint, request

from app.api import success
from app.services.dataset_service import get_dataset_detail, import_dataset, list_datasets


datasets_bp = Blueprint("datasets", __name__)


@datasets_bp.get("/datasets")
def get_datasets():
    """分页查询数据集列表。"""

    page = request.args.get("page", default=1, type=int)
    page_size = request.args.get("page_size", default=20, type=int)
    status = request.args.get("status", default=None, type=str)
    keyword = request.args.get("keyword", default=None, type=str)
    return success(list_datasets(page=page, page_size=page_size, status=status, keyword=keyword))


@datasets_bp.get("/datasets/<int:dataset_id>")
def get_dataset(dataset_id: int):
    """查询数据集详情。"""

    return success(get_dataset_detail(dataset_id))


@datasets_bp.post("/datasets/import")
def post_dataset_import():
    """导入 CSV 数据集。

    路由层从 multipart/form-data 中读取文件和表单字段；
    CSV 清洗、粒度校验、日聚合和分析结果生成由 dataset_service 完成。
    """

    mapping_text = request.form.get("column_mapping")
    column_mapping = json.loads(mapping_text) if mapping_text else None
    dataset = import_dataset(
        file=request.files.get("file"),
        name=request.form.get("name", ""),
        description=request.form.get("description"),
        household_id=request.form.get("household_id"),
        unit=request.form.get("unit", "w"),
        column_mapping=column_mapping,
    )
    return success({"dataset": dataset}, message="导入成功")
