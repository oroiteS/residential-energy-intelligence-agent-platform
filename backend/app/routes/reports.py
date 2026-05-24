from __future__ import annotations

from flask import Blueprint, send_file

from app.api import success
from app.services.report_service import export_report, get_report_file, list_reports


reports_bp = Blueprint("reports", __name__)


@reports_bp.get("/datasets/<int:dataset_id>/reports")
def get_reports(dataset_id: int):
    """查询数据集报告列表。"""

    return success({"items": list_reports(dataset_id)})


@reports_bp.post("/datasets/<int:dataset_id>/reports/export")
def post_export_report(dataset_id: int):
    """导出数据集 PDF 报告。"""

    return success({"report": export_report(dataset_id)})


@reports_bp.get("/reports/<int:report_id>/download")
def download_report(report_id: int):
    """下载已经生成的报告文件。"""

    return send_file(get_report_file(report_id), as_attachment=True)
