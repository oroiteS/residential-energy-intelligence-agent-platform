from __future__ import annotations

from flask import Blueprint, send_file

from app.api import success
from app.services.report_service import export_report, get_report_file, list_reports


reports_bp = Blueprint("reports", __name__)


@reports_bp.get("/datasets/<int:dataset_id>/reports")
def get_reports(dataset_id: int):
    return success({"items": list_reports(dataset_id)})


@reports_bp.post("/datasets/<int:dataset_id>/reports/export")
def post_export_report(dataset_id: int):
    return success({"report": export_report(dataset_id)})


@reports_bp.get("/reports/<int:report_id>/download")
def download_report(report_id: int):
    return send_file(get_report_file(report_id), as_attachment=True)

