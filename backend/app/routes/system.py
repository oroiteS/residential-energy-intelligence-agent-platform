from __future__ import annotations

from flask import Blueprint, request

from app.api import success
from app.services.system_service import get_system_config, patch_system_config


system_bp = Blueprint("system", __name__)


@system_bp.get("/system/config")
def get_config():
    """读取运行时系统配置。"""

    return success(get_system_config())


@system_bp.patch("/system/config")
def update_config():
    """更新运行时系统配置。"""

    payload = request.get_json(silent=True) or {}
    return success(patch_system_config(payload))
