from __future__ import annotations

from copy import deepcopy

from flask import current_app


def get_system_config() -> dict:
    """读取运行时系统配置。

    返回 deepcopy，避免调用方直接修改 Flask config 中的原始对象。
    """

    runtime = current_app.config.setdefault("RUNTIME_SYSTEM_CONFIG", _default_system_config())
    return deepcopy(runtime)


def patch_system_config(payload: dict) -> dict:
    """更新运行时系统配置。

    当前支持峰谷时段配置和模型历史窗口配置；
    这些配置只保存在当前进程内，不会回写 .env。
    """

    runtime = current_app.config.setdefault("RUNTIME_SYSTEM_CONFIG", _default_system_config())
    if "peak_valley_config" in payload:
        runtime["peak_valley_config"] = payload["peak_valley_config"]
    if "model_history_window_config" in payload:
        runtime["model_history_window_config"] = payload["model_history_window_config"]
    return deepcopy(runtime)


def _default_system_config() -> dict:
    """根据 Flask 配置生成默认系统配置。"""

    return {
        "peak_valley_config": {
            "peak": current_app.config["PEAK_PERIODS"],
            "valley": current_app.config["VALLEY_PERIODS"],
        },
        "model_history_window_config": {
            "classification_days": current_app.config["CLASSIFICATION_DAYS"],
            "forecast_history_days": current_app.config["FORECAST_HISTORY_DAYS"],
        },
        "data_upload_dir": str(current_app.config["UPLOAD_DIR"]),
        "report_output_dir": str(current_app.config["REPORT_DIR"]),
    }
