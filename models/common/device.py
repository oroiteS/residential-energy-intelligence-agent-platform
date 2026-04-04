"""设备检测工具。"""

from __future__ import annotations


def get_device_priority() -> list[str]:
    return ["cuda", "mps", "cpu"]


def detect_device() -> str:
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"
