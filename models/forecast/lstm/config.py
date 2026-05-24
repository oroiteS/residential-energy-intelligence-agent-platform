"""LSTM 直接预测任务配置读取与路径解析。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_value: str | Path) -> Path:
    """将配置中的相对路径解析到 `models/` 工程根目录下。"""

    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: Path) -> dict[str, Any]:
    """读取 YAML 配置文件。"""

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
