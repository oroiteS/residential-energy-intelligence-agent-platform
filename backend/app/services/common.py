from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any


def to_iso(value: datetime | date | None) -> str | None:
    """将日期时间对象转换为接口层可直接返回的 ISO 字符串。"""

    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    return value.isoformat()


def ensure_parent(path: Path) -> None:
    """确保目标文件的父目录存在。"""

    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    """按 UTF-8 和缩进格式写入 JSON 文件。

    后端的分析、预测、质量报告等明细结果都会以 JSON 形式落盘，
    数据库只保存摘要字段和文件路径。
    """

    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path, default: Any = None) -> Any:
    """读取 JSON 文件，文件不存在时返回默认值。"""

    file_path = Path(path)
    if not file_path.exists():
        return default
    return json.loads(file_path.read_text(encoding="utf-8"))


def slugify(value: str) -> str:
    """将数据集名称转换为适合文件名使用的短标识。"""

    cleaned = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff_-]+", "-", value.strip())
    return cleaned.strip("-") or "dataset"
