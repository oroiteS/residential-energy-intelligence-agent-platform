from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Sequence


def classify_future_window(window_rows: Sequence[dict], *, timeout_seconds: float = 30.0) -> dict:
    return _run_worker(
        "classify",
        {"window_rows": [_serialize_row(item) for item in window_rows]},
        timeout_seconds=timeout_seconds,
    )


def detect_future_window(
    window_rows: Sequence[dict],
    history_rows: Sequence[dict],
    *,
    timeout_seconds: float = 30.0,
) -> dict:
    return _run_worker(
        "detect",
        {
            "window_rows": [_serialize_row(item) for item in window_rows],
            "history_rows": [_serialize_row(item) for item in history_rows],
        },
        timeout_seconds=timeout_seconds,
    )


def _run_worker(task: str, payload: dict[str, Any], *, timeout_seconds: float) -> dict:
    worker_path = Path(__file__).resolve().with_name("forecast_postprocess_worker.py")
    completed = subprocess.run(
        [sys.executable, str(worker_path), task],
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        error_message = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"预测后处理子进程执行失败: {error_message or '子进程异常退出'}")

    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError("预测后处理子进程未返回结果")
    return json.loads(stdout)


def _serialize_row(row: dict) -> dict:
    serialized = dict(row)
    row_date = serialized.get("date")
    if isinstance(row_date, date):
        serialized["date"] = row_date.isoformat()
    return serialized
