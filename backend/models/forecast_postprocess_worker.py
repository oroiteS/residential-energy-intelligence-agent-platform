from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# 限制数值计算库线程数。
# worker 是短生命周期子进程，避免每次后处理都启动过多 BLAS/OpenMP 线程。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def main() -> int:
    """预测后处理子进程入口。"""

    if len(sys.argv) != 2:
        raise ValueError("预测后处理子进程必须指定任务类型")

    task = sys.argv[1]
    payload = json.loads(sys.stdin.read() or "{}")
    if task == "classify":
        return _run_classification(payload)
    if task == "detect":
        return _run_detection(payload)
    raise ValueError(f"未知预测后处理任务: {task}")


def _run_classification(payload: dict) -> int:
    """执行未来窗口分类任务并输出 JSON。"""

    from models.classification import classify_daily_window

    window_rows = _parse_rows(payload.get("window_rows"))
    result = classify_daily_window(window_rows)
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    return 0


def _run_detection(payload: dict) -> int:
    """执行未来窗口异常检测任务并输出 JSON。"""

    from models.detection import detect_daily_window

    window_rows = _parse_rows(payload.get("window_rows"))
    history_rows = _parse_rows(payload.get("history_rows"))
    result = detect_daily_window(window_rows, history_rows, window_role="future")
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    return 0


def _parse_rows(value) -> list[dict]:
    """解析父进程传入的日级数据行。"""

    if not isinstance(value, list):
        raise ValueError("预测后处理输入 rows 必须为列表")

    rows: list[dict] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("预测后处理输入 row 必须为对象")
        row = dict(item)
        if isinstance(row.get("date"), str):
            row["date"] = date.fromisoformat(row["date"])
        rows.append(row)
    return rows


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        sys.stderr.write(str(exc).strip() or repr(exc))
        raise SystemExit(1)
