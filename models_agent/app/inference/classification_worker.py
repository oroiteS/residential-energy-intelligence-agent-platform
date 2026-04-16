"""分类推理子进程入口。"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from models_agent.app.inference.classification import predict_single_sample


def main() -> int:
    raw_payload = sys.stdin.read().strip()
    if not raw_payload:
        raise ValueError("分类子进程未收到输入")

    payload = json.loads(raw_payload)
    sample = payload.get("sample")
    config_path = payload.get("config_path")
    if not isinstance(sample, dict):
        raise ValueError("分类子进程 sample 格式错误")
    if not isinstance(config_path, str) or not config_path.strip():
        raise ValueError("分类子进程 config_path 缺失")

    result = predict_single_sample(sample=sample, config_path=Path(config_path))
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        sys.stderr.write(str(exc).strip() or repr(exc))
        raise SystemExit(1)
