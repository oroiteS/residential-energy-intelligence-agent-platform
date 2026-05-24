#!/usr/bin/env python3
"""兼容旧命令的 XGBoost 预测训练入口。"""

from __future__ import annotations

try:
    from .main import main
except ImportError:  # 兼容 `python forecast/xgboost/train_xgboost.py`
    from main import main


if __name__ == "__main__":
    main()
