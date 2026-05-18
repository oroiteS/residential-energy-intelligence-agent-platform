#!/usr/bin/env python3
"""手动重绘五模型对比主图。"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forecast.visualization.comparison import format_saved_paths, generate_comparison_figures


def main() -> None:
    saved = generate_comparison_figures()
    print("已生成主图与汇总文件：")
    print(format_saved_paths(saved))


if __name__ == "__main__":
    main()
