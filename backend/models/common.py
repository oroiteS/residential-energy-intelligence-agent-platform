from __future__ import annotations

from pathlib import Path


# 模型模块所在目录。
BACKEND_MODELS_DIR = Path(__file__).resolve().parent

# 训练产物根目录。
# 分类、检测和预测模型都会从该目录下读取各自的权重、编码器和阈值文件。
ARTIFACTS_DIR = BACKEND_MODELS_DIR / "artifacts"
