"""KMeans 聚类分析常量。"""

from __future__ import annotations

SEQUENCE_LENGTH = 96
AGGREGATE_COLUMNS = tuple(f"aggregate_{index:03d}" for index in range(SEQUENCE_LENGTH))
RAW_MEAN_COLUMNS = tuple(f"raw_mean_{index:03d}" for index in range(SEQUENCE_LENGTH))
NORMALIZED_MEAN_COLUMNS = tuple(
    f"normalized_mean_{index:03d}" for index in range(SEQUENCE_LENGTH)
)

__all__ = [
    "AGGREGATE_COLUMNS",
    "NORMALIZED_MEAN_COLUMNS",
    "RAW_MEAN_COLUMNS",
    "SEQUENCE_LENGTH",
]
