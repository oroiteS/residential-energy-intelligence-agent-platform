"""XGBoost 分类模型构造。"""

from __future__ import annotations

from typing import Any

import xgboost as xgb


def build_model(config: dict[str, Any], n_classes: int) -> xgb.XGBClassifier:
    """根据配置创建多分类 XGBoost 模型。

    objective=multi:softprob 表示输出每个类别的概率分布，
    后端推理时可以同时得到 predicted_label 和 confidence。
    """

    model_config = config["model"]
    return xgb.XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=n_classes,
        n_estimators=model_config["n_estimators"],
        max_depth=model_config["max_depth"],
        learning_rate=model_config["learning_rate"],
        subsample=model_config["subsample"],
        colsample_bytree=model_config["colsample_bytree"],
        min_child_weight=model_config["min_child_weight"],
        random_state=model_config["random_seed"],
        n_jobs=-1,
        verbosity=0,
    )
