"""LSTM 预测任务配置加载。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from common.config_validation import validate_config_schema
from common.device import detect_device
from forecast.LSTM.constants import ALL_FEATURE_NAMES, TARGET_LENGTH


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "config.yaml"
TOP_LEVEL_KEYS = {"data", "model", "train", "test", "predict"}
SECTION_KEYS = {
    "data": {
        "data_path",
        "split_mode",
        "train_ratio",
        "val_ratio",
        "feature_names",
        "aggregate_normalization",
        "aggregate_norm_eps",
    },
    "model": {
        "input_size",
        "hidden_size",
        "num_layers",
        "dropout",
        "target_length",
        "teacher_forcing_ratio",
    },
    "train": {
        "output_dir",
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "seed",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "loss_name",
        "huber_delta",
    },
    "test": {"checkpoint_path", "output_dir", "batch_size"},
    "predict": {"checkpoint_path", "output_dir", "batch_size"},
}


@dataclass(slots=True)
class DataConfig:
    data_path: Path
    split_mode: str
    train_ratio: float
    val_ratio: float
    feature_names: list[str]
    aggregate_normalization: str
    aggregate_norm_eps: float


@dataclass(slots=True)
class ModelConfig:
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    target_length: int
    teacher_forcing_ratio: float


@dataclass(slots=True)
class TrainConfig:
    output_dir: Path
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float | None
    seed: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    loss_name: str
    huber_delta: float


@dataclass(slots=True)
class TestConfig:
    checkpoint_path: Path | None
    output_dir: Path
    batch_size: int


@dataclass(slots=True)
class PredictConfig:
    checkpoint_path: Path | None
    output_dir: Path
    batch_size: int


@dataclass(slots=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    test: TestConfig
    predict: PredictConfig

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["data"]["data_path"] = str(self.data.data_path)
        payload["train"]["output_dir"] = str(self.train.output_dir)
        payload["test"]["output_dir"] = str(self.test.output_dir)
        payload["predict"]["output_dir"] = str(self.predict.output_dir)
        payload["test"]["checkpoint_path"] = (
            str(self.test.checkpoint_path)
            if self.test.checkpoint_path is not None
            else None
        )
        payload["predict"]["checkpoint_path"] = (
            str(self.predict.checkpoint_path)
            if self.predict.checkpoint_path is not None
            else None
        )
        return payload


def _resolve_path(path_value: str | None, base_dir: Path) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_experiment_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> ExperimentConfig:
    config_path = config_path.resolve()
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_dir = Path(__file__).resolve().parents[2]
    sections = validate_config_schema(
        raw_config,
        config_path=config_path,
        allowed_top_level_keys=TOP_LEVEL_KEYS,
        allowed_section_keys=SECTION_KEYS,
    )
    data_raw = sections["data"]
    model_raw = sections["model"]
    train_raw = sections["train"]
    test_raw = sections["test"]
    predict_raw = sections["predict"]

    data_config = DataConfig(
        data_path=_resolve_path(data_raw["data_path"], base_dir),
        split_mode=str(data_raw.get("split_mode", "by_house")),
        train_ratio=float(data_raw.get("train_ratio", 0.7)),
        val_ratio=float(data_raw.get("val_ratio", 0.15)),
        feature_names=[
            str(item)
            for item in data_raw.get("feature_names", ["aggregate"])
        ],
        aggregate_normalization=str(
            data_raw.get("aggregate_normalization", "input_window")
        ),
        aggregate_norm_eps=float(data_raw.get("aggregate_norm_eps", 1e-6)),
    )
    model_config = ModelConfig(
        input_size=int(
            model_raw.get(
                "input_size",
                len(data_config.feature_names),
            )
        ),
        hidden_size=int(model_raw.get("hidden_size", 128)),
        num_layers=int(model_raw.get("num_layers", 2)),
        dropout=float(model_raw.get("dropout", 0.2)),
        target_length=int(model_raw.get("target_length", TARGET_LENGTH)),
        teacher_forcing_ratio=float(
            model_raw.get("teacher_forcing_ratio", 0.5)
        ),
    )
    train_config = TrainConfig(
        output_dir=_resolve_path(train_raw["output_dir"], base_dir),
        batch_size=int(train_raw.get("batch_size", 128)),
        epochs=int(train_raw.get("epochs", 20)),
        learning_rate=float(train_raw.get("learning_rate", 1e-3)),
        weight_decay=float(train_raw.get("weight_decay", 1e-4)),
        gradient_clip_norm=(
            None
            if train_raw.get("gradient_clip_norm") is None
            else float(train_raw.get("gradient_clip_norm", 1.0))
        ),
        seed=int(train_raw.get("seed", 42)),
        early_stopping_patience=int(
            train_raw.get("early_stopping_patience", 10)
        ),
        early_stopping_min_delta=float(
            train_raw.get("early_stopping_min_delta", 0.0)
        ),
        loss_name=str(train_raw.get("loss_name", "huber")),
        huber_delta=float(train_raw.get("huber_delta", 1.0)),
    )
    test_checkpoint = test_raw.get("checkpoint_path")
    test_config = TestConfig(
        checkpoint_path=(
            _resolve_path(test_checkpoint, base_dir) if test_checkpoint else None
        ),
        output_dir=_resolve_path(
            test_raw.get("output_dir", str(train_config.output_dir)),
            base_dir,
        ),
        batch_size=int(test_raw.get("batch_size", train_config.batch_size)),
    )
    predict_checkpoint = predict_raw.get("checkpoint_path")
    predict_config = PredictConfig(
        checkpoint_path=(
            _resolve_path(predict_checkpoint, base_dir)
            if predict_checkpoint
            else None
        ),
        output_dir=_resolve_path(predict_raw["output_dir"], base_dir),
        batch_size=int(
            predict_raw.get("batch_size", train_config.batch_size)
        ),
    )

    if data_config.train_ratio + data_config.val_ratio >= 1:
        raise ValueError("配置错误：train_ratio + val_ratio 必须小于 1")
    if data_config.split_mode not in {"random", "by_house"}:
        raise ValueError("配置错误：split_mode 只支持 random 或 by_house")
    if not data_config.feature_names:
        raise ValueError("配置错误：feature_names 不能为空")
    if data_config.feature_names[0] != "aggregate":
        raise ValueError("配置错误：feature_names 第一个特征必须是 aggregate")
    if len(set(data_config.feature_names)) != len(data_config.feature_names):
        raise ValueError("配置错误：feature_names 不允许重复")
    invalid_feature_names = set(data_config.feature_names).difference(ALL_FEATURE_NAMES)
    if invalid_feature_names:
        raise ValueError(
            f"配置错误：存在不支持的 feature_names: {sorted(invalid_feature_names)}"
        )
    if data_config.aggregate_normalization not in {"input_window", "global"}:
        raise ValueError(
            "配置错误：aggregate_normalization 只支持 input_window 或 global"
        )
    if model_config.input_size != len(data_config.feature_names):
        raise ValueError(
            "配置错误：model.input_size 必须等于 feature_names 长度"
        )
    if train_config.loss_name.lower() not in {"mse", "huber"}:
        raise ValueError("配置错误：loss_name 只支持 mse 或 huber")
    if train_config.huber_delta <= 0:
        raise ValueError("配置错误：huber_delta 必须大于 0")
    if (
        train_config.gradient_clip_norm is not None
        and train_config.gradient_clip_norm <= 0
    ):
        raise ValueError("配置错误：gradient_clip_norm 必须大于 0 或设为 null")

    return ExperimentConfig(
        data=data_config,
        model=model_config,
        train=train_config,
        test=test_config,
        predict=predict_config,
    )
