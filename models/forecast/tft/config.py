"""TFT 预测任务配置加载。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from common.config_validation import validate_config_schema


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "config.yaml"
TOP_LEVEL_KEYS = {"data", "model", "loss", "train", "test"}
SECTION_KEYS = {
    "data": {
        "metadata_path",
        "feature_spec_path",
        "features_path",
        "targets_path",
        "split_mode",
        "train_ratio",
        "val_ratio",
        "batch_size",
        "num_workers",
        "pin_memory",
        "prefetch_factor",
        "baseline_week_weight",
        "baseline_recent_weight",
        "aggregate_feature_name",
        "normalization_quantile",
    },
    "model": {
        "hidden_size",
        "lstm_layers",
        "attention_heads",
        "dropout",
        "return_attention_weights",
        "router_prior_weight",
        "global_gate_bias_init",
        "local_gate_bias_init",
    },
    "loss": {
        "huber_delta",
        "diff_weight",
        "peak_weight",
        "peak_quantile",
        "ramp_weight",
        "guard_weight",
        "guard_margin",
        "gate_regularization",
        "regularization_warmup_epochs",
        "regularization_ramp_epochs",
    },
    "train": {
        "output_dir",
        "seed",
        "max_epochs",
        "learning_rate",
        "weight_decay",
        "gradient_clip_val",
        "accelerator",
        "devices",
        "precision",
        "early_stopping_patience",
        "log_every_n_steps",
        "deterministic",
        "benchmark",
        "enable_tf32",
        "matmul_precision",
        "compile_model",
        "compile_mode",
        "enable_progress_bar",
        "progress_bar_refresh_rate",
        "enable_model_summary",
        "num_sanity_val_steps",
    },
    "test": {
        "checkpoint_path",
        "output_dir",
    },
}


@dataclass(slots=True)
class DataConfig:
    metadata_path: Path
    feature_spec_path: Path
    features_path: Path | None
    targets_path: Path | None
    split_mode: str
    train_ratio: float
    val_ratio: float
    batch_size: int
    num_workers: int
    pin_memory: bool
    prefetch_factor: int
    baseline_week_weight: float
    baseline_recent_weight: float
    aggregate_feature_name: str
    normalization_quantile: float


@dataclass(slots=True)
class ModelConfig:
    hidden_size: int
    lstm_layers: int
    attention_heads: int
    dropout: float
    return_attention_weights: bool
    router_prior_weight: float
    global_gate_bias_init: float
    local_gate_bias_init: float


@dataclass(slots=True)
class LossConfig:
    huber_delta: float
    diff_weight: float
    peak_weight: float
    peak_quantile: float
    ramp_weight: float
    guard_weight: float
    guard_margin: float
    gate_regularization: float
    regularization_warmup_epochs: int
    regularization_ramp_epochs: int


@dataclass(slots=True)
class TrainConfig:
    output_dir: Path
    seed: int
    max_epochs: int
    learning_rate: float
    weight_decay: float
    gradient_clip_val: float
    accelerator: str
    devices: int | str
    precision: str
    early_stopping_patience: int
    log_every_n_steps: int
    deterministic: bool
    benchmark: bool
    enable_tf32: bool
    matmul_precision: str
    compile_model: bool
    compile_mode: str
    enable_progress_bar: bool
    progress_bar_refresh_rate: int
    enable_model_summary: bool
    num_sanity_val_steps: int


@dataclass(slots=True)
class TestConfig:
    checkpoint_path: Path | None
    output_dir: Path


@dataclass(slots=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    train: TrainConfig
    test: TestConfig

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["data"]["metadata_path"] = str(self.data.metadata_path)
        payload["data"]["feature_spec_path"] = str(self.data.feature_spec_path)
        payload["data"]["features_path"] = (
            str(self.data.features_path) if self.data.features_path is not None else None
        )
        payload["data"]["targets_path"] = (
            str(self.data.targets_path) if self.data.targets_path is not None else None
        )
        payload["train"]["output_dir"] = str(self.train.output_dir)
        payload["test"]["output_dir"] = str(self.test.output_dir)
        payload["test"]["checkpoint_path"] = (
            str(self.test.checkpoint_path) if self.test.checkpoint_path is not None else None
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
    resolved_config_path = config_path.resolve()
    raw_config = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
    base_dir = Path(__file__).resolve().parents[2]
    sections = validate_config_schema(
        raw_config,
        config_path=resolved_config_path,
        allowed_top_level_keys=TOP_LEVEL_KEYS,
        allowed_section_keys=SECTION_KEYS,
    )

    data_raw = sections["data"]
    model_raw = sections["model"]
    loss_raw = sections["loss"]
    train_raw = sections["train"]
    test_raw = sections["test"]

    data_config = DataConfig(
        metadata_path=_resolve_path(str(data_raw["metadata_path"]), base_dir),
        feature_spec_path=_resolve_path(str(data_raw["feature_spec_path"]), base_dir),
        features_path=_resolve_path(data_raw.get("features_path"), base_dir),
        targets_path=_resolve_path(data_raw.get("targets_path"), base_dir),
        split_mode=str(data_raw.get("split_mode", "by_house")),
        train_ratio=float(data_raw.get("train_ratio", 0.7)),
        val_ratio=float(data_raw.get("val_ratio", 0.15)),
        batch_size=int(data_raw.get("batch_size", 64)),
        num_workers=int(data_raw.get("num_workers", 4)),
        pin_memory=bool(data_raw.get("pin_memory", True)),
        prefetch_factor=int(data_raw.get("prefetch_factor", 4)),
        baseline_week_weight=float(data_raw.get("baseline_week_weight", 0.8)),
        baseline_recent_weight=float(data_raw.get("baseline_recent_weight", 0.2)),
        aggregate_feature_name=str(data_raw.get("aggregate_feature_name", "aggregate")),
        normalization_quantile=float(data_raw.get("normalization_quantile", 0.95)),
    )
    model_config = ModelConfig(
        hidden_size=int(model_raw.get("hidden_size", 96)),
        lstm_layers=int(model_raw.get("lstm_layers", 2)),
        attention_heads=int(model_raw.get("attention_heads", 4)),
        dropout=float(model_raw.get("dropout", 0.15)),
        return_attention_weights=bool(model_raw.get("return_attention_weights", False)),
        router_prior_weight=float(model_raw.get("router_prior_weight", 1.0)),
        global_gate_bias_init=float(model_raw.get("global_gate_bias_init", -0.2)),
        local_gate_bias_init=float(model_raw.get("local_gate_bias_init", 0.0)),
    )
    loss_config = LossConfig(
        huber_delta=float(loss_raw.get("huber_delta", 1.0)),
        diff_weight=float(loss_raw.get("diff_weight", 0.2)),
        peak_weight=float(loss_raw.get("peak_weight", 0.4)),
        peak_quantile=float(loss_raw.get("peak_quantile", 0.9)),
        ramp_weight=float(loss_raw.get("ramp_weight", 0.3)),
        guard_weight=float(loss_raw.get("guard_weight", 0.05)),
        guard_margin=float(loss_raw.get("guard_margin", 0.02)),
        gate_regularization=float(loss_raw.get("gate_regularization", 0.001)),
        regularization_warmup_epochs=int(loss_raw.get("regularization_warmup_epochs", 4)),
        regularization_ramp_epochs=int(loss_raw.get("regularization_ramp_epochs", 6)),
    )
    train_config = TrainConfig(
        output_dir=_resolve_path(str(train_raw["output_dir"]), base_dir),
        seed=int(train_raw.get("seed", 42)),
        max_epochs=int(train_raw.get("max_epochs", 30)),
        learning_rate=float(train_raw.get("learning_rate", 1e-3)),
        weight_decay=float(train_raw.get("weight_decay", 1e-4)),
        gradient_clip_val=float(train_raw.get("gradient_clip_val", 1.0)),
        accelerator=str(train_raw.get("accelerator", "auto")),
        devices=train_raw.get("devices", 1),
        precision=str(train_raw.get("precision", "32-true")),
        early_stopping_patience=int(train_raw.get("early_stopping_patience", 8)),
        log_every_n_steps=int(train_raw.get("log_every_n_steps", 20)),
        deterministic=bool(train_raw.get("deterministic", True)),
        benchmark=bool(train_raw.get("benchmark", True)),
        enable_tf32=bool(train_raw.get("enable_tf32", True)),
        matmul_precision=str(train_raw.get("matmul_precision", "high")),
        compile_model=bool(train_raw.get("compile_model", True)),
        compile_mode=str(train_raw.get("compile_mode", "max-autotune")),
        enable_progress_bar=bool(train_raw.get("enable_progress_bar", False)),
        progress_bar_refresh_rate=int(train_raw.get("progress_bar_refresh_rate", 10)),
        enable_model_summary=bool(train_raw.get("enable_model_summary", False)),
        num_sanity_val_steps=int(train_raw.get("num_sanity_val_steps", 0)),
    )
    checkpoint_path = test_raw.get("checkpoint_path")
    test_config = TestConfig(
        checkpoint_path=_resolve_path(checkpoint_path, base_dir) if checkpoint_path else None,
        output_dir=_resolve_path(str(test_raw["output_dir"]), base_dir),
    )

    if data_config.split_mode not in {"by_house", "random"}:
        raise ValueError("配置错误：data.split_mode 仅支持 by_house 或 random")
    if data_config.train_ratio + data_config.val_ratio >= 1.0:
        raise ValueError("配置错误：data.train_ratio + data.val_ratio 必须小于 1")
    if data_config.batch_size <= 0:
        raise ValueError("配置错误：data.batch_size 必须大于 0")
    if data_config.num_workers < 0:
        raise ValueError("配置错误：data.num_workers 不能为负数")
    if data_config.prefetch_factor <= 0:
        raise ValueError("配置错误：data.prefetch_factor 必须大于 0")
    if data_config.baseline_week_weight < 0 or data_config.baseline_recent_weight < 0:
        raise ValueError("配置错误：baseline 权重不能为负数")
    if abs(data_config.baseline_week_weight + data_config.baseline_recent_weight - 1.0) > 1e-6:
        raise ValueError("配置错误：baseline_week_weight + baseline_recent_weight 必须等于 1")
    if not (0.0 < data_config.normalization_quantile <= 1.0):
        raise ValueError("配置错误：normalization_quantile 必须在 (0, 1] 区间内")
    if model_config.hidden_size <= 0:
        raise ValueError("配置错误：model.hidden_size 必须大于 0")
    if model_config.attention_heads <= 0:
        raise ValueError("配置错误：model.attention_heads 必须大于 0")
    if model_config.hidden_size % model_config.attention_heads != 0:
        raise ValueError("配置错误：model.hidden_size 必须能被 model.attention_heads 整除")
    if model_config.router_prior_weight < 0:
        raise ValueError("配置错误：model.router_prior_weight 不能为负数")
    if not (0.0 < loss_config.peak_quantile < 1.0):
        raise ValueError("配置错误：loss.peak_quantile 必须在 (0, 1) 区间内")
    if loss_config.peak_weight < 0 or loss_config.ramp_weight < 0:
        raise ValueError("配置错误：loss.peak_weight 与 loss.ramp_weight 不能为负数")
    if loss_config.guard_weight < 0 or loss_config.gate_regularization < 0:
        raise ValueError("配置错误：guard_weight 与 gate_regularization 不能为负数")
    if loss_config.guard_margin < 0:
        raise ValueError("配置错误：loss.guard_margin 不能为负数")
    if loss_config.regularization_warmup_epochs < 0:
        raise ValueError("配置错误：loss.regularization_warmup_epochs 不能为负数")
    if loss_config.regularization_ramp_epochs < 0:
        raise ValueError("配置错误：loss.regularization_ramp_epochs 不能为负数")
    if train_config.matmul_precision not in {"highest", "high", "medium"}:
        raise ValueError("配置错误：train.matmul_precision 仅支持 highest / high / medium")
    if train_config.compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
        raise ValueError(
            "配置错误：train.compile_mode 仅支持 default / reduce-overhead / max-autotune"
        )
    if train_config.progress_bar_refresh_rate <= 0:
        raise ValueError("配置错误：train.progress_bar_refresh_rate 必须大于 0")
    if train_config.num_sanity_val_steps < 0:
        raise ValueError("配置错误：train.num_sanity_val_steps 不能为负数")
    if train_config.max_epochs <= 0:
        raise ValueError("配置错误：train.max_epochs 必须大于 0")
    return ExperimentConfig(
        data=data_config,
        model=model_config,
        loss=loss_config,
        train=train_config,
        test=test_config,
    )
