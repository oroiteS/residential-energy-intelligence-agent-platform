from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Enum, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.extensions import db


class TimestampMixin:
    """为结果类数据表提供统一创建时间字段。"""

    created_at: Mapped[datetime] = mapped_column(default=datetime.now)


class ModelKwargsMixin:
    """兼容 SQLAlchemy 模型的关键字参数初始化。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


class Dataset(ModelKwargsMixin, db.Model):
    """数据集主表。

    保存用户上传文件、规范化文件、按日聚合文件、质量报告和处理状态。
    后续分析、预测、分类、检测和报告都通过 dataset_id 关联到这张表。
    """

    __tablename__ = "datasets"

    # 文件路径字段保存后端落盘产物位置。
    # raw 是原始上传文件，normalized 是清洗后的明细数据，daily 是日级聚合数据。
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(db.String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    household_id: Mapped[str | None] = mapped_column(db.String(64))
    source_file_name: Mapped[str | None] = mapped_column(db.String(255))
    raw_file_path: Mapped[str] = mapped_column(db.String(512), nullable=False)
    normalized_file_path: Mapped[str | None] = mapped_column(db.String(512))
    daily_aggregate_file_path: Mapped[str | None] = mapped_column(db.String(512))
    row_count: Mapped[int] = mapped_column(default=0, nullable=False)
    time_start: Mapped[datetime | None] = mapped_column()
    time_end: Mapped[datetime | None] = mapped_column()
    source_granularity_minutes: Mapped[int | None] = mapped_column()
    column_mapping: Mapped[dict | None] = mapped_column(JSON)
    quality_summary: Mapped[dict | None] = mapped_column(JSON)
    quality_report_path: Mapped[str | None] = mapped_column(db.String(512))

    # 数据集状态描述导入流程进度。
    # uploaded 表示刚上传，processing 表示处理中，ready 表示可用于分析，error 表示处理失败。
    status: Mapped[str] = mapped_column(
        Enum("uploaded", "processing", "ready", "error", name="dataset_status"),
        default="uploaded",
        nullable=False,
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.now,
        onupdate=datetime.now,
        nullable=False,
    )


class AnalysisResult(ModelKwargsMixin, db.Model, TimestampMixin):
    """用电分析结果表。

    该表保存一个数据集的总体用电量、日均用电、最大/最小负荷、峰谷用电量和图表明细路径。
    它是上传数据完成清洗和聚合之后的第一层统计结果。
    """

    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), unique=True, nullable=False)
    total_kwh: Mapped[float | None] = mapped_column()
    daily_avg_kwh: Mapped[float | None] = mapped_column()
    max_load_w: Mapped[float | None] = mapped_column()
    max_load_time: Mapped[datetime | None] = mapped_column()
    min_load_w: Mapped[float | None] = mapped_column()
    min_load_time: Mapped[datetime | None] = mapped_column()
    peak_kwh: Mapped[float | None] = mapped_column()
    valley_kwh: Mapped[float | None] = mapped_column()
    peak_ratio: Mapped[float | None] = mapped_column()
    valley_ratio: Mapped[float | None] = mapped_column()
    summary_json: Mapped[dict | None] = mapped_column(JSON)
    detail_path: Mapped[str | None] = mapped_column(db.String(512))


class ForecastResult(ModelKwargsMixin, db.Model, TimestampMixin):
    """用电预测结果表。

    保存预测任务的时间范围、历史窗口长度、预测步长、摘要指标和明细 JSON 路径。
    当前模型类型字段保留为 lstm，方便后续替换真实模型或增加模型版本。
    """

    __tablename__ = "forecast_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    schema_version: Mapped[str] = mapped_column(db.String(16), default="v1", nullable=False)
    model_type: Mapped[str] = mapped_column(Enum("lstm", name="forecast_model_type"), default="lstm", nullable=False)
    history_days: Mapped[int] = mapped_column(default=30, nullable=False)
    forecast_horizon_days: Mapped[int] = mapped_column(default=7, nullable=False)
    forecast_start: Mapped[datetime] = mapped_column()
    forecast_end: Mapped[datetime] = mapped_column()
    granularity: Mapped[str] = mapped_column(Enum("daily", name="forecast_granularity"), default="daily", nullable=False)
    summary: Mapped[dict | None] = mapped_column(JSON)
    detail_path: Mapped[str] = mapped_column(db.String(512), nullable=False)
    metrics: Mapped[dict | None] = mapped_column(JSON)


class ClassificationResult(ModelKwargsMixin, db.Model, TimestampMixin):
    """用电类型分类结果表。

    window_role 区分当前历史窗口和未来预测窗口；
    predicted_label、confidence 和 probabilities 用于前端展示分类结论和置信度。
    """

    __tablename__ = "classification_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    forecast_id: Mapped[int | None] = mapped_column(ForeignKey("forecast_results.id", ondelete="SET NULL"))
    schema_version: Mapped[str] = mapped_column(db.String(16), default="v1", nullable=False)
    model_type: Mapped[str] = mapped_column(Enum("xgboost", name="classification_model_type"), default="xgboost", nullable=False)
    window_role: Mapped[str] = mapped_column(Enum("current", "future", name="window_role"), default="current", nullable=False)
    predicted_label: Mapped[str] = mapped_column(db.String(32), nullable=False)
    confidence: Mapped[float | None] = mapped_column()
    probabilities: Mapped[dict | None] = mapped_column(JSON)
    explanation: Mapped[str | None] = mapped_column(Text)
    sample_id: Mapped[str | None] = mapped_column(db.String(128))
    runtime_library: Mapped[str | None] = mapped_column(db.String(64))
    window_start: Mapped[datetime | None] = mapped_column()
    window_end: Mapped[datetime | None] = mapped_column()


'''
class DetectionResult(ModelKwargsMixin, db.Model, TimestampMixin):
    """异常检测结果表。

    检测结果可以关联当前数据窗口，也可以关联某次预测结果；
    reasons 和 feature_summary 保存可解释信息，方便前端和报告说明异常原因。
    """

    __tablename__ = "detection_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    forecast_id: Mapped[int | None] = mapped_column(ForeignKey("forecast_results.id", ondelete="SET NULL"))
    schema_version: Mapped[str] = mapped_column(db.String(16), default="v1", nullable=False)
    model_type: Mapped[str] = mapped_column(Enum("iforest_rules", name="detection_model_type"), default="iforest_rules", nullable=False)
    window_role: Mapped[str] = mapped_column(Enum("current", "future", name="detection_window_role"), default="current", nullable=False)
    window_start: Mapped[datetime | None]
    window_end: Mapped[datetime | None]
    is_anomaly: Mapped[bool] = mapped_column(default=False, nullable=False)
    anomaly_score: Mapped[float | None]
    severity: Mapped[str | None] = mapped_column(Enum("low", "medium", "high", name="detection_severity"))
    reasons: Mapped[list | None] = mapped_column(JSON)
    feature_summary: Mapped[dict | None] = mapped_column(JSON)
    classification_hint: Mapped[str | None] = mapped_column(db.String(64))
'''

class DetectionResult(ModelKwargsMixin, db.Model, TimestampMixin):
    __tablename__ = "detection_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    forecast_id: Mapped[int | None] = mapped_column(ForeignKey("forecast_results.id", ondelete="SET NULL"))
    schema_version: Mapped[str] = mapped_column(db.String(16), default="v1", nullable=False)
    model_type: Mapped[str] = mapped_column(Enum("iforest_rules", name="detection_model_type"), default="iforest_rules", nullable=False)
    window_role: Mapped[str] = mapped_column(Enum("current", "future", name="detection_window_role"), default="current", nullable=False)

    window_start: Mapped[datetime | None] = mapped_column()
    window_end: Mapped[datetime | None] = mapped_column()
    is_anomaly: Mapped[bool] = mapped_column(default=False, nullable=False)
    anomaly_score: Mapped[float | None] = mapped_column()
    severity: Mapped[str | None] = mapped_column(Enum("low", "medium", "high", name="detection_severity"))
    reasons: Mapped[list | None] = mapped_column(JSON)
    feature_summary: Mapped[dict | None] = mapped_column(JSON)
    classification_hint: Mapped[str | None] = mapped_column(db.String(64))


class ChatSession(ModelKwargsMixin, db.Model):
    """对话会话表。

    一个会话可以绑定某个数据集，也可以作为普通问答会话存在；
    messages 关系按创建时间排序，便于恢复完整上下文。
    """

    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.id", ondelete="SET NULL"))
    title: Mapped[str | None] = mapped_column(db.String(128))
    created_at: Mapped[datetime] = mapped_column(default=datetime.now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.now, onupdate=datetime.now, nullable=False)

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )


class ChatMessage(ModelKwargsMixin, db.Model):
    """对话消息表。

    保存用户、助手或系统消息；assistant_payload 用于记录结构化工具调用结果，
    content_path 可保存较长内容的外部文件路径。
    """

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[str] = mapped_column(Enum("user", "assistant", "system", name="chat_role"), nullable=False)
    content: Mapped[str | None] = mapped_column(Text)
    assistant_payload: Mapped[dict | None] = mapped_column(JSON)
    content_path: Mapped[str | None] = mapped_column(db.String(512))
    model_name: Mapped[str | None] = mapped_column(db.String(128))
    tokens_used: Mapped[int | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(default=datetime.now, nullable=False)

    session: Mapped[ChatSession] = relationship(back_populates="messages")


class Report(ModelKwargsMixin, db.Model, TimestampMixin):
    """报告文件表。

    记录某个数据集生成的报告类型、文件路径和文件大小；
    当前 report_type 只支持 pdf，后续可扩展为 docx、html 等格式。
    """

    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    report_type: Mapped[str] = mapped_column(Enum("pdf", name="report_type"), default="pdf", nullable=False)
    file_path: Mapped[str] = mapped_column(db.String(512), nullable=False)
    file_size: Mapped[int] = mapped_column(default=0, nullable=False)
