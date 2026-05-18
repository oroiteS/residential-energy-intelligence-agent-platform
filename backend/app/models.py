from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Enum, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.extensions import db


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(default=datetime.now)


class Dataset(db.Model):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(db.String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    household_id: Mapped[str | None] = mapped_column(db.String(64))
    source_file_name: Mapped[str | None] = mapped_column(db.String(255))
    raw_file_path: Mapped[str] = mapped_column(db.String(512), nullable=False)
    normalized_file_path: Mapped[str | None] = mapped_column(db.String(512))
    daily_aggregate_file_path: Mapped[str | None] = mapped_column(db.String(512))
    row_count: Mapped[int] = mapped_column(default=0, nullable=False)
    time_start: Mapped[datetime | None]
    time_end: Mapped[datetime | None]
    source_granularity_minutes: Mapped[int | None]
    column_mapping: Mapped[dict | None] = mapped_column(JSON)
    quality_summary: Mapped[dict | None] = mapped_column(JSON)
    quality_report_path: Mapped[str | None] = mapped_column(db.String(512))
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


class AnalysisResult(db.Model, TimestampMixin):
    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), unique=True, nullable=False)
    total_kwh: Mapped[float | None]
    daily_avg_kwh: Mapped[float | None]
    max_load_w: Mapped[float | None]
    max_load_time: Mapped[datetime | None]
    min_load_w: Mapped[float | None]
    min_load_time: Mapped[datetime | None]
    peak_kwh: Mapped[float | None]
    valley_kwh: Mapped[float | None]
    peak_ratio: Mapped[float | None]
    valley_ratio: Mapped[float | None]
    summary_json: Mapped[dict | None] = mapped_column(JSON)
    detail_path: Mapped[str | None] = mapped_column(db.String(512))


class ForecastResult(db.Model, TimestampMixin):
    __tablename__ = "forecast_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    schema_version: Mapped[str] = mapped_column(db.String(16), default="v1", nullable=False)
    model_type: Mapped[str] = mapped_column(Enum("lstm", name="forecast_model_type"), default="lstm", nullable=False)
    history_days: Mapped[int] = mapped_column(default=30, nullable=False)
    forecast_horizon_days: Mapped[int] = mapped_column(default=7, nullable=False)
    forecast_start: Mapped[datetime]
    forecast_end: Mapped[datetime]
    granularity: Mapped[str] = mapped_column(Enum("daily", name="forecast_granularity"), default="daily", nullable=False)
    summary: Mapped[dict | None] = mapped_column(JSON)
    detail_path: Mapped[str] = mapped_column(db.String(512), nullable=False)
    metrics: Mapped[dict | None] = mapped_column(JSON)


class ClassificationResult(db.Model, TimestampMixin):
    __tablename__ = "classification_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    forecast_id: Mapped[int | None] = mapped_column(ForeignKey("forecast_results.id", ondelete="SET NULL"))
    schema_version: Mapped[str] = mapped_column(db.String(16), default="v1", nullable=False)
    model_type: Mapped[str] = mapped_column(Enum("xgboost", name="classification_model_type"), default="xgboost", nullable=False)
    window_role: Mapped[str] = mapped_column(Enum("current", "future", name="window_role"), default="current", nullable=False)
    predicted_label: Mapped[str] = mapped_column(db.String(32), nullable=False)
    confidence: Mapped[float | None]
    probabilities: Mapped[dict | None] = mapped_column(JSON)
    explanation: Mapped[str | None] = mapped_column(Text)
    sample_id: Mapped[str | None] = mapped_column(db.String(128))
    runtime_library: Mapped[str | None] = mapped_column(db.String(64))
    window_start: Mapped[datetime | None]
    window_end: Mapped[datetime | None]


class DetectionResult(db.Model, TimestampMixin):
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


class ChatSession(db.Model):
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


class ChatMessage(db.Model):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[str] = mapped_column(Enum("user", "assistant", "system", name="chat_role"), nullable=False)
    content: Mapped[str | None] = mapped_column(Text)
    assistant_payload: Mapped[dict | None] = mapped_column(JSON)
    content_path: Mapped[str | None] = mapped_column(db.String(512))
    model_name: Mapped[str | None] = mapped_column(db.String(128))
    tokens_used: Mapped[int | None]
    created_at: Mapped[datetime] = mapped_column(default=datetime.now, nullable=False)

    session: Mapped[ChatSession] = relationship(back_populates="messages")


class Report(db.Model, TimestampMixin):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    report_type: Mapped[str] = mapped_column(Enum("pdf", name="report_type"), default="pdf", nullable=False)
    file_path: Mapped[str] = mapped_column(db.String(512), nullable=False)
    file_size: Mapped[int] = mapped_column(default=0, nullable=False)
