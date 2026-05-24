from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from flask import current_app

from app.errors import NotFoundError, ServiceUnavailableError
from app.extensions import db
from app.models import Report
from app.services.analysis_service import get_analysis_payload
from app.services.classification_service import get_latest_classification
from app.services.common import ensure_parent, read_json, to_iso
from app.services.dataset_service import get_dataset_or_404
from app.services.detection_service import get_latest_detection_record, detection_dto
from app.services.forecast_service import get_latest_forecast
from app.services.llm_client import can_use_llm, create_chat_model, llm_log_target, unavailable_reason


def list_reports(dataset_id: int) -> list[dict]:
    """列出某个数据集已经生成的报告。"""

    records = Report.query.filter_by(dataset_id=dataset_id).order_by(Report.created_at.desc()).all()
    return [report_dto(item) for item in records]


def export_report(dataset_id: int) -> dict:
    """导出数据集 PDF 分析报告。

    整体流程：
    1. 汇总数据集、分析、分类、检测和预测上下文；
    2. 优先调用 LLM 生成报告摘要，失败时使用本地摘要；
    3. 组装 Markdown 报告；
    4. 调用 md2pdf 脚本渲染 PDF；
    5. 将报告文件记录写入数据库。
    """

    dataset = get_dataset_or_404(dataset_id)
    current_app.logger.info("[report] 开始导出报告 dataset_id=%s name=%s", dataset.id, dataset.name)

    # 报告生成先构造结构化上下文。
    # LLM 和本地降级摘要都只基于这份上下文，避免生成无法追溯的数据。
    context = _build_report_context(dataset_id)
    summary = _summarize_report_with_llm(context)
    markdown = _build_markdown_report(context, summary)

    # Markdown 和 PDF 分开保存。
    # Markdown 便于排查报告内容，PDF 是最终给用户下载的交付物。
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    stem = f"report_{dataset.id}_{timestamp}"
    markdown_path = _absolute_storage_path(current_app.config["REPORT_MARKDOWN_DIR"] / f"{stem}.md")
    file_path = _absolute_storage_path(current_app.config["REPORT_DIR"] / f"{stem}.pdf")
    ensure_parent(markdown_path)
    ensure_parent(file_path)
    markdown_path.write_text(markdown, encoding="utf-8")

    _render_markdown_to_pdf(
        markdown_path=markdown_path,
        pdf_path=file_path,
        title=summary["title"],
    )

    if not file_path.exists():
        raise ServiceUnavailableError("PDF 渲染完成但未产出文件", code="PDF_FILE_MISSING")

    # PDF 生成成功后才写数据库记录，避免前端看到不存在的报告。
    report = Report(
        dataset_id=dataset.id,
        report_type="pdf",
        file_path=str(file_path),
        file_size=file_path.stat().st_size,
    )
    db.session.add(report)
    db.session.commit()
    current_app.logger.info(
        "[report] 报告导出完成 dataset_id=%s report_id=%s file=%s size=%s",
        dataset.id,
        report.id,
        file_path,
        report.file_size,
    )
    return report_dto(report)


def get_report_file(report_id: int) -> Path:
    """定位报告文件下载路径。"""

    report = Report.query.get(report_id)
    if report is None:
        raise NotFoundError("报告不存在", code="REPORT_NOT_FOUND")

    path = _resolve_report_path(report.file_path)
    if not path.exists():
        raise NotFoundError("报告文件不存在，请重新导出报告", code="REPORT_FILE_NOT_FOUND")
    return path


def report_dto(record: Report) -> dict:
    """转换报告记录为接口返回结构。"""

    return {
        "id": record.id,
        "dataset_id": record.dataset_id,
        "report_type": record.report_type,
        "file_path": record.file_path,
        "file_size": record.file_size,
        "created_at": to_iso(record.created_at),
    }


def _build_report_context(dataset_id: int) -> dict[str, Any]:
    """构造报告生成所需的结构化上下文。

    上下文包含数据集元信息、分析结果、当前分类、当前检测、最新预测和未来检测。
    这是报告内容唯一的数据来源。
    """

    dataset = get_dataset_or_404(dataset_id)
    analysis = get_analysis_payload(dataset_id)
    classification = get_latest_classification(dataset_id)
    forecast = get_latest_forecast(dataset_id)
    current_detection = detection_dto(get_latest_detection_record(dataset_id, window_role="current"))
    future_detection = detection_dto(get_latest_detection_record(dataset_id, window_role="future"))
    forecast_detail = read_json(forecast.detail_path, default={}) if forecast else {}

    return {
        "dataset": {
            "id": dataset.id,
            "name": dataset.name,
            "description": dataset.description,
            "household_id": dataset.household_id,
            "time_start": to_iso(dataset.time_start),
            "time_end": to_iso(dataset.time_end),
            "row_count": dataset.row_count,
            "source_granularity_minutes": dataset.source_granularity_minutes,
            "quality_summary": dataset.quality_summary or {},
        },
        "analysis": analysis,
        "classification": classification,
        "current_detection": current_detection,
        "forecast": {
            "id": forecast.id,
            "forecast_start": to_iso(forecast.forecast_start),
            "forecast_end": to_iso(forecast.forecast_end),
            "summary": forecast.summary or {},
            "series": forecast_detail.get("series", []),
        } if forecast else None,
        "future_detection": future_detection,
        "peak_valley_config": analysis.get("peak_valley_config", {}),
        "generated_at": datetime.now().astimezone().isoformat(),
    }


def _summarize_report_with_llm(context: dict[str, Any]) -> dict[str, Any]:
    """使用 LLM 生成报告摘要，失败时回退到本地摘要。

    这里的 LangChain 用法比智能问答更简单：
    没有历史消息，只把报告上下文填入 ChatPromptTemplate，
    再通过 prompt | ChatOpenAI 发送给大模型。
    """

    fallback = _fallback_summary(context, error_reason=unavailable_reason())
    if not can_use_llm():
        current_app.logger.info("[llm][report] 未调用大模型，使用本地报告摘要 reason=%s", fallback["error_reason"])
        return fallback

    try:
        from langchain_core.prompts import ChatPromptTemplate

        # Prompt 明确限制只能基于结构化数据写报告。
        # 这能降低大模型编造设备原因、夸大风险或输出非 JSON 的概率。
        # system 消息规定报告写作边界；human 消息给出具体输出格式和 context_json。
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "你是居民用电分析报告撰写助手。"
                        "只能基于给定结构化数据整理报告，禁止编造不存在的数据。"
                        "不要写实现细节，不要提接口、后端、降级、数据库。"
                        "不要给出设备级确定性归因，只能给出复核方向。"
                        "谷时用电占比高或峰时占比低本身不是坏习惯，不能据此提示用户减少谷时用电。"
                        "只有在峰时占比偏高时，才可以建议把可转移的大功率用电调整到谷时段。"
                        "如果问题是谷时绝对用电量持续偏高，只能建议复核夜间常开设备、待机用电或持续运行负荷。"
                        "必须只输出 JSON，包含 title、overview、sections、recommendations。"
                    ),
                ),
                (
                    "human",
                    (
                        "请把下面的数据整理成适合 PDF 报告的摘要。\n"
                        "要求：\n"
                        "1. title 不超过 24 个中文字符。\n"
                        "2. overview 用 2 到 4 句话概括结论。\n"
                        "3. sections 是数组，固定输出 5 个章节：数据概览、当前行为、异常复核、未来预测、行动建议。\n"
                        "4. 每个 section 只包含 title 和 body，body 要清晰、克制、可回溯。\n"
                        "5. recommendations 输出 3 到 5 条可执行建议。\n"
                        "结构化数据：{context_json}"
                    ),
                ),
            ]
        )
        target = llm_log_target()
        started_at = perf_counter()
        current_app.logger.info(
            "[llm][report] 正在发送给 %s model=%s dataset_id=%s",
            target["base_url"],
            target["model"],
            context["dataset"]["id"],
        )

        # prompt | model 是 LangChain 的链式调用写法。
        # invoke 传入 context_json 后，LangChain 会先渲染 prompt，再调用 ChatOpenAI。
        # 返回值通常是 AIMessage，正文在 content 字段中。
        raw = (prompt | create_chat_model(timeout_seconds=current_app.config["LLM_REPORT_TIMEOUT_SECONDS"])).invoke(
            {"context_json": json.dumps(context, ensure_ascii=False, default=str)}
        )
        elapsed_ms = (perf_counter() - started_at) * 1000
        current_app.logger.info(
            "[llm][report] 大模型回复完成 model=%s duration_ms=%.1f",
            target["model"],
            elapsed_ms,
        )

        # LLM 输出会先解析为 JSON，再经过业务安全清洗。
        # 清洗主要用于修正与峰谷业务口径冲突的建议。
        parsed = _sanitize_summary(
            _parse_summary_payload(getattr(raw, "content", str(raw)), fallback),
            context,
        )
        parsed["degraded"] = False
        parsed["error_reason"] = None
        return parsed
    except Exception as exc:
        target = llm_log_target()
        current_app.logger.exception(
            "[llm][report] 发送失败 base_url=%s model=%s error=%s",
            target["base_url"],
            target["model"],
            exc,
        )
        return _fallback_summary(context, error_reason="LLM_REQUEST_FAILED")


def _parse_summary_payload(raw: str, fallback: dict[str, Any]) -> dict[str, Any]:
    """解析 LLM 返回的报告摘要 JSON。

    报告生成要求模型返回 title、overview、sections、recommendations。
    如果模型没有严格返回 JSON 或字段缺失，则回退到本地摘要中的对应内容。
    """

    text = _extract_json_text(raw)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return fallback

    title = str(parsed.get("title") or fallback["title"]).strip()[:40]
    overview = str(parsed.get("overview") or fallback["overview"]).strip()
    sections = []
    for item in parsed.get("sections") or []:
        if not isinstance(item, dict):
            continue
        section_title = str(item.get("title") or "").strip()
        body = str(item.get("body") or "").strip()
        if section_title and body:
            sections.append({"title": section_title, "body": body})

    recommendations = []
    for item in parsed.get("recommendations") or []:
        cleaned = str(item).strip()
        if cleaned and cleaned not in recommendations:
            recommendations.append(cleaned)

    return {
        "title": title or fallback["title"],
        "overview": overview or fallback["overview"],
        "sections": sections or fallback["sections"],
        "recommendations": recommendations[:5] or fallback["recommendations"],
        "degraded": fallback["degraded"],
        "error_reason": fallback["error_reason"],
    }


def _sanitize_summary(summary: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """清洗报告摘要中的不合理峰谷建议。"""

    peak_ratio = _safe_float(context.get("analysis", {}).get("summary", {}).get("peak_ratio"))
    if peak_ratio is None:
        return summary

    # 峰时占比不高时，禁止输出“峰时占比偏低，所以转移到谷时段”这类自相矛盾建议。
    if peak_ratio < 0.5:
        replacements = {
            "峰时用电占比偏低，可考虑将部分大功率设备调整至谷时段使用，以优化用电结构。":
                "谷时用电占比较高本身不属于不良习惯；如果谷时绝对用电量持续偏高，建议复核夜间常开设备、待机用电或其他持续运行负荷。",
            "峰时占比偏低，可考虑将部分大功率设备调整至谷时段使用":
                "谷时用电占比较高本身不属于不良习惯，建议重点复核谷时段绝对用电量持续偏高的原因",
        }
        for old, new in replacements.items():
            summary["overview"] = str(summary.get("overview", "")).replace(old, new)
            for section in summary.get("sections", []):
                section["body"] = str(section.get("body", "")).replace(old, new)
            summary["recommendations"] = [
                str(item).replace(old, new) for item in summary.get("recommendations", [])
            ]

    return summary


def _fallback_summary(context: dict[str, Any], *, error_reason: str) -> dict[str, Any]:
    """构造不依赖 LLM 的本地报告摘要。

    当 LLM 配置缺失、依赖缺失或请求失败时使用该摘要；
    这样报告导出能力不会完全依赖外部模型服务。
    """

    dataset = context["dataset"]
    analysis_summary = context["analysis"]["summary"]
    classification = context.get("classification")
    forecast = context.get("forecast")
    detection = context.get("current_detection")

    title = f"居民用电分析报告 - {dataset['name']}"
    overview_parts = [
        f"本报告覆盖 {dataset.get('time_start') or '未知'} 至 {dataset.get('time_end') or '未知'} 的用电数据。",
        f"统计期内总用电量为 {analysis_summary.get('total_kwh')} kWh，日均用电量为 {analysis_summary.get('daily_avg_kwh')} kWh。",
    ]
    if classification:
        overview_parts.append(f"最近窗口行为类型为 {classification.get('predicted_label')}。")
    if forecast:
        overview_parts.append(f"未来 7 天预测总用电量约为 {forecast['summary'].get('predicted_total_kwh')} kWh。")

    sections = [
        {"title": "数据概览", "body": " ".join(overview_parts[:2])},
        {
            "title": "当前行为",
            "body": (
                f"当前行为类型为 {classification.get('predicted_label')}，"
                f"置信度约为 {_percent(classification.get('confidence'))}。"
            ) if classification else "当前尚未形成行为分类结果。",
        },
        {
            "title": "异常复核",
            "body": _detection_text(detection),
        },
        {
            "title": "未来预测",
            "body": (
                f"预测区间为 {forecast.get('forecast_start')} 至 {forecast.get('forecast_end')}，"
                f"预测总量约为 {forecast['summary'].get('predicted_total_kwh')} kWh。"
            ) if forecast else "当前尚未生成未来用电预测。",
        },
        {
            "title": "行动建议",
            "body": "建议结合峰谷时段、近期偏离项和未来预测结果安排用电，并对持续偏高时段进行复核。",
        },
    ]
    return {
        "title": title[:40],
        "overview": " ".join(overview_parts),
        "sections": sections,
        "recommendations": [
            "优先关注连续多日用电偏高的时段。",
            "将可调整的用电尽量安排在谷时段。",
            "若夜间持续用电偏高，建议复核常开设备或待机用电。",
        ],
        "degraded": True,
        "error_reason": error_reason,
    }


def _build_markdown_report(context: dict[str, Any], summary: dict[str, Any]) -> str:
    """将结构化上下文和摘要组装为 Markdown 报告。"""

    dataset = context["dataset"]
    analysis = context["analysis"]["summary"]
    forecast = context.get("forecast")
    lines = [
        f"# {summary['title']}",
        "",
        f"> 生成时间：{_fmt_dt(context['generated_at'])}",
        "",
        "## 报告摘要",
        "",
        summary["overview"],
        "",
        "## 核心指标",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 数据集 | {dataset['name']} |",
        f"| 时间范围 | {_fmt_dt(dataset.get('time_start'))} 至 {_fmt_dt(dataset.get('time_end'))} |",
        f"| 原始记录数 | {dataset.get('row_count') or 0} |",
        f"| 总用电量 | {_fmt_number(analysis.get('total_kwh'))} kWh |",
        f"| 日均用电量 | {_fmt_number(analysis.get('daily_avg_kwh'))} kWh |",
        f"| 峰时用电占比 | {_percent(analysis.get('peak_ratio'))} |",
        f"| 谷时用电占比 | {_percent(analysis.get('valley_ratio'))} |",
        f"| 最高用电功率 | {_fmt_number(analysis.get('max_load_w'))} W |",
        "",
    ]

    if forecast:
        # 预测明细最多展示未来 7 天。
        # 这与当前 LSTM Direct 模型固定 7 天预测窗口保持一致。
        lines.extend(
            [
                "## 未来 7 天预测明细",
                "",
                "| 日期 | 总用电量 kWh | 峰时 kWh | 谷时 kWh |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for item in forecast.get("series", [])[:7]:
            lines.append(
                "| {date} | {total} | {peak} | {valley} |".format(
                    date=item.get("date", ""),
                    total=_fmt_number(item.get("predicted_total_kwh", item.get("total_kwh"))),
                    peak=_fmt_number(item.get("predicted_peak_kwh", item.get("peak_kwh"))),
                    valley=_fmt_number(item.get("predicted_valley_kwh", item.get("valley_kwh"))),
                )
            )
        lines.append("")

    for section in summary["sections"]:
        lines.extend(["## " + section["title"], "", section["body"], ""])

    lines.extend(["## 建议清单", ""])
    for item in summary["recommendations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _render_markdown_to_pdf(*, markdown_path: Path, pdf_path: Path, title: str) -> None:
    """调用内置 md2pdf 脚本将 Markdown 渲染为 PDF。"""

    script_path = _resolve_md2pdf_script()
    current_app.logger.info("[report] 开始渲染 PDF markdown=%s output=%s", markdown_path, pdf_path)

    # 先渲染到临时目录，再复制到最终路径。
    # 这样可以避免渲染失败时留下半成品 PDF。
    with tempfile.TemporaryDirectory(prefix="md2pdf_", dir=str(current_app.config["REPORT_DIR"])) as temp_dir:
        temp_pdf = Path(temp_dir) / "report.pdf"
        command = [
            sys.executable,
            str(script_path),
            "--input",
            str(markdown_path),
            "--output",
            str(temp_pdf),
            "--title",
            title,
            "--author",
            "居民用电分析系统",
            "--date",
            datetime.now().strftime("%Y-%m-%d"),
            "--theme",
            current_app.config["PDF_THEME"],
            "--cover",
            str(bool(current_app.config["PDF_COVER"])),
            "--toc",
            str(bool(current_app.config["PDF_TOC"])),
            "--header-title",
            title,
            "--footer-left",
            "居民用电分析系统",
        ]
        try:
            started_at = perf_counter()
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=current_app.config["PDF_RENDER_TIMEOUT_SECONDS"],
            )
        except subprocess.TimeoutExpired as exc:
            current_app.logger.exception("[report] PDF 渲染超时 output=%s", pdf_path)
            raise ServiceUnavailableError("PDF 生成超时", code="PDF_RENDER_TIMEOUT") from exc

        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "").strip()
            current_app.logger.error("[report] PDF 渲染失败 output=%s error=%s", pdf_path, message)
            if "No module named 'reportlab'" in message or "No module named reportlab" in message:
                raise ServiceUnavailableError("缺少 reportlab 依赖，请在 backend 下执行 uv sync", code="REPORTLAB_MISSING")
            raise ServiceUnavailableError(message or "PDF 生成失败", code="PDF_RENDER_FAILED")
        if not temp_pdf.exists():
            raise ServiceUnavailableError("PDF 生成失败，未产出文件", code="PDF_RENDER_FAILED")
        pdf_path.write_bytes(temp_pdf.read_bytes())
        elapsed_ms = (perf_counter() - started_at) * 1000
        current_app.logger.info("[report] PDF 渲染完成 output=%s duration_ms=%.1f", pdf_path, elapsed_ms)


def _resolve_md2pdf_script() -> Path:
    """解析 md2pdf 脚本路径。"""

    configured = Path(current_app.config["MD2PDF_SCRIPT_PATH"])
    candidates = [configured]
    if not configured.is_absolute():
        candidates.append(Path(current_app.root_path).parent / configured)
        candidates.append(Path(current_app.root_path) / configured)
    for item in candidates:
        if item.exists():
            return item.resolve()
    raise ServiceUnavailableError("未找到内置 md2pdf 脚本", code="MD2PDF_SCRIPT_MISSING")


def _resolve_report_path(value: str) -> Path:
    """兼容绝对路径和相对路径形式的报告文件位置。"""

    path = Path(value)
    if path.is_absolute():
        return path
    backend_root = Path(current_app.root_path).parent
    storage_root = Path(current_app.config["STORAGE_ROOT"])
    if path.parts and path.parts[0] == storage_root.name:
        candidates = [backend_root / path, Path(current_app.root_path) / path]
    else:
        candidates = [storage_root / path, backend_root / path, Path(current_app.root_path) / path]
    for item in candidates:
        if item.exists():
            return item.resolve()
    return candidates[0].resolve()


def _absolute_storage_path(path: Path) -> Path:
    """将存储路径转换为绝对路径。"""

    return path if path.is_absolute() else (Path(current_app.root_path).parent / path).resolve()


def _extract_json_text(raw: str) -> str:
    """从 LLM 回复中提取 JSON 文本。"""

    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _detection_text(detection: dict | None) -> str:
    """将异常检测结果转换为报告中的一句解释文本。"""

    if not detection:
        return "当前尚未生成异常检测结果。"
    status = "建议复核" if detection.get("is_anomaly") else "未触发复核"
    reasons = detection.get("reasons") or []
    reason_text = "；".join(str(item.get("message") or item) for item in reasons[:3]) if reasons else "未发现明显偏离项"
    return f"当前状态为{status}，偏离分数为 {_fmt_number(detection.get('anomaly_score'))}。{reason_text}。"


def _fmt_number(value: Any) -> str:
    """格式化数值，缺失或非法时显示“暂无”。"""

    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "暂无"


def _safe_float(value: Any) -> float | None:
    """安全转换浮点数，失败时返回 None。"""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percent(value: Any) -> str:
    """将比例或百分数格式化为百分比文本。"""

    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return "暂无"
    if ratio <= 1:
        ratio *= 100
    return f"{ratio:.2f}%"


def _fmt_dt(value: Any) -> str:
    """格式化日期时间文本用于报告展示。"""

    if not value:
        return "暂无"
    text = str(value)
    return text.replace("T", " ")[:19]
