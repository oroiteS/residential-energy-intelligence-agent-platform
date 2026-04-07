"""Markdown 转 PDF 服务。"""

from __future__ import annotations

import base64
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from app.config import MODELS_AGENT_ROOT, Settings
from app.contracts import PDFRenderRequest
from app.errors import ServiceUnavailableError


class PDFService:
    """调用 vendored md2pdf 脚本生成 PDF。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def render_pdf(self, request: PDFRenderRequest) -> dict[str, Any]:
        script_path = self.settings.md2pdf_script_path
        if not script_path.exists():
            raise ServiceUnavailableError("MD2PDF_SCRIPT_MISSING", "未找到内置 md2pdf 脚本")

        temp_root = MODELS_AGENT_ROOT / ".tmp" / "pdf"
        temp_root.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="render_", dir=temp_root) as temp_dir:
            temp_path = Path(temp_dir)
            markdown_path = temp_path / "report.md"
            pdf_path = temp_path / "report.pdf"
            markdown_path.write_text(request.markdown, encoding="utf-8")

            command = [
                sys.executable,
                str(script_path),
                "--input",
                str(markdown_path),
                "--output",
                str(pdf_path),
                "--title",
                request.title,
                "--author",
                request.author or "居民能源智能分析平台",
                "--date",
                request.date,
                "--theme",
                request.theme or self.settings.pdf_theme,
                "--cover",
                str(request.cover),
                "--toc",
                str(request.toc),
                "--header-title",
                request.header_title or request.title,
                "--footer-left",
                request.footer_left or "居民能源智能分析平台",
            ]

            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.settings.pdf_render_timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise ServiceUnavailableError("PDF_RENDER_TIMEOUT", "PDF 生成超时") from exc

            if completed.returncode != 0:
                error_message = (completed.stderr or completed.stdout or "").strip()
                if "No module named 'reportlab'" in error_message or "No module named reportlab" in error_message:
                    raise ServiceUnavailableError("REPORTLAB_MISSING", "models_agent 缺少 reportlab 依赖")
                raise ServiceUnavailableError(
                    "PDF_RENDER_FAILED",
                    error_message or "PDF 生成失败",
                )

            if not pdf_path.exists():
                raise ServiceUnavailableError("PDF_RENDER_FAILED", "PDF 生成失败，未产出文件")

            pdf_bytes = pdf_path.read_bytes()
            return {
                "file_name": "report.pdf",
                "content_type": "application/pdf",
                "pdf_base64": base64.b64encode(pdf_bytes).decode("ascii"),
                "file_size": len(pdf_bytes),
            }
