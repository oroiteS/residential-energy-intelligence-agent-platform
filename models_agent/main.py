"""Robyn 服务入口。"""

from __future__ import annotations

from app.bootstrap import create_app
from app.config import load_settings


def main() -> None:
    settings = load_settings()
    application = create_app(settings)
    application.start(host=settings.host, port=settings.port, _check_port=False)


if __name__ == "__main__":
    main()
