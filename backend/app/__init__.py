from __future__ import annotations

from pathlib import Path

from flask import Flask
from flask_cors import CORS

from app.api import register_error_handlers, register_hooks
from app.extensions import db
from config import Config


def create_app(config_class: type[Config] = Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_class)

    CORS(app, supports_credentials=True)
    db.init_app(app)

    _ensure_storage_dirs(app)
    register_hooks(app)
    register_error_handlers(app)
    _register_blueprints(app)

    return app


def _ensure_storage_dirs(app: Flask) -> None:
    for key in [
        "STORAGE_ROOT",
        "UPLOAD_DIR",
        "NORMALIZED_DIR",
        "DAILY_DIR",
        "QUALITY_DIR",
        "ANALYSIS_DIR",
        "FORECAST_DIR",
        "REPORT_DIR",
        "REPORT_MARKDOWN_DIR",
    ]:
        path = Path(app.config[key])
        if not path.is_absolute():
            path = Path(app.root_path).parent / path
        path = path.resolve()
        path.mkdir(parents=True, exist_ok=True)
        app.config[key] = path

    script_path = Path(app.config["MD2PDF_SCRIPT_PATH"])
    if not script_path.is_absolute():
        script_path = Path(app.root_path).parent / script_path
    app.config["MD2PDF_SCRIPT_PATH"] = script_path.resolve()


def _register_blueprints(app: Flask) -> None:
    from app.routes.agent import agent_bp
    from app.routes.analysis import analysis_bp
    from app.routes.chat import chat_bp
    from app.routes.classifications import classifications_bp
    from app.routes.datasets import datasets_bp
    from app.routes.detections import detections_bp
    from app.routes.forecasts import forecasts_bp
    from app.routes.health import health_bp
    from app.routes.reports import reports_bp
    from app.routes.system import system_bp

    prefix = app.config["API_PREFIX"]
    for blueprint in [
        health_bp,
        system_bp,
        datasets_bp,
        analysis_bp,
        classifications_bp,
        detections_bp,
        forecasts_bp,
        reports_bp,
        chat_bp,
        agent_bp,
    ]:
        app.register_blueprint(blueprint, url_prefix=prefix)
