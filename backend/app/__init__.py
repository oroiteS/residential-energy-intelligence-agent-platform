from __future__ import annotations

from pathlib import Path

from flask import Flask
from flask_cors import CORS

from app.api import register_error_handlers, register_hooks
from app.extensions import db
from config import Config


def create_app(config_class: type[Config] = Config) -> Flask:
    """创建并配置 Flask 应用实例。

    这是后端的应用工厂函数，负责完成配置注入、跨域支持、数据库绑定、
    存储目录准备、请求钩子、错误处理和业务路由注册。
    """

    app = Flask(__name__)
    app.config.from_object(config_class)

    # 初始化跨域和数据库扩展。
    # 前端会通过浏览器访问 API，因此这里开启 supports_credentials 支持携带凭据。
    CORS(app, supports_credentials=True)
    db.init_app(app)

    # 按顺序完成应用启动依赖。
    # 目录先创建，随后注册请求日志、错误响应和各业务蓝图。
    _ensure_storage_dirs(app)
    register_hooks(app)
    register_error_handlers(app)
    _register_blueprints(app)

    return app


def _ensure_storage_dirs(app: Flask) -> None:
    """确保后端运行所需的所有存储目录存在。

    上传文件、规范化数据、分析结果、预测结果和报告都会落盘；
    这里在应用启动阶段统一解析相对路径并创建目录。
    """

    # 统一处理文件产物目录。
    # 如果配置里是相对路径，则以 backend 目录为基准转换为绝对路径。
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

    # md2pdf 是独立脚本路径，也需要转换为绝对路径，避免工作目录变化导致找不到脚本。
    script_path = Path(app.config["MD2PDF_SCRIPT_PATH"])
    if not script_path.is_absolute():
        script_path = Path(app.root_path).parent / script_path
    app.config["MD2PDF_SCRIPT_PATH"] = script_path.resolve()


def _register_blueprints(app: Flask) -> None:
    """注册所有业务路由蓝图。

    路由按“健康检查、系统配置、数据集、分析、模型结果、报告、对话、智能体”的顺序挂载，
    所有接口共享 Config.API_PREFIX 中的版本前缀。
    """

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

    # 所有蓝图使用同一个版本前缀，例如 /api/v1/datasets。
    # 这样前端只需要维护一套基础 API 地址。
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
