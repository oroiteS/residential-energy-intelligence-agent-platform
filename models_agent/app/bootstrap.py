"""Robyn 应用装配。"""

from __future__ import annotations

from typing import Any, Callable

from robyn import Request, Robyn

from app.config import Settings
from app.contracts import AgentAskRequest, BacktestRequest, ForecastRequest, PredictRequest
from app.errors import AppError
from app.http import error_response, load_request_json, success
from app.services.agent_service import AgentService
from app.services.classification_service import ClassificationService
from app.services.forecast_service import ForecastService


def _handle_request(handler: Callable[[], dict[str, Any]]):
    try:
        return success(handler())
    except AppError as error:
        return error_response(error)


def create_app(settings: Settings) -> Robyn:
    app = Robyn(__file__)

    classification_service = ClassificationService(settings)
    forecast_service = ForecastService(settings)
    agent_service = AgentService(settings)

    @app.get("/internal/model/v1/health")
    def model_health():
        return _handle_request(classification_service.health)

    @app.get("/internal/model/v1/model/info")
    def model_info():
        return _handle_request(classification_service.model_info)

    @app.post("/internal/model/v1/predict")
    def predict(request: Request):
        def run() -> dict[str, Any]:
            payload = load_request_json(request)
            predict_request = PredictRequest.from_dict(payload)
            return classification_service.predict(predict_request)

        return _handle_request(run)

    @app.post("/internal/model/v1/forecast")
    def forecast(request: Request):
        def run() -> dict[str, Any]:
            payload = load_request_json(request)
            forecast_request = ForecastRequest.from_dict(payload)
            return forecast_service.forecast(forecast_request)

        return _handle_request(run)

    @app.post("/internal/model/v1/backtest")
    def backtest(request: Request):
        def run() -> dict[str, Any]:
            payload = load_request_json(request)
            backtest_request = BacktestRequest.from_dict(payload)
            return forecast_service.backtest(backtest_request)

        return _handle_request(run)

    @app.get("/internal/agent/v1/health")
    def agent_health():
        return _handle_request(agent_service.health)

    @app.post("/internal/agent/v1/ask")
    def agent_ask(request: Request):
        def run() -> dict[str, Any]:
            payload = load_request_json(request)
            ask_request = AgentAskRequest.from_dict(payload)
            return agent_service.ask(ask_request)

        return _handle_request(run)

    return app
