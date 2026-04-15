"""Agent 工作流最小测试。"""

from __future__ import annotations

import sys
from pathlib import Path
import unittest


MODELS_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(MODELS_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_AGENT_ROOT))


from app.agent.advice_planner import AdvicePlanner
from app.agent.evidence_builder import EvidenceBuilder
from app.agent.intent_router import IntentRouter
from app.agent.memory import ShortTermMemoryManager
from app.agent.report_builder import ReportBuilder
from app.agent.response_renderer import ResponseRenderer
from app.agent.state import AgentIntent
from app.agent.workflow import AgentWorkflow, ReportWorkflow
from app.contracts import AgentAskRequest, AgentReportSummaryRequest
from app.errors import ValidationError


def build_agent_workflow() -> AgentWorkflow:
    return AgentWorkflow(
        memory_manager=ShortTermMemoryManager(),
        intent_router=IntentRouter(),
        evidence_builder=EvidenceBuilder(),
        advice_planner=AdvicePlanner(),
        response_renderer=ResponseRenderer(),
    )


def build_context_payload() -> dict:
    return {
        "dataset": {"name": "House 2"},
        "analysis_summary": {
            "peak_ratio": 0.42,
            "daily_avg_kwh": 14.6,
        },
        "classification_result": {
            "schema_version": "v1",
            "model_type": "xgboost",
            "predicted_label": "day_low_night_high",
            "confidence": 0.88,
            "probabilities": {
                "day_high_night_low": 0.05,
                "day_low_night_high": 0.88,
                "all_day_high": 0.04,
                "all_day_low": 0.03,
            },
        },
        "forecast_summary": {
            "schema_version": "v1",
            "model_type": "transformer",
            "forecast_horizon": "1d",
            "peak_period": "19:00-22:00",
            "predicted_avg_load_w": 520.0,
            "predicted_peak_load_w": 1680.0,
            "predicted_total_kwh": 12.48,
            "risk_flags": ["evening_peak"],
            "confidence_hint": "medium",
        },
        "recent_history_summary": {
            "avg_active_appliance_count": 2.6,
            "avg_burst_event_count": 1.2,
        },
        "rule_advices": [
            {
                "key": "shift_peak_tasks",
                "action": "把洗衣和热水任务尽量移到晚高峰之后。",
                "summary": "高峰任务后移",
                "reason": "晚间峰值窗口明确，先错峰收益更直接。",
                "priority": 92,
                "category": "forecast",
            }
        ],
        "user_preferences": {"objective": "优先降低晚间成本"},
    }


class AgentWorkflowTestCase(unittest.TestCase):
    def test_context_should_be_normalized_with_default_blocks(self) -> None:
        request = AgentAskRequest.from_dict(
            {
                "dataset_id": 2,
                "session_id": 1000,
                "question": "今天是什么类型？",
                "history": [],
                "context": {
                    "classification_result": {
                        "predicted_label": "day_low_night_high",
                    }
                },
            }
        )

        self.assertIn("forecast_summary", request.context)
        self.assertIn("rule_advices", request.context)
        self.assertEqual(request.context["rule_advices"], [])
        self.assertEqual(request.context["forecast_summary"]["schema_version"], "v1")
        self.assertEqual(request.context["classification_result"]["model_type"], "xgboost")
        self.assertEqual(
            request.context["classification_result"]["label_display_name"],
            "白天低晚上高型",
        )

    def test_schema_fields_should_be_derived_and_preserved(self) -> None:
        request = AgentAskRequest.from_dict(
            {
                "dataset_id": 2,
                "session_id": 1006,
                "question": "今天是什么类型？",
                "history": [],
                "context": {
                    "classification_result": {
                        "predicted_label": "day_low_night_high",
                        "probabilities": {
                            "day_high_night_low": 0.05,
                            "day_low_night_high": 0.88,
                            "all_day_high": 0.04,
                            "all_day_low": 0.03,
                        },
                    },
                    "forecast_summary": {
                        "predicted_peak_load_w": 1680.0,
                    },
                },
            }
        )

        self.assertEqual(request.context["classification_result"]["schema_version"], "v1")
        self.assertEqual(request.context["classification_result"]["confidence"], 0.88)
        self.assertEqual(request.context["forecast_summary"]["model_type"], "transformer")
        self.assertEqual(request.context["forecast_summary"]["forecast_horizon"], "1d")

    def test_prepare_advice_should_route_and_rank_actions(self) -> None:
        workflow = build_agent_workflow()
        request = AgentAskRequest.from_dict(
            {
                "dataset_id": 2,
                "session_id": 1001,
                "question": "明天我应该怎么省电？",
                "history": [],
                "context": build_context_payload(),
            }
        )

        prepared = workflow.prepare(request)

        self.assertEqual(prepared.intent, AgentIntent.ADVICE)
        self.assertTrue(prepared.advice_candidates)
        self.assertIn("晚高峰之后", prepared.advice_candidates[0].action)
        self.assertGreaterEqual(len(prepared.evidence), 4)

    def test_follow_up_question_should_reuse_session_intent(self) -> None:
        workflow = build_agent_workflow()
        first_request = AgentAskRequest.from_dict(
            {
                "dataset_id": 2,
                "session_id": 1002,
                "question": "明天我应该怎么省电？",
                "history": [],
                "context": build_context_payload(),
            }
        )
        first_prepared = workflow.prepare(first_request)
        workflow.commit(
            session_id=first_request.session_id,
            intent=first_prepared.intent,
            question=first_request.question,
            actions=first_prepared.fallback_output.actions,
            missing_information=first_prepared.fallback_output.missing_information,
            context=first_prepared.context,
        )

        second_request = AgentAskRequest.from_dict(
            {
                "dataset_id": 2,
                "session_id": 1002,
                "question": "那我先看什么？",
                "history": [],
                "context": build_context_payload(),
            }
        )
        second_prepared = workflow.prepare(second_request)

        self.assertEqual(second_prepared.intent, AgentIntent.ADVICE)
        self.assertTrue(second_prepared.memory.recent_questions)

    def test_missing_forecast_should_expose_missing_information(self) -> None:
        workflow = build_agent_workflow()
        context = build_context_payload()
        context["forecast_summary"] = {}
        request = AgentAskRequest.from_dict(
            {
                "dataset_id": 2,
                "session_id": 1003,
                "question": "明天有什么风险？",
                "history": [],
                "context": context,
            }
        )

        prepared = workflow.prepare(request)
        missing_keys = {item.key for item in prepared.missing_information}

        self.assertEqual(prepared.intent, AgentIntent.RISK)
        self.assertIn("forecast_summary", missing_keys)

    def test_invalid_context_shape_should_raise_validation_error(self) -> None:
        with self.assertRaises(ValidationError):
            AgentAskRequest.from_dict(
                {
                    "dataset_id": 2,
                    "session_id": 1004,
                    "question": "明天有什么风险？",
                    "history": [],
                    "context": {
                        "forecast_summary": [],
                    },
                }
            )

        with self.assertRaises(ValidationError):
            AgentAskRequest.from_dict(
                {
                    "dataset_id": 2,
                    "session_id": 1006,
                    "question": "今天是什么类型？",
                    "history": [],
                    "context": {
                        "classification_result": {
                            "predicted_label": "unknown_label",
                        },
                    },
                }
            )

        with self.assertRaises(ValidationError):
            AgentAskRequest.from_dict(
                {
                    "dataset_id": 2,
                    "session_id": 1007,
                    "question": "明天怎么样？",
                    "history": [],
                    "context": {
                        "forecast_summary": {
                            "risk_flags": ["made_up_flag"],
                        },
                    },
                }
            )

        with self.assertRaises(ValidationError):
            AgentAskRequest.from_dict(
                {
                    "dataset_id": 2,
                    "session_id": 1005,
                    "question": "给我建议。",
                    "history": [],
                    "context": {
                        "rule_advices": [123],
                    },
                }
            )


class ReportWorkflowTestCase(unittest.TestCase):
    def test_report_sections_should_keep_fixed_order(self) -> None:
        workflow = ReportWorkflow(
            evidence_builder=EvidenceBuilder(),
            advice_planner=AdvicePlanner(),
            report_builder=ReportBuilder(),
        )
        request = AgentReportSummaryRequest.from_dict(
            {
                "dataset_id": 2,
                "context": build_context_payload(),
            }
        )

        prepared = workflow.prepare(request)
        self.assertEqual(
            [item.title for item in prepared.fallback_output.sections],
            ["总体概览", "行为判断", "预测风险", "附注"],
        )


if __name__ == "__main__":
    unittest.main()
