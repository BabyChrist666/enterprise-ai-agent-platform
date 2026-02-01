"""
Tests for the evaluation framework itself.

Tests cover: metrics computation, dataset filtering, routing evaluation,
tool usage scoring, and response quality assessment.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from backend.evaluation.metrics import (
    EvaluationMetrics,
    CaseResult,
    RoutingResult,
    ToolUsageResult,
    ResponseQualityResult,
    LatencyResult,
    Verdict,
)
from backend.evaluation.datasets import (
    EvalDataset,
    EvalCase,
    ExpectedDomain,
    ROUTING_EVAL,
    TOOL_USAGE_EVAL,
    RESPONSE_QUALITY_EVAL,
    FULL_EVAL_SUITE,
)
from backend.evaluation.evaluator import AgentEvaluator


# --- Dataset Tests ---


class TestEvalDataset:
    def test_routing_dataset_has_cases(self):
        assert len(ROUTING_EVAL.cases) > 0

    def test_tool_usage_dataset_has_cases(self):
        assert len(TOOL_USAGE_EVAL.cases) > 0

    def test_response_quality_dataset_has_cases(self):
        assert len(RESPONSE_QUALITY_EVAL.cases) > 0

    def test_full_suite_combines_all(self):
        expected = (
            len(ROUTING_EVAL.cases)
            + len(TOOL_USAGE_EVAL.cases)
            + len(RESPONSE_QUALITY_EVAL.cases)
        )
        assert len(FULL_EVAL_SUITE.cases) == expected

    def test_filter_by_domain(self):
        finance_only = ROUTING_EVAL.filter_by_domain(ExpectedDomain.FINANCE)
        for case in finance_only.cases:
            assert case.expected_domain == ExpectedDomain.FINANCE

    def test_filter_by_difficulty(self):
        easy_only = ROUTING_EVAL.filter_by_difficulty("easy")
        for case in easy_only.cases:
            assert case.difficulty == "easy"

    def test_all_cases_have_ids(self):
        for case in FULL_EVAL_SUITE.cases:
            assert case.id, f"Case missing id: {case.query[:50]}"

    def test_no_duplicate_ids(self):
        ids = [case.id for case in FULL_EVAL_SUITE.cases]
        assert len(ids) == len(set(ids)), "Duplicate case IDs found"

    def test_multi_agent_cases_have_secondary_domains(self):
        for case in ROUTING_EVAL.cases:
            if case.requires_multi_agent:
                assert len(case.secondary_domains) >= 2, (
                    f"Multi-agent case {case.id} needs >=2 secondary domains"
                )


# --- Metrics Tests ---


class TestEvaluationMetrics:
    def _make_pass_result(self, case_id: str = "test_01") -> CaseResult:
        return CaseResult(
            case_id=case_id,
            query="test query",
            verdict=Verdict.PASS,
            routing=RoutingResult(
                case_id=case_id,
                expected_domain="finance",
                actual_domain="finance",
                expected_multi_agent=False,
                actual_multi_agent=False,
                routing_confidence=0.85,
                correct=True,
                multi_agent_correct=True,
            ),
            tool_usage=ToolUsageResult(
                case_id=case_id,
                expected_tools=["calculate_risk_metrics"],
                actual_tools=["calculate_risk_metrics"],
                tools_correct=True,
                precision=1.0,
                recall=1.0,
            ),
            response_quality=ResponseQualityResult(
                case_id=case_id,
                expected_keywords=["VaR", "Sharpe"],
                found_keywords=["VaR", "Sharpe"],
                missing_keywords=[],
                keyword_coverage=1.0,
                response_length=200,
                has_reasoning=True,
            ),
            latency=LatencyResult(
                case_id=case_id,
                total_ms=1500.0,
                routing_ms=50.0,
                agent_ms=1450.0,
                tool_calls=1,
            ),
        )

    def _make_fail_result(self, case_id: str = "test_02") -> CaseResult:
        return CaseResult(
            case_id=case_id,
            query="test query fail",
            verdict=Verdict.FAIL,
            routing=RoutingResult(
                case_id=case_id,
                expected_domain="legal",
                actual_domain="finance",
                expected_multi_agent=False,
                actual_multi_agent=False,
                routing_confidence=0.6,
                correct=False,
                multi_agent_correct=True,
            ),
        )

    def test_empty_metrics(self):
        m = EvaluationMetrics()
        assert m.total_cases == 0
        assert m.pass_rate == 0.0
        assert m.routing_accuracy == 0.0

    def test_pass_rate_all_pass(self):
        m = EvaluationMetrics()
        m.add_result(self._make_pass_result("t1"))
        m.add_result(self._make_pass_result("t2"))
        assert m.pass_rate == 1.0

    def test_pass_rate_mixed(self):
        m = EvaluationMetrics()
        m.add_result(self._make_pass_result("t1"))
        m.add_result(self._make_fail_result("t2"))
        assert m.pass_rate == 0.5

    def test_routing_accuracy(self):
        m = EvaluationMetrics()
        m.add_result(self._make_pass_result("t1"))
        m.add_result(self._make_fail_result("t2"))
        assert m.routing_accuracy == 0.5

    def test_tool_selection_accuracy(self):
        m = EvaluationMetrics()
        m.add_result(self._make_pass_result("t1"))
        assert m.tool_selection_accuracy == 1.0

    def test_avg_keyword_coverage(self):
        m = EvaluationMetrics()
        m.add_result(self._make_pass_result("t1"))
        assert m.avg_keyword_coverage == 1.0

    def test_latency_percentiles(self):
        m = EvaluationMetrics()
        for i in range(10):
            r = self._make_pass_result(f"t{i}")
            r.latency.total_ms = (i + 1) * 100  # 100, 200, ..., 1000
            m.add_result(r)
        assert 500.0 <= m.p50_latency_ms <= 600.0
        assert m.p95_latency_ms >= 900.0

    def test_summary_format(self):
        m = EvaluationMetrics()
        m.add_result(self._make_pass_result("t1"))
        m.complete()
        summary = m.summary()
        assert "overview" in summary
        assert "routing" in summary
        assert "tool_usage" in summary
        assert "response_quality" in summary
        assert "latency" in summary

    def test_to_json(self):
        m = EvaluationMetrics()
        m.add_result(self._make_pass_result("t1"))
        m.complete()
        json_str = m.to_json()
        import json
        data = json.loads(json_str)
        assert "summary" in data
        assert "all_results" in data

    def test_failed_cases_only_returns_failures(self):
        m = EvaluationMetrics()
        m.add_result(self._make_pass_result("t1"))
        m.add_result(self._make_fail_result("t2"))
        failures = m.failed_cases()
        assert len(failures) == 1
        assert failures[0]["case_id"] == "t2"


# --- Evaluator Tests ---


class TestAgentEvaluator:
    def _mock_orchestrator(self):
        orchestrator = MagicMock()

        @dataclass
        class MockRoutingDecision:
            primary_agent: MagicMock = None
            secondary_agents: list = None
            confidence: float = 0.85
            reasoning: str = "test"
            requires_multi_agent: bool = False

            def __post_init__(self):
                if self.primary_agent is None:
                    self.primary_agent = MagicMock(value="finance")
                if self.secondary_agents is None:
                    self.secondary_agents = []

        @dataclass
        class MockThought:
            action: str = "calculate_risk_metrics"

        @dataclass
        class MockResponse:
            answer: str = "The portfolio VaR is 5.2% with a Sharpe ratio of 1.3"
            thoughts: list = None
            sources: list = None
            metadata: dict = None
            execution_time_ms: float = 1000.0

            def __post_init__(self):
                if self.thoughts is None:
                    self.thoughts = [MockThought()]
                if self.sources is None:
                    self.sources = []
                if self.metadata is None:
                    self.metadata = {}

        orchestrator.analyze_and_route = AsyncMock(
            return_value=MockRoutingDecision()
        )
        orchestrator.execute_single_agent = AsyncMock(
            return_value=MockResponse()
        )
        return orchestrator

    @pytest.mark.asyncio
    async def test_evaluate_routing_only(self):
        orchestrator = self._mock_orchestrator()
        evaluator = AgentEvaluator(orchestrator)

        dataset = EvalDataset(
            name="test",
            cases=[
                EvalCase(
                    id="t1",
                    query="Calculate portfolio risk",
                    expected_domain=ExpectedDomain.FINANCE,
                ),
            ],
        )
        metrics = await evaluator.evaluate_routing_only(dataset)
        assert metrics.total_cases == 1
        assert metrics.routing_accuracy == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_full_case(self):
        orchestrator = self._mock_orchestrator()
        evaluator = AgentEvaluator(orchestrator)

        dataset = EvalDataset(
            name="test",
            cases=[
                EvalCase(
                    id="t1",
                    query="Calculate portfolio risk metrics",
                    expected_domain=ExpectedDomain.FINANCE,
                    expected_tools=["calculate_risk_metrics"],
                    expected_keywords=["VaR", "Sharpe"],
                ),
            ],
        )
        metrics = await evaluator.evaluate(dataset)
        assert metrics.total_cases == 1
        assert metrics.pass_rate == 1.0
        assert metrics.routing_accuracy == 1.0
        assert metrics.tool_selection_accuracy == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_handles_error(self):
        orchestrator = self._mock_orchestrator()
        orchestrator.analyze_and_route = AsyncMock(
            side_effect=RuntimeError("API error")
        )
        evaluator = AgentEvaluator(orchestrator)

        dataset = EvalDataset(
            name="test",
            cases=[
                EvalCase(
                    id="t1",
                    query="test",
                    expected_domain=ExpectedDomain.FINANCE,
                ),
            ],
        )
        metrics = await evaluator.evaluate(dataset)
        assert metrics.total_cases == 1
        assert metrics.error_count == 1
        assert metrics.pass_rate == 0.0

    @pytest.mark.asyncio
    async def test_wrong_routing_fails(self):
        orchestrator = self._mock_orchestrator()
        evaluator = AgentEvaluator(orchestrator)

        # Query expects legal but mock returns finance
        dataset = EvalDataset(
            name="test",
            cases=[
                EvalCase(
                    id="t1",
                    query="Review this NDA",
                    expected_domain=ExpectedDomain.LEGAL,
                ),
            ],
        )
        metrics = await evaluator.evaluate_routing_only(dataset)
        assert metrics.routing_accuracy == 0.0
