"""
Evaluation metrics for agent system performance.

Tracks routing accuracy, tool usage correctness, response quality,
latency percentiles, and generates summary reports.
"""

import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    SKIP = "skip"


@dataclass
class RoutingResult:
    """Result of evaluating routing for a single query."""
    case_id: str
    expected_domain: str
    actual_domain: str
    expected_multi_agent: bool
    actual_multi_agent: bool
    routing_confidence: float
    correct: bool
    multi_agent_correct: bool


@dataclass
class ToolUsageResult:
    """Result of evaluating tool selection for a single query."""
    case_id: str
    expected_tools: List[str]
    actual_tools: List[str]
    tools_correct: bool
    precision: float  # correct tools / actual tools
    recall: float     # correct tools / expected tools


@dataclass
class ResponseQualityResult:
    """Result of evaluating response content for a single query."""
    case_id: str
    expected_keywords: List[str]
    found_keywords: List[str]
    missing_keywords: List[str]
    keyword_coverage: float  # found / expected
    response_length: int
    has_reasoning: bool


@dataclass
class LatencyResult:
    """Latency measurement for a single query."""
    case_id: str
    total_ms: float
    routing_ms: float = 0.0
    agent_ms: float = 0.0
    tool_calls: int = 0


@dataclass
class CaseResult:
    """Complete evaluation result for a single test case."""
    case_id: str
    query: str
    verdict: Verdict
    routing: Optional[RoutingResult] = None
    tool_usage: Optional[ToolUsageResult] = None
    response_quality: Optional[ResponseQualityResult] = None
    latency: Optional[LatencyResult] = None
    error: Optional[str] = None


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics across all test cases."""
    results: List[CaseResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def add_result(self, result: CaseResult):
        self.results.append(result)

    def complete(self):
        self.completed_at = time.time()

    # --- Routing Metrics ---

    @property
    def routing_results(self) -> List[RoutingResult]:
        return [r.routing for r in self.results if r.routing is not None]

    @property
    def routing_accuracy(self) -> float:
        results = self.routing_results
        if not results:
            return 0.0
        return sum(1 for r in results if r.correct) / len(results)

    @property
    def multi_agent_accuracy(self) -> float:
        results = [r for r in self.routing_results if r.expected_multi_agent]
        if not results:
            return 0.0
        return sum(1 for r in results if r.multi_agent_correct) / len(results)

    @property
    def avg_routing_confidence(self) -> float:
        results = self.routing_results
        if not results:
            return 0.0
        return sum(r.routing_confidence for r in results) / len(results)

    # --- Tool Usage Metrics ---

    @property
    def tool_results(self) -> List[ToolUsageResult]:
        return [r.tool_usage for r in self.results if r.tool_usage is not None]

    @property
    def tool_selection_accuracy(self) -> float:
        results = self.tool_results
        if not results:
            return 0.0
        return sum(1 for r in results if r.tools_correct) / len(results)

    @property
    def avg_tool_precision(self) -> float:
        results = self.tool_results
        if not results:
            return 0.0
        return sum(r.precision for r in results) / len(results)

    @property
    def avg_tool_recall(self) -> float:
        results = self.tool_results
        if not results:
            return 0.0
        return sum(r.recall for r in results) / len(results)

    # --- Response Quality Metrics ---

    @property
    def quality_results(self) -> List[ResponseQualityResult]:
        return [r.response_quality for r in self.results if r.response_quality is not None]

    @property
    def avg_keyword_coverage(self) -> float:
        results = self.quality_results
        if not results:
            return 0.0
        return sum(r.keyword_coverage for r in results) / len(results)

    # --- Latency Metrics ---

    @property
    def latency_results(self) -> List[LatencyResult]:
        return [r.latency for r in self.results if r.latency is not None]

    @property
    def avg_latency_ms(self) -> float:
        results = self.latency_results
        if not results:
            return 0.0
        return sum(r.total_ms for r in results) / len(results)

    @property
    def p50_latency_ms(self) -> float:
        return self._percentile_latency(0.50)

    @property
    def p95_latency_ms(self) -> float:
        return self._percentile_latency(0.95)

    @property
    def p99_latency_ms(self) -> float:
        return self._percentile_latency(0.99)

    def _percentile_latency(self, percentile: float) -> float:
        results = self.latency_results
        if not results:
            return 0.0
        sorted_latencies = sorted(r.total_ms for r in results)
        idx = int(len(sorted_latencies) * percentile)
        idx = min(idx, len(sorted_latencies) - 1)
        return sorted_latencies[idx]

    # --- Aggregate Metrics ---

    @property
    def total_cases(self) -> int:
        return len(self.results)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.verdict == Verdict.PASS)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if r.verdict == Verdict.FAIL)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.error is not None)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.pass_count / len(self.results)

    @property
    def total_duration_s(self) -> float:
        if self.completed_at is None:
            return time.time() - self.started_at
        return self.completed_at - self.started_at

    # --- Reporting ---

    def summary(self) -> Dict[str, Any]:
        return {
            "overview": {
                "total_cases": self.total_cases,
                "passed": self.pass_count,
                "failed": self.fail_count,
                "errors": self.error_count,
                "pass_rate": f"{self.pass_rate:.1%}",
                "duration_s": round(self.total_duration_s, 2),
            },
            "routing": {
                "accuracy": f"{self.routing_accuracy:.1%}",
                "multi_agent_accuracy": f"{self.multi_agent_accuracy:.1%}",
                "avg_confidence": round(self.avg_routing_confidence, 3),
            },
            "tool_usage": {
                "selection_accuracy": f"{self.tool_selection_accuracy:.1%}",
                "avg_precision": round(self.avg_tool_precision, 3),
                "avg_recall": round(self.avg_tool_recall, 3),
            },
            "response_quality": {
                "avg_keyword_coverage": f"{self.avg_keyword_coverage:.1%}",
            },
            "latency": {
                "avg_ms": round(self.avg_latency_ms, 1),
                "p50_ms": round(self.p50_latency_ms, 1),
                "p95_ms": round(self.p95_latency_ms, 1),
                "p99_ms": round(self.p99_latency_ms, 1),
            },
        }

    def failed_cases(self) -> List[Dict[str, Any]]:
        return [
            {
                "case_id": r.case_id,
                "query": r.query,
                "verdict": r.verdict.value,
                "error": r.error,
                "routing": {
                    "expected": r.routing.expected_domain,
                    "actual": r.routing.actual_domain,
                } if r.routing else None,
                "tools": {
                    "expected": r.tool_usage.expected_tools,
                    "actual": r.tool_usage.actual_tools,
                } if r.tool_usage else None,
            }
            for r in self.results
            if r.verdict in (Verdict.FAIL, Verdict.PARTIAL)
        ]

    def to_json(self) -> str:
        return json.dumps(
            {
                "summary": self.summary(),
                "failures": self.failed_cases(),
                "all_results": [
                    {
                        "case_id": r.case_id,
                        "verdict": r.verdict.value,
                        "error": r.error,
                    }
                    for r in self.results
                ],
            },
            indent=2,
        )
