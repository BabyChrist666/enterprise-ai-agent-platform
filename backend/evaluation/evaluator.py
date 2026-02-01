"""
Core evaluation engine that runs test cases against the agent system
and produces structured metrics.
"""

import asyncio
import time
import logging
from typing import Optional

from backend.evaluation.datasets import EvalDataset, EvalCase, ExpectedDomain
from backend.evaluation.metrics import (
    EvaluationMetrics,
    CaseResult,
    RoutingResult,
    ToolUsageResult,
    ResponseQualityResult,
    LatencyResult,
    Verdict,
)

logger = logging.getLogger(__name__)


class AgentEvaluator:
    """
    Runs evaluation datasets against the agent orchestrator and
    produces detailed metrics on routing, tool usage, response quality,
    and latency.
    """

    def __init__(self, orchestrator):
        """
        Args:
            orchestrator: An initialized AgentOrchestrator instance.
        """
        self.orchestrator = orchestrator

    async def evaluate(
        self,
        dataset: EvalDataset,
        max_concurrent: int = 1,
        timeout_s: float = 60.0,
    ) -> EvaluationMetrics:
        """
        Run all cases in a dataset and return aggregated metrics.

        Args:
            dataset: The evaluation dataset to run.
            max_concurrent: Max parallel evaluations (1 = sequential).
            timeout_s: Per-case timeout in seconds.

        Returns:
            EvaluationMetrics with results for every case.
        """
        metrics = EvaluationMetrics()
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_case(case: EvalCase) -> CaseResult:
            async with semaphore:
                return await self._evaluate_case(case, timeout_s)

        tasks = [run_case(case) for case in dataset.cases]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                case = dataset.cases[i]
                metrics.add_result(CaseResult(
                    case_id=case.id,
                    query=case.query,
                    verdict=Verdict.FAIL,
                    error=str(result),
                ))
            else:
                metrics.add_result(result)

        metrics.complete()
        return metrics

    async def evaluate_routing_only(
        self,
        dataset: EvalDataset,
    ) -> EvaluationMetrics:
        """
        Fast evaluation that only tests routing decisions
        without executing agents. Useful for rapid iteration.
        """
        metrics = EvaluationMetrics()

        for case in dataset.cases:
            try:
                start = time.perf_counter()
                decision = await self.orchestrator.analyze_and_route(case.query)
                elapsed_ms = (time.perf_counter() - start) * 1000

                routing = self._evaluate_routing(case, decision)
                verdict = Verdict.PASS if routing.correct else Verdict.FAIL

                if case.requires_multi_agent:
                    verdict = (
                        Verdict.PASS
                        if routing.correct and routing.multi_agent_correct
                        else Verdict.FAIL
                    )

                metrics.add_result(CaseResult(
                    case_id=case.id,
                    query=case.query,
                    verdict=verdict,
                    routing=routing,
                    latency=LatencyResult(
                        case_id=case.id,
                        total_ms=elapsed_ms,
                        routing_ms=elapsed_ms,
                    ),
                ))
            except Exception as e:
                metrics.add_result(CaseResult(
                    case_id=case.id,
                    query=case.query,
                    verdict=Verdict.FAIL,
                    error=str(e),
                ))

        metrics.complete()
        return metrics

    async def _evaluate_case(
        self,
        case: EvalCase,
        timeout_s: float,
    ) -> CaseResult:
        """Evaluate a single test case end-to-end."""
        start = time.perf_counter()

        try:
            # Step 1: Evaluate routing
            route_start = time.perf_counter()
            decision = await asyncio.wait_for(
                self.orchestrator.analyze_and_route(case.query),
                timeout=timeout_s,
            )
            route_ms = (time.perf_counter() - route_start) * 1000
            routing = self._evaluate_routing(case, decision)

            # Step 2: Execute agent and collect response
            agent_start = time.perf_counter()

            if decision.requires_multi_agent:
                agent_types = [decision.primary_agent] + [
                    a for a in decision.secondary_agents
                    if a != decision.primary_agent
                ]
                responses = await asyncio.wait_for(
                    self.orchestrator.execute_multi_agent(
                        case.query, agent_types
                    ),
                    timeout=timeout_s,
                )
                # Combine all responses for evaluation
                combined_answer = " ".join(
                    r.answer for r in responses.values() if hasattr(r, "answer")
                )
                all_thoughts = []
                for r in responses.values():
                    if hasattr(r, "thoughts"):
                        all_thoughts.extend(r.thoughts)
            else:
                response = await asyncio.wait_for(
                    self.orchestrator.execute_single_agent(
                        decision.primary_agent, case.query
                    ),
                    timeout=timeout_s,
                )
                combined_answer = response.answer
                all_thoughts = response.thoughts

            agent_ms = (time.perf_counter() - agent_start) * 1000
            total_ms = (time.perf_counter() - start) * 1000

            # Step 3: Evaluate tool usage
            actual_tools = self._extract_tools_from_thoughts(all_thoughts)
            tool_usage = self._evaluate_tool_usage(case, actual_tools)

            # Step 4: Evaluate response quality
            response_quality = self._evaluate_response_quality(
                case, combined_answer, all_thoughts
            )

            # Step 5: Determine verdict
            verdict = self._determine_verdict(
                routing, tool_usage, response_quality
            )

            tool_call_count = len(actual_tools)

            return CaseResult(
                case_id=case.id,
                query=case.query,
                verdict=verdict,
                routing=routing,
                tool_usage=tool_usage,
                response_quality=response_quality,
                latency=LatencyResult(
                    case_id=case.id,
                    total_ms=total_ms,
                    routing_ms=route_ms,
                    agent_ms=agent_ms,
                    tool_calls=tool_call_count,
                ),
            )

        except asyncio.TimeoutError:
            total_ms = (time.perf_counter() - start) * 1000
            return CaseResult(
                case_id=case.id,
                query=case.query,
                verdict=Verdict.FAIL,
                error=f"Timeout after {timeout_s}s",
                latency=LatencyResult(case_id=case.id, total_ms=total_ms),
            )
        except Exception as e:
            total_ms = (time.perf_counter() - start) * 1000
            logger.error(f"Error evaluating case {case.id}: {e}")
            return CaseResult(
                case_id=case.id,
                query=case.query,
                verdict=Verdict.FAIL,
                error=str(e),
                latency=LatencyResult(case_id=case.id, total_ms=total_ms),
            )

    def _evaluate_routing(self, case: EvalCase, decision) -> RoutingResult:
        """Check if the routing decision matches expectations."""
        actual_domain = decision.primary_agent.value

        if case.expected_domain == ExpectedDomain.MULTI:
            # For multi-agent cases, primary can be any of the expected domains
            expected_domains = [d.value for d in case.secondary_domains]
            correct = actual_domain in expected_domains
        else:
            correct = actual_domain == case.expected_domain.value

        multi_agent_correct = (
            decision.requires_multi_agent == case.requires_multi_agent
        )

        return RoutingResult(
            case_id=case.id,
            expected_domain=case.expected_domain.value,
            actual_domain=actual_domain,
            expected_multi_agent=case.requires_multi_agent,
            actual_multi_agent=decision.requires_multi_agent,
            routing_confidence=decision.confidence,
            correct=correct,
            multi_agent_correct=multi_agent_correct,
        )

    def _evaluate_tool_usage(
        self, case: EvalCase, actual_tools: list
    ) -> ToolUsageResult:
        """Check if the correct tools were selected."""
        expected_set = set(case.expected_tools)
        actual_set = set(actual_tools)

        if not expected_set:
            # No expected tools specified - skip tool evaluation
            return ToolUsageResult(
                case_id=case.id,
                expected_tools=case.expected_tools,
                actual_tools=actual_tools,
                tools_correct=True,
                precision=1.0,
                recall=1.0,
            )

        correct_tools = expected_set & actual_set

        precision = (
            len(correct_tools) / len(actual_set) if actual_set else 0.0
        )
        recall = (
            len(correct_tools) / len(expected_set) if expected_set else 0.0
        )

        return ToolUsageResult(
            case_id=case.id,
            expected_tools=case.expected_tools,
            actual_tools=actual_tools,
            tools_correct=expected_set <= actual_set,
            precision=precision,
            recall=recall,
        )

    def _evaluate_response_quality(
        self,
        case: EvalCase,
        answer: str,
        thoughts: list,
    ) -> ResponseQualityResult:
        """Check if the response contains expected information."""
        answer_lower = answer.lower()

        found = []
        missing = []
        for kw in case.expected_keywords:
            if kw.lower() in answer_lower:
                found.append(kw)
            else:
                missing.append(kw)

        coverage = (
            len(found) / len(case.expected_keywords)
            if case.expected_keywords
            else 1.0
        )

        return ResponseQualityResult(
            case_id=case.id,
            expected_keywords=case.expected_keywords,
            found_keywords=found,
            missing_keywords=missing,
            keyword_coverage=coverage,
            response_length=len(answer),
            has_reasoning=len(thoughts) > 0,
        )

    def _extract_tools_from_thoughts(self, thoughts: list) -> list:
        """Extract tool names from agent reasoning trace."""
        tools = []
        for thought in thoughts:
            if hasattr(thought, "action") and thought.action:
                action = thought.action
                if action != "FINAL_ANSWER" and action not in tools:
                    tools.append(action)
        return tools

    def _determine_verdict(
        self,
        routing: RoutingResult,
        tool_usage: ToolUsageResult,
        response_quality: ResponseQualityResult,
    ) -> Verdict:
        """Determine overall pass/fail/partial verdict."""
        if not routing.correct:
            return Verdict.FAIL

        all_pass = (
            routing.correct
            and tool_usage.tools_correct
            and response_quality.keyword_coverage >= 0.5
        )

        if all_pass:
            return Verdict.PASS

        # Partial: routing correct but something else is off
        if routing.correct and (
            not tool_usage.tools_correct
            or response_quality.keyword_coverage < 0.5
        ):
            return Verdict.PARTIAL

        return Verdict.FAIL
