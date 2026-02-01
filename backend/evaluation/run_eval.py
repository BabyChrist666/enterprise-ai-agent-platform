"""
CLI script to run evaluations against the agent platform.

Usage:
    python -m backend.evaluation.run_eval --dataset routing
    python -m backend.evaluation.run_eval --dataset tools
    python -m backend.evaluation.run_eval --dataset quality
    python -m backend.evaluation.run_eval --dataset full
    python -m backend.evaluation.run_eval --dataset routing --routing-only
"""

import asyncio
import argparse
import json
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.evaluation.evaluator import AgentEvaluator
from backend.evaluation.datasets import (
    ROUTING_EVAL,
    TOOL_USAGE_EVAL,
    RESPONSE_QUALITY_EVAL,
    FULL_EVAL_SUITE,
)


DATASETS = {
    "routing": ROUTING_EVAL,
    "tools": TOOL_USAGE_EVAL,
    "quality": RESPONSE_QUALITY_EVAL,
    "full": FULL_EVAL_SUITE,
}


def print_report(metrics):
    """Print a formatted evaluation report."""
    summary = metrics.summary()

    print("\n" + "=" * 70)
    print("  EVALUATION REPORT")
    print("=" * 70)

    # Overview
    ov = summary["overview"]
    print(f"\n  Total Cases:  {ov['total_cases']}")
    print(f"  Passed:       {ov['passed']}")
    print(f"  Failed:       {ov['failed']}")
    print(f"  Errors:       {ov['errors']}")
    print(f"  Pass Rate:    {ov['pass_rate']}")
    print(f"  Duration:     {ov['duration_s']}s")

    # Routing
    rt = summary["routing"]
    print(f"\n  Routing")
    print(f"    Accuracy:           {rt['accuracy']}")
    print(f"    Multi-Agent Acc:    {rt['multi_agent_accuracy']}")
    print(f"    Avg Confidence:     {rt['avg_confidence']}")

    # Tool Usage
    tu = summary["tool_usage"]
    print(f"\n  Tool Usage")
    print(f"    Selection Acc:      {tu['selection_accuracy']}")
    print(f"    Avg Precision:      {tu['avg_precision']}")
    print(f"    Avg Recall:         {tu['avg_recall']}")

    # Response Quality
    rq = summary["response_quality"]
    print(f"\n  Response Quality")
    print(f"    Keyword Coverage:   {rq['avg_keyword_coverage']}")

    # Latency
    lt = summary["latency"]
    print(f"\n  Latency")
    print(f"    Average:            {lt['avg_ms']}ms")
    print(f"    P50:                {lt['p50_ms']}ms")
    print(f"    P95:                {lt['p95_ms']}ms")
    print(f"    P99:                {lt['p99_ms']}ms")

    # Failures
    failures = metrics.failed_cases()
    if failures:
        print(f"\n  Failed Cases ({len(failures)}):")
        for f in failures:
            print(f"    [{f['case_id']}] {f['query'][:60]}...")
            if f.get("error"):
                print(f"      Error: {f['error']}")
            if f.get("routing"):
                print(f"      Expected: {f['routing']['expected']}, Got: {f['routing']['actual']}")

    print("\n" + "=" * 70)


async def main():
    parser = argparse.ArgumentParser(description="Run agent evaluations")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="routing",
        help="Which evaluation dataset to run",
    )
    parser.add_argument(
        "--routing-only",
        action="store_true",
        help="Only test routing decisions (fast, no API calls)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write JSON results to file",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=1,
        help="Max concurrent evaluations",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-case timeout in seconds",
    )
    args = parser.parse_args()

    dataset = DATASETS[args.dataset]
    print(f"Running evaluation: {dataset.name} ({len(dataset.cases)} cases)")

    # Initialize the orchestrator
    from backend.core.cohere_client import CohereRAGEngine
    from backend.agents.orchestrator import AgentOrchestrator

    rag_engine = CohereRAGEngine()
    orchestrator = AgentOrchestrator(rag_engine)
    evaluator = AgentEvaluator(orchestrator)

    if args.routing_only:
        metrics = await evaluator.evaluate_routing_only(dataset)
    else:
        metrics = await evaluator.evaluate(
            dataset,
            max_concurrent=args.concurrent,
            timeout_s=args.timeout,
        )

    print_report(metrics)

    if args.output:
        with open(args.output, "w") as f:
            f.write(metrics.to_json())
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
