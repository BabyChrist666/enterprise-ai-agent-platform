"""
Evaluation datasets with ground truth for agent testing.

Each EvalCase defines a query, the expected routing, expected tools,
and criteria for judging response quality.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class ExpectedDomain(str, Enum):
    FINANCE = "finance"
    LEGAL = "legal"
    HEALTHCARE = "healthcare"
    MULTI = "multi"


@dataclass
class EvalCase:
    """A single evaluation test case."""
    id: str
    query: str
    expected_domain: ExpectedDomain
    expected_tools: List[str] = field(default_factory=list)
    expected_keywords: List[str] = field(default_factory=list)
    requires_multi_agent: bool = False
    secondary_domains: List[ExpectedDomain] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard
    description: str = ""


@dataclass
class EvalDataset:
    """Collection of evaluation cases."""
    name: str
    cases: List[EvalCase]
    description: str = ""

    def filter_by_domain(self, domain: ExpectedDomain) -> "EvalDataset":
        filtered = [c for c in self.cases if c.expected_domain == domain]
        return EvalDataset(
            name=f"{self.name}_{domain.value}",
            cases=filtered,
            description=f"{self.description} (filtered: {domain.value})",
        )

    def filter_by_difficulty(self, difficulty: str) -> "EvalDataset":
        filtered = [c for c in self.cases if c.difficulty == difficulty]
        return EvalDataset(
            name=f"{self.name}_{difficulty}",
            cases=filtered,
            description=f"{self.description} (filtered: {difficulty})",
        )


# --- Built-in Evaluation Datasets ---

ROUTING_EVAL = EvalDataset(
    name="routing_accuracy",
    description="Tests whether the orchestrator routes queries to the correct agent",
    cases=[
        # Finance - clear signals
        EvalCase(
            id="route_fin_01",
            query="What is the current P/E ratio for AAPL and how does it compare to the sector average?",
            expected_domain=ExpectedDomain.FINANCE,
            expected_tools=["analyze_financial_ratios"],
            expected_keywords=["P/E", "ratio", "AAPL"],
            difficulty="easy",
            description="Clear finance query with ticker symbol",
        ),
        EvalCase(
            id="route_fin_02",
            query="Run a DCF valuation on Tesla with 10% discount rate and 3% terminal growth",
            expected_domain=ExpectedDomain.FINANCE,
            expected_tools=["calculate_dcf_valuation"],
            expected_keywords=["DCF", "valuation", "discount rate"],
            difficulty="easy",
        ),
        EvalCase(
            id="route_fin_03",
            query="Calculate VaR and Sharpe ratio for a portfolio with 40% AAPL, 30% MSFT, 30% GOOGL",
            expected_domain=ExpectedDomain.FINANCE,
            expected_tools=["calculate_risk_metrics"],
            expected_keywords=["VaR", "Sharpe", "portfolio"],
            difficulty="easy",
        ),
        EvalCase(
            id="route_fin_04",
            query="What were the key takeaways from the latest NVIDIA earnings call?",
            expected_domain=ExpectedDomain.FINANCE,
            expected_tools=["analyze_earnings"],
            expected_keywords=["earnings"],
            difficulty="medium",
        ),

        # Legal - clear signals
        EvalCase(
            id="route_leg_01",
            query="Review this NDA and identify any non-standard clauses or risks",
            expected_domain=ExpectedDomain.LEGAL,
            expected_tools=["analyze_nda"],
            expected_keywords=["NDA", "clauses", "risks"],
            difficulty="easy",
        ),
        EvalCase(
            id="route_leg_02",
            query="Check if our data processing practices comply with GDPR and CCPA requirements",
            expected_domain=ExpectedDomain.LEGAL,
            expected_tools=["check_compliance"],
            expected_keywords=["GDPR", "CCPA", "compliance"],
            difficulty="easy",
        ),
        EvalCase(
            id="route_leg_03",
            query="Extract all termination clauses and liability caps from this vendor contract",
            expected_domain=ExpectedDomain.LEGAL,
            expected_tools=["extract_contract_clauses"],
            expected_keywords=["clauses", "contract", "liability"],
            difficulty="easy",
        ),

        # Healthcare - clear signals
        EvalCase(
            id="route_hc_01",
            query="Parse this clinical note and extract diagnoses, medications, and procedures",
            expected_domain=ExpectedDomain.HEALTHCARE,
            expected_tools=["parse_clinical_note"],
            expected_keywords=["clinical", "diagnoses", "medications"],
            difficulty="easy",
        ),
        EvalCase(
            id="route_hc_02",
            query="Check for drug interactions between metformin, lisinopril, and warfarin",
            expected_domain=ExpectedDomain.HEALTHCARE,
            expected_tools=["check_drug_interactions"],
            expected_keywords=["drug interactions", "metformin"],
            difficulty="easy",
        ),
        EvalCase(
            id="route_hc_03",
            query="Suggest ICD-10 codes for a patient with type 2 diabetes and hypertension",
            expected_domain=ExpectedDomain.HEALTHCARE,
            expected_tools=["suggest_icd_codes"],
            expected_keywords=["ICD-10", "diabetes"],
            difficulty="easy",
        ),

        # Ambiguous queries - harder routing
        EvalCase(
            id="route_amb_01",
            query="What are the risks involved in this situation?",
            expected_domain=ExpectedDomain.FINANCE,
            expected_tools=[],
            expected_keywords=["risk"],
            difficulty="hard",
            description="Ambiguous - 'risk' appears in all domains",
        ),
        EvalCase(
            id="route_amb_02",
            query="Summarize the key points from this document",
            expected_domain=ExpectedDomain.LEGAL,
            expected_tools=[],
            expected_keywords=[],
            difficulty="hard",
            description="No clear domain signal",
        ),

        # Multi-agent queries
        EvalCase(
            id="route_multi_01",
            query="What are the legal and financial implications of a healthcare data breach?",
            expected_domain=ExpectedDomain.MULTI,
            requires_multi_agent=True,
            secondary_domains=[
                ExpectedDomain.LEGAL,
                ExpectedDomain.FINANCE,
                ExpectedDomain.HEALTHCARE,
            ],
            expected_keywords=["legal", "financial", "healthcare"],
            difficulty="medium",
        ),
        EvalCase(
            id="route_multi_02",
            query="Review the compliance requirements and financial impact of HIPAA violations",
            expected_domain=ExpectedDomain.MULTI,
            requires_multi_agent=True,
            secondary_domains=[ExpectedDomain.LEGAL, ExpectedDomain.FINANCE],
            expected_keywords=["compliance", "financial", "HIPAA"],
            difficulty="medium",
        ),
        EvalCase(
            id="route_multi_03",
            query="Analyze the contract terms for a pharmaceutical company acquisition including drug portfolio risks",
            expected_domain=ExpectedDomain.MULTI,
            requires_multi_agent=True,
            secondary_domains=[
                ExpectedDomain.LEGAL,
                ExpectedDomain.FINANCE,
                ExpectedDomain.HEALTHCARE,
            ],
            expected_keywords=["contract", "acquisition", "pharmaceutical", "drug"],
            difficulty="hard",
        ),
    ],
)


TOOL_USAGE_EVAL = EvalDataset(
    name="tool_usage",
    description="Tests whether agents select and use the correct tools",
    cases=[
        EvalCase(
            id="tool_fin_01",
            query="Calculate the risk metrics for my tech-heavy portfolio: 50% AAPL, 30% GOOGL, 20% MSFT",
            expected_domain=ExpectedDomain.FINANCE,
            expected_tools=["calculate_risk_metrics"],
            difficulty="easy",
        ),
        EvalCase(
            id="tool_fin_02",
            query="What is Apple's P/E ratio and return on equity?",
            expected_domain=ExpectedDomain.FINANCE,
            expected_tools=["analyze_financial_ratios"],
            difficulty="easy",
        ),
        EvalCase(
            id="tool_fin_03",
            query="I need a full valuation of Microsoft using DCF analysis with 8% WACC",
            expected_domain=ExpectedDomain.FINANCE,
            expected_tools=["calculate_dcf_valuation"],
            difficulty="medium",
        ),
        EvalCase(
            id="tool_leg_01",
            query="Extract all indemnification and limitation of liability clauses from this agreement",
            expected_domain=ExpectedDomain.LEGAL,
            expected_tools=["extract_contract_clauses"],
            difficulty="easy",
        ),
        EvalCase(
            id="tool_leg_02",
            query="Is our European customer data handling GDPR compliant?",
            expected_domain=ExpectedDomain.LEGAL,
            expected_tools=["check_compliance"],
            difficulty="medium",
        ),
        EvalCase(
            id="tool_hc_01",
            query="Patient presents with chest pain, shortness of breath, and elevated troponin. Parse this note.",
            expected_domain=ExpectedDomain.HEALTHCARE,
            expected_tools=["parse_clinical_note"],
            difficulty="easy",
        ),
        EvalCase(
            id="tool_hc_02",
            query="Can a patient safely take aspirin with clopidogrel and omeprazole together?",
            expected_domain=ExpectedDomain.HEALTHCARE,
            expected_tools=["check_drug_interactions"],
            difficulty="medium",
        ),
    ],
)


RESPONSE_QUALITY_EVAL = EvalDataset(
    name="response_quality",
    description="Tests whether agent responses contain expected information",
    cases=[
        EvalCase(
            id="qual_fin_01",
            query="Analyze the risk profile of a 60/40 stock/bond portfolio",
            expected_domain=ExpectedDomain.FINANCE,
            expected_tools=["calculate_risk_metrics"],
            expected_keywords=["VaR", "volatility", "Sharpe", "risk"],
            difficulty="medium",
            description="Response should mention key risk metrics",
        ),
        EvalCase(
            id="qual_leg_01",
            query="What GDPR requirements apply to processing EU customer data?",
            expected_domain=ExpectedDomain.LEGAL,
            expected_tools=["check_compliance"],
            expected_keywords=["data protection", "consent", "right to erasure", "DPO"],
            difficulty="medium",
            description="Response should cover core GDPR principles",
        ),
        EvalCase(
            id="qual_hc_01",
            query="What ICD-10 codes apply to a patient with congestive heart failure and type 2 diabetes?",
            expected_domain=ExpectedDomain.HEALTHCARE,
            expected_tools=["suggest_icd_codes"],
            expected_keywords=["I50", "E11", "ICD"],
            difficulty="medium",
            description="Response should include relevant ICD codes",
        ),
    ],
)


# Combined full evaluation suite
FULL_EVAL_SUITE = EvalDataset(
    name="full_suite",
    description="Complete evaluation across routing, tool usage, and response quality",
    cases=ROUTING_EVAL.cases + TOOL_USAGE_EVAL.cases + RESPONSE_QUALITY_EVAL.cases,
)
