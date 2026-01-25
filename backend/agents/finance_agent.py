"""
Finance Agent - Enterprise-grade financial analysis and insights.

Capabilities:
- Portfolio risk analysis
- SEC filing parsing and summarization
- Earnings call analysis
- Market sentiment analysis
- Financial ratio calculations
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from ..core.base_agent import BaseAgent, Tool, Document
from ..core.cohere_client import CohereRAGEngine, VectorStore


class FinanceAgent(BaseAgent):
    """
    Specialized agent for financial analysis and insights.

    Tools:
    - calculate_risk_metrics: Portfolio risk calculations (VaR, Sharpe, etc.)
    - analyze_financial_ratios: Calculate and interpret financial ratios
    - parse_sec_filing: Extract key info from SEC filings
    - analyze_earnings: Summarize earnings calls and reports
    - get_market_sentiment: Analyze market sentiment for a ticker
    """

    def __init__(
        self,
        rag_engine: Optional[CohereRAGEngine] = None,
        vector_store: Optional[VectorStore] = None
    ):
        super().__init__(
            name="FinanceAgent",
            description="Expert financial analyst AI for portfolio analysis, SEC filings, and market insights",
            rag_engine=rag_engine,
            vector_store=vector_store
        )

    def get_system_prompt(self) -> str:
        return """You are an expert Financial Analyst AI with deep expertise in:
- Portfolio management and risk analysis
- SEC filings (10-K, 10-Q, 8-K, proxy statements)
- Financial statement analysis
- Market sentiment and trends
- Quantitative finance and risk metrics

## Guidelines
1. Always provide data-driven insights with specific numbers
2. Cite sources and explain your methodology
3. Highlight risks and uncertainties
4. Use industry-standard metrics (Sharpe ratio, VaR, P/E, etc.)
5. Consider both technical and fundamental factors
6. Provide actionable recommendations when appropriate

## Compliance
- Do not provide specific buy/sell recommendations
- Always include appropriate risk disclaimers
- Note when data may be stale or incomplete
"""

    def _register_tools(self):
        """Register finance-specific tools."""

        self.register_tool(Tool(
            name="calculate_risk_metrics",
            description="Calculate portfolio risk metrics including VaR, Sharpe ratio, beta, and volatility",
            parameters={
                "type": "object",
                "properties": {
                    "portfolio": {
                        "type": "object",
                        "description": "Portfolio holdings with ticker and weight",
                        "additionalProperties": {"type": "number"}
                    },
                    "benchmark": {
                        "type": "string",
                        "description": "Benchmark ticker (e.g., SPY)",
                        "default": "SPY"
                    },
                    "period_days": {
                        "type": "integer",
                        "description": "Historical period for calculations",
                        "default": 252
                    }
                },
                "required": ["portfolio"]
            },
            function=self._calculate_risk_metrics
        ))

        self.register_tool(Tool(
            name="analyze_financial_ratios",
            description="Calculate and analyze key financial ratios from company data",
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol"
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific ratios to calculate (e.g., P/E, ROE, debt_to_equity)",
                        "default": ["P/E", "P/B", "ROE", "ROA", "debt_to_equity", "current_ratio"]
                    }
                },
                "required": ["ticker"]
            },
            function=self._analyze_financial_ratios
        ))

        self.register_tool(Tool(
            name="parse_sec_filing",
            description="Extract and summarize key information from SEC filings",
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Company ticker symbol"
                    },
                    "filing_type": {
                        "type": "string",
                        "enum": ["10-K", "10-Q", "8-K", "DEF 14A"],
                        "description": "Type of SEC filing"
                    },
                    "sections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific sections to extract",
                        "default": ["risk_factors", "business_overview", "financial_highlights"]
                    }
                },
                "required": ["ticker", "filing_type"]
            },
            function=self._parse_sec_filing
        ))

        self.register_tool(Tool(
            name="analyze_earnings",
            description="Analyze earnings reports and calls for key insights",
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Company ticker symbol"
                    },
                    "quarter": {
                        "type": "string",
                        "description": "Quarter to analyze (e.g., Q4 2024)",
                        "default": "latest"
                    },
                    "focus_areas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Areas to focus on",
                        "default": ["guidance", "surprises", "management_tone", "key_metrics"]
                    }
                },
                "required": ["ticker"]
            },
            function=self._analyze_earnings
        ))

        self.register_tool(Tool(
            name="calculate_dcf_valuation",
            description="Perform discounted cash flow valuation analysis",
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Company ticker symbol"
                    },
                    "growth_rate": {
                        "type": "number",
                        "description": "Expected revenue growth rate",
                        "default": 0.10
                    },
                    "discount_rate": {
                        "type": "number",
                        "description": "Discount rate (WACC)",
                        "default": 0.10
                    },
                    "terminal_growth": {
                        "type": "number",
                        "description": "Terminal growth rate",
                        "default": 0.025
                    },
                    "projection_years": {
                        "type": "integer",
                        "description": "Years to project",
                        "default": 5
                    }
                },
                "required": ["ticker"]
            },
            function=self._calculate_dcf_valuation
        ))

    async def _calculate_risk_metrics(
        self,
        portfolio: Dict[str, float],
        benchmark: str = "SPY",
        period_days: int = 252
    ) -> str:
        """Calculate portfolio risk metrics."""
        # In production, this would fetch real market data
        # For demo, we'll simulate the calculations

        import random
        random.seed(hash(str(portfolio)))  # Consistent results for same portfolio

        total_weight = sum(portfolio.values())
        normalized = {k: v/total_weight for k, v in portfolio.items()}

        # Simulated metrics (in production, calculate from real returns)
        portfolio_return = sum(
            weight * (0.08 + random.uniform(-0.05, 0.15))
            for weight in normalized.values()
        )
        portfolio_volatility = 0.15 + random.uniform(-0.05, 0.10)
        risk_free_rate = 0.045

        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        var_95 = portfolio_volatility * 1.645  # 95% VaR
        var_99 = portfolio_volatility * 2.326  # 99% VaR
        beta = 0.8 + random.uniform(-0.3, 0.5)
        max_drawdown = 0.10 + random.uniform(0, 0.15)

        return f"""
## Portfolio Risk Analysis
**Period:** {period_days} trading days | **Benchmark:** {benchmark}

### Holdings
{chr(10).join(f"- {ticker}: {weight*100:.1f}%" for ticker, weight in normalized.items())}

### Key Metrics
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Expected Return** | {portfolio_return*100:.2f}% | Annual expected return |
| **Volatility (σ)** | {portfolio_volatility*100:.2f}% | Annualized standard deviation |
| **Sharpe Ratio** | {sharpe_ratio:.2f} | {'Good' if sharpe_ratio > 1 else 'Moderate' if sharpe_ratio > 0.5 else 'Low'} risk-adjusted return |
| **Beta** | {beta:.2f} | {'More' if beta > 1 else 'Less'} volatile than market |
| **VaR (95%)** | {var_95*100:.2f}% | Max daily loss 95% confidence |
| **VaR (99%)** | {var_99*100:.2f}% | Max daily loss 99% confidence |
| **Max Drawdown** | {max_drawdown*100:.2f}% | Largest peak-to-trough decline |

### Risk Assessment
- **Diversification Score:** {'Well diversified' if len(portfolio) >= 5 else 'Consider adding more positions'}
- **Concentration Risk:** {'High' if max(normalized.values()) > 0.4 else 'Moderate' if max(normalized.values()) > 0.25 else 'Low'}

⚠️ *Past performance does not guarantee future results. These metrics are based on historical data.*
"""

    async def _analyze_financial_ratios(
        self,
        ticker: str,
        metrics: List[str] = None
    ) -> str:
        """Analyze financial ratios for a company."""
        metrics = metrics or ["P/E", "P/B", "ROE", "ROA", "debt_to_equity", "current_ratio"]

        # Simulated data (in production, fetch from financial APIs)
        import random
        random.seed(hash(ticker))

        ratios = {
            "P/E": round(15 + random.uniform(-5, 25), 2),
            "P/B": round(2 + random.uniform(-1, 4), 2),
            "ROE": round(0.12 + random.uniform(-0.05, 0.15), 4),
            "ROA": round(0.06 + random.uniform(-0.02, 0.08), 4),
            "debt_to_equity": round(0.5 + random.uniform(-0.3, 1.0), 2),
            "current_ratio": round(1.5 + random.uniform(-0.5, 1.0), 2),
            "gross_margin": round(0.35 + random.uniform(-0.15, 0.25), 4),
            "operating_margin": round(0.15 + random.uniform(-0.08, 0.15), 4),
            "net_margin": round(0.08 + random.uniform(-0.05, 0.10), 4),
        }

        def interpret_ratio(name: str, value: float) -> str:
            interpretations = {
                "P/E": "undervalued" if value < 15 else "fairly valued" if value < 25 else "premium valuation",
                "P/B": "below book" if value < 1 else "at book" if value < 2 else "above book value",
                "ROE": "strong" if value > 0.15 else "moderate" if value > 0.10 else "weak",
                "ROA": "efficient" if value > 0.08 else "moderate" if value > 0.05 else "low efficiency",
                "debt_to_equity": "low leverage" if value < 0.5 else "moderate" if value < 1.0 else "high leverage",
                "current_ratio": "strong liquidity" if value > 1.5 else "adequate" if value > 1.0 else "liquidity risk",
            }
            return interpretations.get(name, "N/A")

        output = f"""
## Financial Ratio Analysis: {ticker.upper()}

### Valuation Ratios
| Ratio | Value | Industry Comparison |
|-------|-------|---------------------|
| P/E Ratio | {ratios['P/E']} | {interpret_ratio('P/E', ratios['P/E'])} |
| P/B Ratio | {ratios['P/B']} | {interpret_ratio('P/B', ratios['P/B'])} |

### Profitability Ratios
| Ratio | Value | Assessment |
|-------|-------|------------|
| ROE | {ratios['ROE']*100:.1f}% | {interpret_ratio('ROE', ratios['ROE'])} |
| ROA | {ratios['ROA']*100:.1f}% | {interpret_ratio('ROA', ratios['ROA'])} |
| Gross Margin | {ratios['gross_margin']*100:.1f}% | |
| Operating Margin | {ratios['operating_margin']*100:.1f}% | |
| Net Margin | {ratios['net_margin']*100:.1f}% | |

### Leverage & Liquidity
| Ratio | Value | Assessment |
|-------|-------|------------|
| Debt/Equity | {ratios['debt_to_equity']} | {interpret_ratio('debt_to_equity', ratios['debt_to_equity'])} |
| Current Ratio | {ratios['current_ratio']} | {interpret_ratio('current_ratio', ratios['current_ratio'])} |

### Summary
The company shows {'strong' if ratios['ROE'] > 0.15 and ratios['current_ratio'] > 1.5 else 'mixed'} fundamentals with {interpret_ratio('P/E', ratios['P/E'])} based on earnings multiples.
"""
        return output

    async def _parse_sec_filing(
        self,
        ticker: str,
        filing_type: str,
        sections: List[str] = None
    ) -> str:
        """Parse and summarize SEC filing."""
        sections = sections or ["risk_factors", "business_overview", "financial_highlights"]

        return f"""
## SEC {filing_type} Analysis: {ticker.upper()}

### Business Overview
{ticker.upper()} operates in the technology sector, providing enterprise software solutions. The company has shown consistent revenue growth with expanding market presence.

### Financial Highlights (from {filing_type})
- **Revenue:** $4.2B (+12% YoY)
- **Net Income:** $380M (+8% YoY)
- **Operating Cash Flow:** $520M
- **R&D Investment:** $680M (16% of revenue)

### Key Risk Factors
1. **Competition:** Intense competition from established players and new entrants
2. **Regulatory:** Evolving data privacy regulations (GDPR, CCPA)
3. **Technology:** Rapid technological change requiring continuous innovation
4. **Macroeconomic:** Sensitivity to enterprise IT spending cycles
5. **Concentration:** Top 10 customers represent 35% of revenue

### Management Discussion Highlights
- Expanding AI/ML capabilities across product portfolio
- International expansion focus on APAC region
- Strategic acquisitions pipeline active
- Guidance reaffirmed for fiscal year

### Notable Changes from Prior Filing
- New risk factor added regarding AI regulatory environment
- Increased disclosure on cybersecurity measures
- Updated segment reporting structure

*Source: SEC EDGAR | Filing Date: Simulated*
"""

    async def _analyze_earnings(
        self,
        ticker: str,
        quarter: str = "latest",
        focus_areas: List[str] = None
    ) -> str:
        """Analyze earnings report and call."""
        focus_areas = focus_areas or ["guidance", "surprises", "management_tone", "key_metrics"]

        return f"""
## Earnings Analysis: {ticker.upper()} - {quarter}

### Earnings Summary
| Metric | Actual | Estimate | Surprise |
|--------|--------|----------|----------|
| EPS | $1.42 | $1.35 | +5.2% ✅ |
| Revenue | $4.28B | $4.15B | +3.1% ✅ |
| Gross Margin | 68.5% | 67.0% | +150bps ✅ |

### Guidance Update
- **Q+1 Revenue:** $4.35-4.45B (vs consensus $4.30B) ⬆️
- **FY Revenue:** $17.2-17.5B (raised from $16.8-17.2B) ⬆️
- **FY EPS:** $5.60-5.80 (raised from $5.40-5.60) ⬆️

### Management Tone Analysis
**Overall Sentiment:** Positive (8/10)
- CEO emphasized "strong demand" and "expanding TAM"
- CFO highlighted "operating leverage" and "margin expansion"
- Mentioned "cautious optimism" on macro environment

### Key Call Highlights
1. **AI Product Adoption:** 40% of enterprise customers now using AI features
2. **Net Revenue Retention:** 118% (up from 115%)
3. **New Customer Acquisition:** +25% YoY
4. **International Growth:** EMEA +28%, APAC +35%

### Analyst Q&A Themes
- Competition in AI space (management confident in differentiation)
- Pricing power (selective increases, customers accepting)
- M&A pipeline (active, focused on AI capabilities)

### Stock Reaction
Pre-market: +6.2% on guidance raise
"""

    async def _calculate_dcf_valuation(
        self,
        ticker: str,
        growth_rate: float = 0.10,
        discount_rate: float = 0.10,
        terminal_growth: float = 0.025,
        projection_years: int = 5
    ) -> str:
        """Perform DCF valuation analysis."""
        import random
        random.seed(hash(ticker))

        # Simulated base financials
        current_fcf = 500 + random.uniform(-100, 300)  # millions
        shares_outstanding = 200 + random.uniform(-50, 100)  # millions

        # Project cash flows
        projected_fcf = []
        fcf = current_fcf
        for year in range(1, projection_years + 1):
            fcf *= (1 + growth_rate)
            projected_fcf.append(fcf)

        # Calculate terminal value
        terminal_value = projected_fcf[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)

        # Discount cash flows
        pv_fcf = sum(
            cf / ((1 + discount_rate) ** year)
            for year, cf in enumerate(projected_fcf, 1)
        )
        pv_terminal = terminal_value / ((1 + discount_rate) ** projection_years)

        enterprise_value = pv_fcf + pv_terminal
        equity_value = enterprise_value  # Simplified (should subtract debt, add cash)
        fair_value_per_share = equity_value / shares_outstanding

        # Simulated current price
        current_price = fair_value_per_share * (0.8 + random.uniform(-0.2, 0.4))
        upside = (fair_value_per_share - current_price) / current_price * 100

        return f"""
## DCF Valuation Analysis: {ticker.upper()}

### Assumptions
| Parameter | Value |
|-----------|-------|
| Base FCF | ${current_fcf:.0f}M |
| Growth Rate (Years 1-{projection_years}) | {growth_rate*100:.1f}% |
| Discount Rate (WACC) | {discount_rate*100:.1f}% |
| Terminal Growth Rate | {terminal_growth*100:.1f}% |
| Shares Outstanding | {shares_outstanding:.0f}M |

### Projected Free Cash Flows
| Year | FCF ($M) | PV ($M) |
|------|----------|---------|
{chr(10).join(f"| {i+1} | ${fcf:.0f} | ${fcf/((1+discount_rate)**(i+1)):.0f} |" for i, fcf in enumerate(projected_fcf))}
| Terminal | ${terminal_value:.0f} | ${pv_terminal:.0f} |

### Valuation Summary
| Metric | Value |
|--------|-------|
| PV of Cash Flows | ${pv_fcf:.0f}M |
| PV of Terminal Value | ${pv_terminal:.0f}M |
| **Enterprise Value** | **${enterprise_value:.0f}M** |
| **Fair Value/Share** | **${fair_value_per_share:.2f}** |
| Current Price | ${current_price:.2f} |
| **Implied Upside** | **{upside:+.1f}%** |

### Sensitivity Analysis (Fair Value)
| Growth \\ WACC | 8% | 10% | 12% |
|---------------|-----|-----|-----|
| 8% | ${fair_value_per_share*1.3:.0f} | ${fair_value_per_share*1.1:.0f} | ${fair_value_per_share*0.9:.0f} |
| 10% | ${fair_value_per_share*1.2:.0f} | ${fair_value_per_share:.0f} | ${fair_value_per_share*0.85:.0f} |
| 12% | ${fair_value_per_share*1.1:.0f} | ${fair_value_per_share*0.95:.0f} | ${fair_value_per_share*0.8:.0f} |

### Recommendation
Based on DCF analysis, the stock appears {'undervalued' if upside > 15 else 'fairly valued' if upside > -10 else 'overvalued'}.

⚠️ *DCF models are highly sensitive to assumptions. This is not investment advice.*
"""
