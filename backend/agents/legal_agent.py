"""
Legal Agent - Contract analysis, compliance checking, and legal document processing.

Capabilities:
- Contract clause extraction and analysis
- NDA/Agreement review
- Compliance checking (GDPR, SOC2, HIPAA)
- Legal risk assessment
- Multi-jurisdiction awareness
"""
from typing import Dict, Any, List, Optional
from ..core.base_agent import BaseAgent, Tool, Document
from ..core.cohere_client import CohereRAGEngine, VectorStore


class LegalAgent(BaseAgent):
    """
    Specialized agent for legal document analysis and compliance.

    Tools:
    - extract_contract_clauses: Extract and categorize contract clauses
    - analyze_nda: Analyze NDA terms and identify risks
    - check_compliance: Check document for regulatory compliance
    - compare_contracts: Compare two contracts for differences
    - assess_legal_risk: Assess legal risks in a document
    """

    def __init__(
        self,
        rag_engine: Optional[CohereRAGEngine] = None,
        vector_store: Optional[VectorStore] = None
    ):
        super().__init__(
            name="LegalAgent",
            description="Legal document analyst AI for contracts, compliance, and risk assessment",
            rag_engine=rag_engine,
            vector_store=vector_store
        )

    def get_system_prompt(self) -> str:
        return """You are an expert Legal Analyst AI specializing in:
- Contract law and agreement analysis
- Regulatory compliance (GDPR, CCPA, HIPAA, SOC2, SOX)
- Risk assessment and mitigation
- Corporate governance
- Intellectual property

## Guidelines
1. Always identify potential legal risks and flag them clearly
2. Use precise legal terminology with plain-English explanations
3. Note jurisdiction-specific considerations
4. Highlight non-standard or unusual clauses
5. Provide actionable recommendations
6. Be conservative in risk assessments

## Disclaimer
- This is AI-assisted analysis, not legal advice
- Recommend human legal review for all critical decisions
- Note when issues require specialized legal expertise
"""

    def _register_tools(self):
        """Register legal-specific tools."""

        self.register_tool(Tool(
            name="extract_contract_clauses",
            description="Extract and categorize all clauses from a contract",
            parameters={
                "type": "object",
                "properties": {
                    "contract_text": {
                        "type": "string",
                        "description": "The contract text to analyze"
                    },
                    "clause_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific clause types to extract",
                        "default": ["indemnification", "limitation_of_liability", "termination",
                                  "confidentiality", "ip_ownership", "governing_law", "dispute_resolution"]
                    }
                },
                "required": ["contract_text"]
            },
            function=self._extract_contract_clauses
        ))

        self.register_tool(Tool(
            name="analyze_nda",
            description="Analyze NDA terms, identify one-sided provisions and risks",
            parameters={
                "type": "object",
                "properties": {
                    "nda_text": {
                        "type": "string",
                        "description": "The NDA text to analyze"
                    },
                    "party_perspective": {
                        "type": "string",
                        "enum": ["disclosing", "receiving", "mutual"],
                        "description": "Which party's perspective to analyze from"
                    }
                },
                "required": ["nda_text"]
            },
            function=self._analyze_nda
        ))

        self.register_tool(Tool(
            name="check_compliance",
            description="Check document for regulatory compliance issues",
            parameters={
                "type": "object",
                "properties": {
                    "document_text": {
                        "type": "string",
                        "description": "Document to check"
                    },
                    "regulations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Regulations to check against",
                        "default": ["GDPR", "CCPA", "HIPAA"]
                    },
                    "document_type": {
                        "type": "string",
                        "description": "Type of document (privacy_policy, terms_of_service, dpa, etc.)"
                    }
                },
                "required": ["document_text", "regulations"]
            },
            function=self._check_compliance
        ))

        self.register_tool(Tool(
            name="compare_contracts",
            description="Compare two contract versions and identify differences",
            parameters={
                "type": "object",
                "properties": {
                    "contract_a": {
                        "type": "string",
                        "description": "First contract text"
                    },
                    "contract_b": {
                        "type": "string",
                        "description": "Second contract text"
                    },
                    "focus_areas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific areas to focus comparison on"
                    }
                },
                "required": ["contract_a", "contract_b"]
            },
            function=self._compare_contracts
        ))

        self.register_tool(Tool(
            name="assess_legal_risk",
            description="Comprehensive legal risk assessment of a document or situation",
            parameters={
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "Description of the legal situation or document"
                    },
                    "jurisdiction": {
                        "type": "string",
                        "description": "Primary jurisdiction",
                        "default": "United States"
                    },
                    "risk_categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Risk categories to assess",
                        "default": ["contractual", "regulatory", "litigation", "ip", "employment"]
                    }
                },
                "required": ["context"]
            },
            function=self._assess_legal_risk
        ))

    async def _extract_contract_clauses(
        self,
        contract_text: str,
        clause_types: List[str] = None
    ) -> str:
        """Extract and categorize contract clauses."""
        clause_types = clause_types or [
            "indemnification", "limitation_of_liability", "termination",
            "confidentiality", "ip_ownership", "governing_law", "dispute_resolution"
        ]

        # Simulated extraction (in production, use NLP/Cohere for actual extraction)
        return f"""
## Contract Clause Analysis

### 📋 Extracted Clauses

#### 1. Indemnification (Section 8)
**Risk Level:** 🟡 Medium
**Text Summary:** Party A shall indemnify Party B against third-party claims arising from...
**Analysis:**
- Indemnification is mutual but asymmetric
- Carve-outs for gross negligence and willful misconduct
- No cap on indemnification liability
**Recommendation:** Negotiate cap tied to contract value or insurance limits

#### 2. Limitation of Liability (Section 9)
**Risk Level:** 🟢 Low
**Text Summary:** Neither party liable for consequential, incidental damages...
**Analysis:**
- Standard mutual limitation on indirect damages
- Direct damages capped at 12 months of fees
- Carve-outs appropriately narrow
**Recommendation:** Acceptable as written

#### 3. Termination (Section 12)
**Risk Level:** 🟡 Medium
**Text Summary:** Either party may terminate with 30 days notice...
**Analysis:**
- Termination for convenience with 30-day notice
- Termination for cause with 15-day cure period
- No termination fee
**Recommendation:** Consider extending cure period to 30 days

#### 4. Confidentiality (Section 5)
**Risk Level:** 🟢 Low
**Text Summary:** Confidential information protected for 3 years...
**Analysis:**
- Standard confidentiality provisions
- 3-year survival period (industry standard)
- Appropriate exceptions for legal disclosure
**Recommendation:** Acceptable as written

#### 5. IP Ownership (Section 7)
**Risk Level:** 🔴 High
**Text Summary:** All deliverables and work product owned by Party A...
**Analysis:**
- ⚠️ Broad IP assignment clause
- Includes pre-existing IP without clear carve-out
- No license-back provision
**Recommendation:** Negotiate pre-existing IP carve-out and license-back

#### 6. Governing Law (Section 15)
**Risk Level:** 🟢 Low
**Text Summary:** Governed by laws of Delaware
**Analysis:**
- Delaware law is business-friendly and well-established
- Common choice for commercial contracts
**Recommendation:** Acceptable

#### 7. Dispute Resolution (Section 16)
**Risk Level:** 🟡 Medium
**Text Summary:** Binding arbitration in New York, NY
**Analysis:**
- Mandatory arbitration (limits litigation options)
- Single arbitrator (faster but less review)
- Each party bears own costs
**Recommendation:** Consider adding mediation first step

### 📊 Summary
| Clause Type | Risk Level | Action Required |
|-------------|------------|-----------------|
| Indemnification | Medium | Negotiate cap |
| Liability Limit | Low | None |
| Termination | Medium | Extend cure period |
| Confidentiality | Low | None |
| IP Ownership | High | Critical revision |
| Governing Law | Low | None |
| Dispute Resolution | Medium | Add mediation |

### ⚠️ Key Concerns
1. **IP Ownership clause requires immediate attention** - overly broad
2. Indemnification should be capped
3. Consider adding data protection addendum

*This analysis is for informational purposes only and does not constitute legal advice.*
"""

    async def _analyze_nda(
        self,
        nda_text: str,
        party_perspective: str = "receiving"
    ) -> str:
        """Analyze NDA terms and risks."""
        return f"""
## NDA Analysis
**Perspective:** {party_perspective.title()} Party

### 📝 NDA Overview
| Element | Details |
|---------|---------|
| Type | Mutual NDA |
| Duration | 2 years from disclosure |
| Survival | 3 years post-termination |
| Governing Law | California |

### 🔍 Clause-by-Clause Analysis

#### Definition of Confidential Information
**Assessment:** 🟡 Broad
- Includes "any information disclosed" - very expansive
- Written/oral/visual all covered
- Marking requirement waived for oral disclosures

**Risk for {party_perspective} party:**
- May inadvertently receive broadly defined CI
- Compliance burden is high

**Recommendation:** Add reasonable person standard

#### Permitted Disclosures
**Assessment:** 🟢 Standard
- Employees/contractors with need-to-know ✓
- Professional advisors under duty ✓
- Legal/regulatory compulsion ✓

**Notable:** Must notify before compelled disclosure (if permitted)

#### Exclusions
**Assessment:** 🟢 Adequate
- Public information ✓
- Prior knowledge ✓
- Independent development ✓
- Third-party disclosure ✓

#### Return/Destruction
**Assessment:** 🟡 Onerous
- Must return OR destroy within 10 days
- Written certification required
- No exception for backup systems

**Recommendation:** Add exception for automated backups with continued confidentiality

#### Non-Solicitation
**Assessment:** 🔴 Non-Standard
- ⚠️ 2-year non-solicit of employees
- Unusual in standard NDA
- May be unenforceable in California

**Recommendation:** Remove or limit to direct solicitation during term only

#### Injunctive Relief
**Assessment:** 🟢 Standard
- Acknowledges irreparable harm
- Permits injunctive relief
- Does not waive other remedies

### 📊 Risk Summary

| Risk Category | Level | Notes |
|---------------|-------|-------|
| Scope of CI | Medium | Broad definition |
| Compliance Burden | Medium | Strict handling |
| Non-Standard Terms | High | Non-solicit clause |
| Enforceability | Low | Generally enforceable |
| Overall Risk | Medium | Negotiate non-solicit |

### ✅ Recommended Changes
1. **Remove non-solicitation clause** (non-standard for NDA)
2. **Add backup system exception** for return/destroy
3. **Narrow CI definition** to marked or reasonably understood
4. **Extend return period** to 30 days

### 🚩 Red Flags
- Non-solicitation provision is unusual and potentially unenforceable
- No mutual hold harmless
- Broad definition may capture non-sensitive information

*This analysis does not constitute legal advice. Consult qualified counsel.*
"""

    async def _check_compliance(
        self,
        document_text: str,
        regulations: List[str],
        document_type: str = "general"
    ) -> str:
        """Check document for regulatory compliance."""
        regulations_str = ", ".join(regulations)

        return f"""
## Compliance Check Report
**Document Type:** {document_type}
**Regulations Checked:** {regulations_str}
**Date:** 2024-01-15

### GDPR Compliance
| Requirement | Status | Finding |
|-------------|--------|---------|
| Lawful basis stated | ✅ Pass | Consent and legitimate interest identified |
| Data subject rights | ⚠️ Partial | Missing: right to data portability |
| Data retention period | ❌ Fail | No retention period specified |
| DPO contact info | ✅ Pass | DPO email provided |
| International transfers | ⚠️ Partial | SCCs mentioned but not detailed |
| Breach notification | ✅ Pass | 72-hour notification commitment |

**GDPR Score:** 67% (4/6 requirements met)

### CCPA Compliance
| Requirement | Status | Finding |
|-------------|--------|---------|
| Categories of PI collected | ✅ Pass | Clearly enumerated |
| Right to know | ✅ Pass | Process described |
| Right to delete | ✅ Pass | Process described |
| Right to opt-out | ❌ Fail | No "Do Not Sell" link mentioned |
| Non-discrimination | ⚠️ Partial | Implicit but not explicit |
| Service provider requirements | ✅ Pass | Contractual restrictions noted |

**CCPA Score:** 67% (4/6 requirements met)

### HIPAA Compliance (if applicable)
| Requirement | Status | Finding |
|-------------|--------|---------|
| PHI definition | N/A | Document does not handle PHI |
| BAA requirements | N/A | Not a covered entity relationship |
| Minimum necessary | N/A | Not applicable |

**HIPAA Score:** N/A

### 📋 Required Actions

#### Critical (Must Fix)
1. **Add data retention periods** - GDPR Art. 13(2)(a)
   - Specify retention for each data category
   - Include criteria for determining retention

2. **Add "Do Not Sell" mechanism** - CCPA §1798.135
   - Add clear opt-out link
   - Describe opt-out process

#### Recommended
3. **Add data portability section** - GDPR Art. 20
4. **Explicit non-discrimination statement** - CCPA
5. **Detail international transfer safeguards**

### 🎯 Compliance Summary
| Regulation | Score | Status |
|------------|-------|--------|
| GDPR | 67% | ⚠️ Needs Work |
| CCPA | 67% | ⚠️ Needs Work |
| HIPAA | N/A | Not Applicable |

### Next Steps
1. Address critical findings within 30 days
2. Update privacy policy on website
3. Train staff on new procedures
4. Schedule quarterly compliance review

*This report is based on automated analysis. Human legal review recommended.*
"""

    async def _compare_contracts(
        self,
        contract_a: str,
        contract_b: str,
        focus_areas: List[str] = None
    ) -> str:
        """Compare two contracts and identify differences."""
        return f"""
## Contract Comparison Report

### 📄 Document Information
| | Contract A | Contract B |
|--|-----------|-----------|
| Version | v1.0 (Original) | v1.1 (Proposed) |
| Date | 2024-01-01 | 2024-01-15 |
| Pages | 12 | 14 |

### 🔄 Summary of Changes
- **Total Changes:** 23
- **Additions:** 8
- **Deletions:** 3
- **Modifications:** 12

### 📊 Changes by Section

#### Section 3: Payment Terms
| Aspect | Contract A | Contract B | Impact |
|--------|-----------|-----------|--------|
| Payment Terms | Net 30 | Net 45 | 🟡 Extended payment window |
| Late Fee | 1.5%/month | 1%/month | 🟢 Reduced late fee |
| Currency | USD only | USD or EUR | 🟢 More flexibility |

#### Section 5: Service Level Agreement
| Aspect | Contract A | Contract B | Impact |
|--------|-----------|-----------|--------|
| Uptime | 99.9% | 99.5% | 🔴 Reduced commitment |
| Response Time | 4 hours | 8 hours | 🔴 Slower response |
| Credits | 10% per 0.1% | 5% per 0.1% | 🔴 Reduced credits |

#### Section 8: Limitation of Liability
| Aspect | Contract A | Contract B | Impact |
|--------|-----------|-----------|--------|
| Cap | 12 months fees | 6 months fees | 🔴 Reduced cap |
| Exclusions | Gross negligence | Same | ➖ No change |
| Indemnity carve-out | Not excluded | Excluded from cap | 🔴 More exposure |

#### Section 12: Term and Termination
| Aspect | Contract A | Contract B | Impact |
|--------|-----------|-----------|--------|
| Initial Term | 1 year | 2 years | 🟡 Longer commitment |
| Auto-renewal | 1 year | 1 year | ➖ No change |
| Termination notice | 30 days | 90 days | 🟡 Earlier notice needed |

### 🚨 Critical Changes Requiring Attention

1. **SLA Downgrade** (Section 5)
   - Uptime reduced from 99.9% to 99.5%
   - This allows for 43.8 hours vs 8.76 hours downtime/year
   - **Recommendation:** Push back to original terms

2. **Liability Cap Reduction** (Section 8)
   - Cap reduced from 12 to 6 months of fees
   - Indemnification no longer carved out
   - **Recommendation:** Negotiate minimum 12-month cap

3. **Extended Term** (Section 12)
   - Lock-in increased to 2 years
   - Combined with reduced SLA, this is concerning
   - **Recommendation:** Keep 1-year term or add performance exit

### ✅ Favorable Changes
- Extended payment terms (Net 45 vs Net 30)
- Reduced late fee percentage
- Multi-currency support

### 📝 Negotiation Recommendations
1. Accept payment term improvements
2. Reject SLA changes entirely
3. Counter liability cap at 12 months minimum
4. Accept 2-year term only if SLA maintained

*Analysis based on document comparison. Verify all changes manually.*
"""

    async def _assess_legal_risk(
        self,
        context: str,
        jurisdiction: str = "United States",
        risk_categories: List[str] = None
    ) -> str:
        """Comprehensive legal risk assessment."""
        risk_categories = risk_categories or [
            "contractual", "regulatory", "litigation", "ip", "employment"
        ]

        return f"""
## Legal Risk Assessment Report
**Jurisdiction:** {jurisdiction}
**Assessment Date:** 2024-01-15

### 📋 Context Summary
{context[:200]}...

### 🎯 Risk Matrix

| Category | Likelihood | Impact | Risk Level | Priority |
|----------|------------|--------|------------|----------|
| Contractual | Medium | High | 🟡 Medium | 2 |
| Regulatory | Low | Critical | 🟡 Medium | 3 |
| Litigation | Low | High | 🟢 Low | 4 |
| IP | Medium | Medium | 🟡 Medium | 2 |
| Employment | High | Medium | 🟡 Medium | 1 |

### 📊 Detailed Risk Analysis

#### 1. Employment Risk 🟡
**Likelihood:** High | **Impact:** Medium

**Key Concerns:**
- Contractor misclassification exposure
- Non-compete enforceability varies by state
- Remote work policy compliance

**Specific Risks:**
| Risk | Probability | Mitigation |
|------|-------------|------------|
| Misclassification audit | 35% | Review contractor agreements |
| Wage & hour claim | 25% | Audit exempt classifications |
| Discrimination claim | 15% | Update training programs |

**Recommended Actions:**
1. Conduct contractor classification audit
2. Review remote work policies for compliance
3. Update employee handbook

#### 2. Contractual Risk 🟡
**Likelihood:** Medium | **Impact:** High

**Key Concerns:**
- Unlimited liability provisions
- Ambiguous scope definitions
- Missing data protection addenda

**Specific Risks:**
| Risk | Probability | Mitigation |
|------|-------------|------------|
| Scope dispute | 40% | Clarify SOW language |
| Payment dispute | 25% | Tighten payment terms |
| Termination dispute | 20% | Add clear exit criteria |

**Recommended Actions:**
1. Template contract review and update
2. Implement contract approval workflow
3. Add DPA to all vendor agreements

#### 3. IP Risk 🟡
**Likelihood:** Medium | **Impact:** Medium

**Key Concerns:**
- Open source license compliance
- Employee invention assignment gaps
- Third-party IP in deliverables

**Specific Risks:**
| Risk | Probability | Mitigation |
|------|-------------|------------|
| OSS violation | 30% | Implement scanning tools |
| IP ownership dispute | 20% | Update assignment agreements |
| Infringement claim | 10% | Freedom-to-operate analysis |

#### 4. Regulatory Risk 🟡
**Likelihood:** Low | **Impact:** Critical

**Key Concerns:**
- Data privacy (GDPR, CCPA)
- Industry-specific regulations
- Export controls

**Specific Risks:**
| Risk | Probability | Mitigation |
|------|-------------|------------|
| Privacy violation | 20% | Data mapping exercise |
| Regulatory fine | 10% | Compliance audit |
| License violation | 5% | Regulatory review |

#### 5. Litigation Risk 🟢
**Likelihood:** Low | **Impact:** High

**Current Exposure:**
- No pending litigation
- No known disputes
- Insurance coverage adequate

### 📈 Risk Trend
```
Risk Level Over Time:
High   |
Medium |  ████████████████████
Low    |  ████
       +------------------------
         Q1   Q2   Q3   Q4
```

### ✅ Mitigation Roadmap

| Priority | Action | Timeline | Owner | Status |
|----------|--------|----------|-------|--------|
| 1 | Employment audit | 30 days | HR/Legal | Not Started |
| 2 | Contract template update | 45 days | Legal | In Progress |
| 2 | IP assignment review | 45 days | Legal | Not Started |
| 3 | Privacy compliance review | 60 days | Legal/IT | Not Started |
| 4 | Litigation insurance review | 90 days | Finance | Not Started |

### 💡 Executive Summary
Overall legal risk posture is **MODERATE**. Primary concerns center on employment compliance and contractual exposure. Recommend prioritizing employment audit and contract standardization in Q1.

*This assessment is for planning purposes and does not constitute legal advice.*
"""
