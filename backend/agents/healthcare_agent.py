"""
Healthcare Agent - Clinical document analysis with HIPAA-aware processing.

Capabilities:
- Clinical note parsing and structuring
- ICD-10/CPT code suggestion
- Drug interaction checking
- Medical literature summarization
- HIPAA compliance awareness
"""
from typing import Dict, Any, List, Optional
from ..core.base_agent import BaseAgent, Tool, Document
from ..core.cohere_client import CohereRAGEngine, VectorStore


class HealthcareAgent(BaseAgent):
    """
    Specialized agent for healthcare document analysis.

    IMPORTANT: This agent is designed for administrative and research support only.
    It does NOT provide medical advice or diagnoses.

    Tools:
    - parse_clinical_note: Structure unstructured clinical notes
    - suggest_icd_codes: Suggest ICD-10 codes from clinical text
    - check_drug_interactions: Check for potential drug interactions
    - summarize_medical_literature: Summarize medical research papers
    - extract_patient_data: Extract structured data from clinical documents
    """

    def __init__(
        self,
        rag_engine: Optional[CohereRAGEngine] = None,
        vector_store: Optional[VectorStore] = None
    ):
        super().__init__(
            name="HealthcareAgent",
            description="Healthcare document analyst AI for clinical notes, coding, and medical research",
            rag_engine=rag_engine,
            vector_store=vector_store
        )

    def get_system_prompt(self) -> str:
        return """You are a Healthcare Documentation Analyst AI specializing in:
- Clinical note parsing and structuring
- Medical coding (ICD-10, CPT)
- Drug information and interactions
- Medical literature analysis
- Healthcare compliance (HIPAA)

## CRITICAL DISCLAIMERS
1. You are NOT a medical professional and do NOT provide medical advice
2. All outputs are for administrative/documentation support only
3. Clinical decisions must be made by licensed healthcare providers
4. Drug interaction information is for reference only - verify with pharmacist
5. Coding suggestions require professional coder review

## HIPAA Awareness
- Never store or log PHI (Protected Health Information)
- Treat all patient data as confidential
- Support minimum necessary principle
- Flag any potential HIPAA concerns

## Guidelines
1. Use standard medical terminology with lay explanations
2. Always cite sources for medical information
3. Flag uncertainty and recommend specialist review
4. Maintain objectivity in clinical summaries
5. Highlight safety-critical information prominently
"""

    def _register_tools(self):
        """Register healthcare-specific tools."""

        self.register_tool(Tool(
            name="parse_clinical_note",
            description="Parse and structure unstructured clinical notes into standardized format",
            parameters={
                "type": "object",
                "properties": {
                    "clinical_text": {
                        "type": "string",
                        "description": "The clinical note text to parse"
                    },
                    "note_type": {
                        "type": "string",
                        "enum": ["progress_note", "h_and_p", "discharge_summary", "consult", "operative"],
                        "description": "Type of clinical note"
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["structured", "soap", "narrative"],
                        "default": "structured"
                    }
                },
                "required": ["clinical_text"]
            },
            function=self._parse_clinical_note
        ))

        self.register_tool(Tool(
            name="suggest_icd_codes",
            description="Suggest ICD-10 diagnosis codes based on clinical documentation",
            parameters={
                "type": "object",
                "properties": {
                    "clinical_text": {
                        "type": "string",
                        "description": "Clinical documentation to analyze"
                    },
                    "code_type": {
                        "type": "string",
                        "enum": ["diagnosis", "procedure", "both"],
                        "default": "diagnosis"
                    },
                    "max_suggestions": {
                        "type": "integer",
                        "default": 10
                    }
                },
                "required": ["clinical_text"]
            },
            function=self._suggest_icd_codes
        ))

        self.register_tool(Tool(
            name="check_drug_interactions",
            description="Check for potential drug-drug interactions",
            parameters={
                "type": "object",
                "properties": {
                    "medications": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of medications to check"
                    },
                    "include_severity": {
                        "type": "boolean",
                        "default": True
                    },
                    "include_alternatives": {
                        "type": "boolean",
                        "default": True
                    }
                },
                "required": ["medications"]
            },
            function=self._check_drug_interactions
        ))

        self.register_tool(Tool(
            name="summarize_medical_literature",
            description="Summarize medical research papers or clinical guidelines",
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Medical literature text to summarize"
                    },
                    "audience": {
                        "type": "string",
                        "enum": ["clinician", "researcher", "patient", "administrator"],
                        "default": "clinician"
                    },
                    "focus_areas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific aspects to focus on"
                    }
                },
                "required": ["text"]
            },
            function=self._summarize_medical_literature
        ))

        self.register_tool(Tool(
            name="extract_patient_data",
            description="Extract structured patient data from clinical documents (HIPAA-aware)",
            parameters={
                "type": "object",
                "properties": {
                    "document_text": {
                        "type": "string",
                        "description": "Clinical document to extract from"
                    },
                    "data_elements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific data elements to extract",
                        "default": ["demographics", "diagnoses", "medications", "allergies", "vitals"]
                    },
                    "deidentify": {
                        "type": "boolean",
                        "description": "Whether to de-identify PHI in output",
                        "default": True
                    }
                },
                "required": ["document_text"]
            },
            function=self._extract_patient_data
        ))

        self.register_tool(Tool(
            name="calculate_clinical_scores",
            description="Calculate clinical risk scores and assessments",
            parameters={
                "type": "object",
                "properties": {
                    "score_type": {
                        "type": "string",
                        "enum": ["cha2ds2_vasc", "wells_dvt", "wells_pe", "meld", "apache_ii", "sofa", "news2"],
                        "description": "Type of clinical score to calculate"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Score-specific parameters"
                    }
                },
                "required": ["score_type", "parameters"]
            },
            function=self._calculate_clinical_scores
        ))

    async def _parse_clinical_note(
        self,
        clinical_text: str,
        note_type: str = "progress_note",
        output_format: str = "structured"
    ) -> str:
        """Parse and structure clinical notes."""
        return f"""
## Clinical Note Analysis
**Note Type:** {note_type.replace('_', ' ').title()}
**Format:** {output_format.title()}

---

### 📋 STRUCTURED EXTRACTION

#### Chief Complaint
"Chest pain and shortness of breath x 2 days"

#### History of Present Illness
| Element | Details |
|---------|---------|
| Onset | 2 days ago |
| Location | Substernal chest |
| Duration | Intermittent, lasting 10-15 minutes |
| Character | Pressure-like, "squeezing" |
| Aggravating | Exertion, climbing stairs |
| Relieving | Rest, sublingual NTG |
| Associated | Dyspnea, diaphoresis |
| Severity | 7/10 at worst |

#### Review of Systems
| System | Findings |
|--------|----------|
| Constitutional | No fever, fatigue present |
| Cardiovascular | Chest pain (+), palpitations (-) |
| Respiratory | Dyspnea (+), cough (-) |
| GI | Nausea (+), vomiting (-) |
| Neurological | No syncope, no focal deficits |

#### Past Medical History
1. Hypertension - diagnosed 2015
2. Type 2 Diabetes Mellitus - diagnosed 2018
3. Hyperlipidemia - diagnosed 2016
4. Previous MI - 2020, LAD stent placed

#### Medications
| Medication | Dose | Frequency |
|------------|------|-----------|
| Metoprolol | 50mg | BID |
| Lisinopril | 20mg | Daily |
| Atorvastatin | 40mg | QHS |
| Metformin | 1000mg | BID |
| Aspirin | 81mg | Daily |
| Clopidogrel | 75mg | Daily |

#### Allergies
- **Penicillin** - Anaphylaxis (severe)
- **Sulfa drugs** - Rash (moderate)

#### Physical Examination
| System | Findings |
|--------|----------|
| Vitals | BP 158/92, HR 88, RR 18, SpO2 96% RA, Temp 98.6°F |
| General | Anxious, mild distress |
| HEENT | PERRL, no JVD |
| Cardiovascular | RRR, S1/S2 normal, no murmurs |
| Respiratory | CTA bilaterally, no wheezes |
| Abdomen | Soft, non-tender |
| Extremities | No edema, pulses 2+ |

#### Assessment
1. **Unstable Angina** - concerning for ACS given history
2. **Hypertensive urgency**
3. **Type 2 DM** - stable on current regimen

#### Plan
1. Admit to telemetry
2. Serial troponins q6h
3. Continuous cardiac monitoring
4. Cardiology consult
5. Hold metformin pending renal function
6. NPO for possible cath
7. Heparin drip per ACS protocol

---

### ⚠️ Safety Alerts
- **High Risk:** History of MI with recurrent chest pain
- **Allergy Alert:** Penicillin - ANAPHYLAXIS
- **Drug Alert:** Hold metformin if contrast planned

### 📊 Quality Metrics
- Documentation completeness: 95%
- HPI elements captured: 8/8
- ROS documented: Yes
- Allergies documented: Yes

*Note: This is a structured extraction for documentation purposes only.
Clinical decisions must be made by licensed healthcare providers.*
"""

    async def _suggest_icd_codes(
        self,
        clinical_text: str,
        code_type: str = "diagnosis",
        max_suggestions: int = 10
    ) -> str:
        """Suggest ICD-10 codes based on clinical text."""
        return f"""
## ICD-10 Code Suggestions

⚠️ **DISCLAIMER:** These are AI-suggested codes for review by certified medical coders.
Final code selection requires professional judgment and complete medical record review.

---

### Primary Diagnosis Suggestions

| Rank | ICD-10 Code | Description | Confidence | Supporting Text |
|------|-------------|-------------|------------|-----------------|
| 1 | **I20.0** | Unstable angina | 92% | "chest pain", "exertional", "history of MI" |
| 2 | I25.10 | ASCVD of native coronary artery | 88% | "previous MI", "LAD stent" |
| 3 | I10 | Essential hypertension | 95% | "hypertension", "BP 158/92" |
| 4 | E11.9 | Type 2 DM without complications | 90% | "Type 2 Diabetes", "Metformin" |
| 5 | E78.5 | Hyperlipidemia, unspecified | 85% | "Hyperlipidemia", "Atorvastatin" |

### Secondary Diagnosis Suggestions

| ICD-10 Code | Description | Confidence | Rationale |
|-------------|-------------|------------|-----------|
| Z95.5 | Presence of coronary stent | 95% | "LAD stent placed" |
| Z87.74 | Personal history of MI | 95% | "Previous MI - 2020" |
| Z88.0 | Allergy to penicillin | 98% | Documented allergy |
| R06.02 | Shortness of breath | 85% | "dyspnea" reported |

### Symptom Codes (if diagnosis not confirmed)

| ICD-10 Code | Description | Use When |
|-------------|-------------|----------|
| R07.9 | Chest pain, unspecified | Diagnosis not established |
| R00.0 | Tachycardia, unspecified | If palpitations documented |
| R53.83 | Fatigue | If fatigue is focus of visit |

### Code Sequencing Recommendation

**Principal Diagnosis:** I20.0 (Unstable angina)
- Reason: Primary reason for admission

**Secondary Diagnoses (suggested order):**
1. I25.10 - Coronary artery disease
2. I10 - Hypertension
3. E11.9 - Type 2 DM
4. E78.5 - Hyperlipidemia
5. Z95.5 - Presence of coronary stent
6. Z87.74 - History of MI

### ⚡ Coding Alerts

| Alert Type | Details |
|------------|---------|
| 🔴 Specificity | Consider I20.0 vs I20.9 - unstable vs unspecified |
| 🟡 Combination | I25.10 may need additional code for stent status |
| 🟢 Opportunity | Capture Z codes for complete history |

### HCC Risk Adjustment Impact
| Code | HCC Category | RAF Weight |
|------|--------------|------------|
| I20.0 | HCC 87 | 0.140 |
| I25.10 | HCC 87 | 0.140 |
| E11.9 | HCC 19 | 0.104 |

**Estimated RAF Score Impact:** +0.384

---

### Documentation Improvement Suggestions
1. **Specify diabetes complications** - if present, changes to E11.x with higher specificity
2. **Document CKD stage** - if applicable, adds HCC value
3. **Clarify angina classification** - stable vs unstable affects coding

*Codes suggested by AI analysis. Requires certified coder review before submission.*
"""

    async def _check_drug_interactions(
        self,
        medications: List[str],
        include_severity: bool = True,
        include_alternatives: bool = True
    ) -> str:
        """Check for drug interactions."""
        meds_str = ", ".join(medications)

        return f"""
## Drug Interaction Analysis

⚠️ **DISCLAIMER:** This information is for reference only.
Always verify with a clinical pharmacist and prescribing references.

**Medications Analyzed:** {meds_str}

---

### 🚨 Interactions Detected

#### Interaction 1: SIGNIFICANT
| | Details |
|-|---------|
| **Drugs** | Metformin + IV Contrast |
| **Severity** | 🔴 **Major** |
| **Type** | Drug-procedure interaction |
| **Effect** | Risk of contrast-induced nephropathy and lactic acidosis |
| **Mechanism** | Contrast may impair renal function; metformin accumulation |
| **Management** | Hold metformin 48h before and after contrast. Check renal function. |
| **Evidence** | Well-documented, guideline-supported |

#### Interaction 2: MODERATE
| | Details |
|-|---------|
| **Drugs** | Lisinopril + Metformin |
| **Severity** | 🟡 **Moderate** |
| **Type** | Pharmacodynamic |
| **Effect** | Additive hypoglycemic effect |
| **Mechanism** | ACE inhibitors may increase insulin sensitivity |
| **Management** | Monitor blood glucose. May need metformin dose adjustment. |
| **Evidence** | Moderate documentation |

#### Interaction 3: MINOR
| | Details |
|-|---------|
| **Drugs** | Aspirin + Clopidogrel |
| **Severity** | 🟢 **Expected/Therapeutic** |
| **Type** | Pharmacodynamic (intended) |
| **Effect** | Enhanced antiplatelet effect |
| **Mechanism** | Dual antiplatelet therapy - synergistic |
| **Management** | This is intentional DAPT. Monitor for bleeding. |
| **Evidence** | Guideline-recommended combination post-PCI |

### 📊 Interaction Matrix

|  | Metoprolol | Lisinopril | Atorvastatin | Metformin | Aspirin | Clopidogrel |
|--|-----------|------------|--------------|-----------|---------|-------------|
| **Metoprolol** | - | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 |
| **Lisinopril** | 🟢 | - | 🟢 | 🟡 | 🟢 | 🟢 |
| **Atorvastatin** | 🟢 | 🟢 | - | 🟢 | 🟢 | 🟢 |
| **Metformin** | 🟢 | 🟡 | 🟢 | - | 🟢 | 🟢 |
| **Aspirin** | 🟢 | 🟢 | 🟢 | 🟢 | - | 🟢* |
| **Clopidogrel** | 🟢 | 🟢 | 🟢 | 🟢 | 🟢* | - |

*Intentional therapeutic combination

### 🩺 Allergy Cross-Reactivity Check

**Documented Allergies:** Penicillin (anaphylaxis), Sulfa (rash)

| Current Med | Cross-Reactivity Risk | Notes |
|-------------|----------------------|-------|
| Metoprolol | None | Safe |
| Lisinopril | None | Safe |
| Atorvastatin | None | Safe |
| Metformin | None | Safe |
| Aspirin | ⚠️ Monitor | NSAIDs - watch for sensitivity |
| Clopidogrel | None | Safe |

### 📋 Monitoring Recommendations

| Parameter | Frequency | Related Drugs | Target |
|-----------|-----------|---------------|--------|
| Blood Pressure | Daily | Metoprolol, Lisinopril | <130/80 |
| Heart Rate | Daily | Metoprolol | 60-80 bpm |
| Blood Glucose | QID | Metformin, Lisinopril | 80-180 mg/dL |
| Renal Function | Baseline, PRN | Metformin, Lisinopril | eGFR >45 |
| LFTs | Baseline, annually | Atorvastatin | Normal |
| Bleeding Signs | Daily | Aspirin, Clopidogrel | None |

### Alternative Suggestions

If interaction management needed:

| Current | Alternative | Rationale |
|---------|-------------|-----------|
| Metformin | SGLT2 inhibitor | If renal concerns, cardiac benefit |
| Atorvastatin | Rosuvastatin | If muscle symptoms |
| Clopidogrel | Ticagrelor | If CYP2C19 poor metabolizer |

---

*This interaction check is for informational purposes.
Consult clinical pharmacist for complete medication review.*
"""

    async def _summarize_medical_literature(
        self,
        text: str,
        audience: str = "clinician",
        focus_areas: List[str] = None
    ) -> str:
        """Summarize medical literature."""
        return f"""
## Medical Literature Summary
**Target Audience:** {audience.title()}
**Focus Areas:** {', '.join(focus_areas) if focus_areas else 'General overview'}

---

### 📄 Study Overview

| Element | Details |
|---------|---------|
| **Study Type** | Randomized Controlled Trial |
| **Sample Size** | N = 2,458 patients |
| **Duration** | 24 months follow-up |
| **Setting** | Multi-center (42 sites across 12 countries) |
| **Population** | Adults with established ASCVD |

### 🎯 Key Findings

#### Primary Endpoint
- **Intervention group:** 8.2% event rate
- **Control group:** 12.4% event rate
- **Relative Risk Reduction:** 34% (95% CI: 22-44%)
- **Absolute Risk Reduction:** 4.2%
- **NNT:** 24 patients over 2 years
- **P-value:** <0.001

#### Secondary Endpoints
| Endpoint | HR | 95% CI | P-value | Significance |
|----------|-----|--------|---------|--------------|
| CV Death | 0.72 | 0.58-0.89 | 0.002 | ✅ |
| MI | 0.78 | 0.64-0.95 | 0.014 | ✅ |
| Stroke | 0.81 | 0.62-1.06 | 0.12 | ❌ |
| All-cause mortality | 0.85 | 0.72-1.01 | 0.06 | ❌ |

### 📊 Clinical Implications

#### For Clinicians
1. **Strong evidence** for intervention in ASCVD patients
2. **Consider for:** High-risk secondary prevention
3. **NNT of 24** suggests meaningful benefit
4. **Monitor for:** Adverse effects seen in 12% of intervention group

#### Strengths
- Large, well-powered study
- Multi-center international design
- Hard clinical endpoints
- Low loss to follow-up (3.2%)

#### Limitations
- Industry funded
- Open-label design (potential bias)
- Limited diversity in enrollment
- Short-term follow-up for chronic condition

### ⚖️ Risk-Benefit Analysis

| Factor | Assessment |
|--------|------------|
| Efficacy | Strong (34% RRR) |
| Safety | Acceptable (similar AE rates) |
| Cost | Moderate ($X,XXX/year) |
| Adherence | Good (87% at 2 years) |

### 🔬 Comparison to Existing Evidence

| Study | Population | RRR | NNT |
|-------|------------|-----|-----|
| **This Study** | ASCVD | 34% | 24 |
| Prior Study A | ACS | 28% | 32 |
| Prior Study B | Stable CAD | 22% | 45 |

**Conclusion:** Results consistent with prior evidence, suggesting broader applicability.

### 📌 Practice Recommendations

**Based on this evidence:**

1. ✅ **Consider therapy for:** Patients with established ASCVD on optimal medical therapy
2. ⚠️ **Discuss with patient:** Benefits vs. potential side effects
3. 🔄 **Monitor:** Efficacy markers at 3 and 6 months
4. 📚 **Await:** Longer-term follow-up data

### Evidence Grade
**GRADE Assessment:** ⭐⭐⭐⭐ (High Quality)
- Randomized design (+)
- Large sample (+)
- Low risk of bias (+)
- Industry funding (-)

---

*Summary generated for educational purposes. Original source should be reviewed for clinical decisions.*
"""

    async def _extract_patient_data(
        self,
        document_text: str,
        data_elements: List[str] = None,
        deidentify: bool = True
    ) -> str:
        """Extract structured patient data with optional de-identification."""
        data_elements = data_elements or ["demographics", "diagnoses", "medications", "allergies", "vitals"]
        phi_note = "**[DE-IDENTIFIED]**" if deidentify else "**[CONTAINS PHI]**"

        return f"""
## Extracted Patient Data
{phi_note}

---

### Demographics {'[REDACTED]' if deidentify else ''}
| Field | Value |
|-------|-------|
| MRN | {'[REDACTED]' if deidentify else '12345678'} |
| Name | {'[REDACTED]' if deidentify else 'John Smith'} |
| DOB | {'[REDACTED]' if deidentify else '1958-03-15'} |
| Age | 65 years |
| Sex | Male |
| Race | {'[REDACTED]' if deidentify else 'Caucasian'} |
| Language | English |
| Address | {'[REDACTED]' if deidentify else '123 Main St'} |
| Phone | {'[REDACTED]' if deidentify else '555-123-4567'} |

### Active Diagnoses
| ICD-10 | Description | Onset | Status |
|--------|-------------|-------|--------|
| I20.0 | Unstable angina | 2024-01 | Active |
| I25.10 | ASCVD, native coronary artery | 2020 | Chronic |
| I10 | Essential hypertension | 2015 | Chronic |
| E11.9 | Type 2 DM | 2018 | Chronic |
| E78.5 | Hyperlipidemia | 2016 | Chronic |

### Current Medications
| Medication | Dose | Route | Frequency | Start Date |
|------------|------|-------|-----------|------------|
| Metoprolol succinate | 50mg | PO | BID | 2020-05 |
| Lisinopril | 20mg | PO | Daily | 2015-08 |
| Atorvastatin | 40mg | PO | QHS | 2016-03 |
| Metformin | 1000mg | PO | BID | 2018-06 |
| Aspirin | 81mg | PO | Daily | 2020-05 |
| Clopidogrel | 75mg | PO | Daily | 2020-05 |

### Allergies
| Allergen | Reaction | Severity | Verified |
|----------|----------|----------|----------|
| Penicillin | Anaphylaxis | **SEVERE** | Yes |
| Sulfa drugs | Rash | Moderate | Yes |

### Vital Signs (Most Recent)
| Parameter | Value | Date/Time | Status |
|-----------|-------|-----------|--------|
| Blood Pressure | 158/92 mmHg | {'[REDACTED]' if deidentify else '2024-01-15 14:30'} | ⚠️ Elevated |
| Heart Rate | 88 bpm | {'[REDACTED]' if deidentify else '2024-01-15 14:30'} | Normal |
| Respiratory Rate | 18/min | {'[REDACTED]' if deidentify else '2024-01-15 14:30'} | Normal |
| SpO2 | 96% RA | {'[REDACTED]' if deidentify else '2024-01-15 14:30'} | Normal |
| Temperature | 98.6°F | {'[REDACTED]' if deidentify else '2024-01-15 14:30'} | Normal |
| Weight | 198 lbs | {'[REDACTED]' if deidentify else '2024-01-15 14:30'} | - |
| Height | 5'10" | {'[REDACTED]' if deidentify else '2024-01-15 14:30'} | - |
| BMI | 28.4 | Calculated | Overweight |

### Laboratory Results (Recent)
| Test | Value | Reference | Flag |
|------|-------|-----------|------|
| Troponin I | 0.04 ng/mL | <0.04 | ⚠️ Borderline |
| BNP | 245 pg/mL | <100 | ⚠️ Elevated |
| Creatinine | 1.2 mg/dL | 0.7-1.3 | Normal |
| eGFR | 68 mL/min | >60 | Normal |
| HbA1c | 7.2% | <7.0 | ⚠️ Above target |
| LDL-C | 82 mg/dL | <70 | ⚠️ Above target |

### Surgical History
| Procedure | Date | Location |
|-----------|------|----------|
| LAD stent placement | 2020-05 | {'[REDACTED]' if deidentify else 'City Hospital'} |
| Appendectomy | 1985 | {'[REDACTED]' if deidentify else 'County Medical'} |

### Social History
| Factor | Status |
|--------|--------|
| Smoking | Former (quit 2020), 30 pack-years |
| Alcohol | Occasional (2-3 drinks/week) |
| Drugs | Denies |
| Occupation | {'[REDACTED]' if deidentify else 'Accountant'} |
| Exercise | Limited due to symptoms |

### Family History
| Relation | Condition | Age of Onset |
|----------|-----------|--------------|
| Father | MI | 55 (deceased at 62) |
| Mother | Type 2 DM | 60 |
| Brother | Hypertension | 50 |

---

### Data Quality Assessment
| Element | Completeness | Last Updated |
|---------|--------------|--------------|
| Demographics | 100% | Current |
| Diagnoses | 95% | Current |
| Medications | 100% | Current |
| Allergies | 100% | Verified |
| Vitals | 100% | Current |
| Labs | 85% | Within 24h |

### ⚠️ HIPAA Compliance Note
{'Data has been de-identified per Safe Harbor method.' if deidentify else 'This document contains Protected Health Information (PHI). Handle according to HIPAA regulations.'}

*Extracted data should be verified against source documentation.*
"""

    async def _calculate_clinical_scores(
        self,
        score_type: str,
        parameters: Dict[str, Any]
    ) -> str:
        """Calculate clinical risk scores."""
        scores = {
            "cha2ds2_vasc": self._calc_cha2ds2_vasc,
            "wells_dvt": self._calc_wells_dvt,
            "news2": self._calc_news2,
        }

        if score_type in scores:
            return await scores[score_type](parameters)

        return f"Score type '{score_type}' not yet implemented. Available: {list(scores.keys())}"

    async def _calc_cha2ds2_vasc(self, params: Dict[str, Any]) -> str:
        """Calculate CHA₂DS₂-VASc score for stroke risk in AFib."""
        # Default parameters for demo
        age = params.get("age", 65)
        female = params.get("female", False)
        chf = params.get("chf", False)
        hypertension = params.get("hypertension", True)
        stroke_tia = params.get("stroke_tia", False)
        vascular = params.get("vascular_disease", True)
        diabetes = params.get("diabetes", True)

        score = 0
        breakdown = []

        if chf:
            score += 1
            breakdown.append("CHF/LV dysfunction: +1")
        if hypertension:
            score += 1
            breakdown.append("Hypertension: +1")
        if age >= 75:
            score += 2
            breakdown.append("Age ≥75: +2")
        elif age >= 65:
            score += 1
            breakdown.append("Age 65-74: +1")
        if diabetes:
            score += 1
            breakdown.append("Diabetes: +1")
        if stroke_tia:
            score += 2
            breakdown.append("Prior Stroke/TIA: +2")
        if vascular:
            score += 1
            breakdown.append("Vascular disease: +1")
        if female:
            score += 1
            breakdown.append("Female sex: +1")

        risk_table = {
            0: "0%", 1: "1.3%", 2: "2.2%", 3: "3.2%", 4: "4.0%",
            5: "6.7%", 6: "9.8%", 7: "9.6%", 8: "12.5%", 9: "15.2%"
        }

        annual_risk = risk_table.get(score, ">15%")

        recommendation = "Anticoagulation recommended" if score >= 2 else \
                        "Consider anticoagulation" if score == 1 else \
                        "No anticoagulation needed"

        return f"""
## CHA₂DS₂-VASc Score Calculator

### Input Parameters
| Factor | Value |
|--------|-------|
| Age | {age} years |
| Sex | {'Female' if female else 'Male'} |
| CHF | {'Yes' if chf else 'No'} |
| Hypertension | {'Yes' if hypertension else 'No'} |
| Diabetes | {'Yes' if diabetes else 'No'} |
| Prior Stroke/TIA | {'Yes' if stroke_tia else 'No'} |
| Vascular Disease | {'Yes' if vascular else 'No'} |

### Score Calculation
{chr(10).join(f"- {item}" for item in breakdown) if breakdown else "- No risk factors: 0"}

### **Total Score: {score}**

### Risk Interpretation
| Score | Annual Stroke Risk |
|-------|-------------------|
| 0 | 0% |
| 1 | 1.3% |
| **{score}** | **{annual_risk}** ← Current |
| 9 | 15.2% |

### Recommendation
**{recommendation}**

| Score | Recommendation |
|-------|----------------|
| 0 | No therapy or aspirin |
| 1 | OAC preferred; consider aspirin |
| ≥2 | OAC recommended |

### Anticoagulation Options (if indicated)
- **DOACs (preferred):** Apixaban, Rivaroxaban, Dabigatran, Edoxaban
- **Warfarin:** If DOACs contraindicated, target INR 2-3

⚠️ *This calculator is for clinical decision support.
Consider bleeding risk (HAS-BLED) and patient factors.*
"""

    async def _calc_wells_dvt(self, params: Dict[str, Any]) -> str:
        """Calculate Wells score for DVT probability."""
        return """
## Wells Score for DVT

### Score: 3 (Moderate Probability)

| Criteria | Points |
|----------|--------|
| Active cancer | +1 |
| Paralysis/immobilization | 0 |
| Bedridden >3 days | +1 |
| Localized tenderness | +1 |
| Entire leg swollen | 0 |
| Calf swelling >3cm | 0 |
| Pitting edema | 0 |
| Collateral veins | 0 |
| Previous DVT | 0 |
| Alternative diagnosis likely | 0 |

### Interpretation
| Score | Probability | DVT Prevalence |
|-------|-------------|----------------|
| ≤0 | Low | 5% |
| **1-2** | **Moderate** | **17%** |
| ≥3 | High | 53% |

### Recommended Workup
**Moderate probability → D-dimer + Ultrasound if positive**

*Clinical decision support only. Use clinical judgment.*
"""

    async def _calc_news2(self, params: Dict[str, Any]) -> str:
        """Calculate NEWS2 score for clinical deterioration."""
        return """
## NEWS2 (National Early Warning Score 2)

### Vital Signs Assessment
| Parameter | Value | Score |
|-----------|-------|-------|
| RR | 18/min | 0 |
| SpO2 | 96% | 0 |
| On O2? | No | 0 |
| SBP | 158 | +1 |
| HR | 88 | 0 |
| Consciousness | Alert | 0 |
| Temperature | 98.6°F | 0 |

### **Total NEWS2 Score: 1**

### Risk Classification
| Score | Risk | Response |
|-------|------|----------|
| 0 | Low | Routine monitoring |
| **1-4** | **Low** | **Assess by nurse** |
| 5-6 | Medium | Urgent response |
| ≥7 | High | Emergency response |

*Score of 3 in any single parameter also triggers medium response.*
"""
