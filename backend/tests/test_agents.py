"""
Tests for AI Agents.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Test Finance Agent
class TestFinanceAgent:
    """Tests for the Finance Agent."""

    @pytest.mark.asyncio
    async def test_risk_metrics_calculation(self):
        """Test portfolio risk metrics calculation."""
        from backend.agents.finance_agent import FinanceAgent

        # Mock the RAG engine
        with patch('backend.agents.finance_agent.CohereRAGEngine') as mock_rag:
            mock_rag_instance = MagicMock()
            mock_rag.return_value = mock_rag_instance

            agent = FinanceAgent()

            # Test the tool directly
            result = await agent._calculate_risk_metrics(
                portfolio={"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3},
                benchmark="SPY"
            )

            assert "Portfolio Risk Analysis" in result
            assert "Sharpe Ratio" in result
            assert "VaR" in result

    @pytest.mark.asyncio
    async def test_financial_ratios(self):
        """Test financial ratio analysis."""
        from backend.agents.finance_agent import FinanceAgent

        with patch('backend.agents.finance_agent.CohereRAGEngine'):
            agent = FinanceAgent()
            result = await agent._analyze_financial_ratios(ticker="AAPL")

            assert "P/E" in result
            assert "ROE" in result
            assert "Debt/Equity" in result

    @pytest.mark.asyncio
    async def test_dcf_valuation(self):
        """Test DCF valuation calculation."""
        from backend.agents.finance_agent import FinanceAgent

        with patch('backend.agents.finance_agent.CohereRAGEngine'):
            agent = FinanceAgent()
            result = await agent._calculate_dcf_valuation(
                ticker="AAPL",
                growth_rate=0.10,
                discount_rate=0.10
            )

            assert "DCF Valuation" in result
            assert "Fair Value" in result
            assert "Sensitivity Analysis" in result


# Test Legal Agent
class TestLegalAgent:
    """Tests for the Legal Agent."""

    @pytest.mark.asyncio
    async def test_contract_clause_extraction(self):
        """Test contract clause extraction."""
        from backend.agents.legal_agent import LegalAgent

        with patch('backend.agents.legal_agent.CohereRAGEngine'):
            agent = LegalAgent()
            result = await agent._extract_contract_clauses(
                contract_text="Sample contract with indemnification clause..."
            )

            assert "Contract Clause Analysis" in result
            assert "Indemnification" in result
            assert "Risk Level" in result

    @pytest.mark.asyncio
    async def test_compliance_check(self):
        """Test compliance checking."""
        from backend.agents.legal_agent import LegalAgent

        with patch('backend.agents.legal_agent.CohereRAGEngine'):
            agent = LegalAgent()
            result = await agent._check_compliance(
                document_text="Privacy policy text...",
                regulations=["GDPR", "CCPA"]
            )

            assert "Compliance Check" in result
            assert "GDPR" in result
            assert "CCPA" in result


# Test Healthcare Agent
class TestHealthcareAgent:
    """Tests for the Healthcare Agent."""

    @pytest.mark.asyncio
    async def test_clinical_note_parsing(self):
        """Test clinical note parsing."""
        from backend.agents.healthcare_agent import HealthcareAgent

        with patch('backend.agents.healthcare_agent.CohereRAGEngine'):
            agent = HealthcareAgent()
            result = await agent._parse_clinical_note(
                clinical_text="Patient presents with chest pain...",
                note_type="progress_note"
            )

            assert "Clinical Note Analysis" in result
            assert "Chief Complaint" in result
            assert "Assessment" in result

    @pytest.mark.asyncio
    async def test_drug_interaction_check(self):
        """Test drug interaction checking."""
        from backend.agents.healthcare_agent import HealthcareAgent

        with patch('backend.agents.healthcare_agent.CohereRAGEngine'):
            agent = HealthcareAgent()
            result = await agent._check_drug_interactions(
                medications=["Metformin", "Lisinopril", "Aspirin"]
            )

            assert "Drug Interaction" in result
            assert "Severity" in result

    @pytest.mark.asyncio
    async def test_icd_code_suggestions(self):
        """Test ICD-10 code suggestions."""
        from backend.agents.healthcare_agent import HealthcareAgent

        with patch('backend.agents.healthcare_agent.CohereRAGEngine'):
            agent = HealthcareAgent()
            result = await agent._suggest_icd_codes(
                clinical_text="Patient with unstable angina and hypertension"
            )

            assert "ICD-10" in result
            assert "Code" in result
            assert "Confidence" in result


# Test Orchestrator
class TestOrchestrator:
    """Tests for the Agent Orchestrator."""

    @pytest.mark.asyncio
    async def test_routing_finance_query(self):
        """Test routing to finance agent."""
        from backend.agents.orchestrator import AgentOrchestrator, AgentType

        with patch('backend.agents.orchestrator.CohereRAGEngine'):
            orchestrator = AgentOrchestrator()
            routing = await orchestrator.analyze_and_route(
                "What is the P/E ratio of AAPL stock?"
            )

            assert routing.primary_agent == AgentType.FINANCE
            assert routing.confidence > 0.5

    @pytest.mark.asyncio
    async def test_routing_legal_query(self):
        """Test routing to legal agent."""
        from backend.agents.orchestrator import AgentOrchestrator, AgentType

        with patch('backend.agents.orchestrator.CohereRAGEngine'):
            orchestrator = AgentOrchestrator()
            routing = await orchestrator.analyze_and_route(
                "Review this NDA contract for compliance issues"
            )

            assert routing.primary_agent == AgentType.LEGAL

    @pytest.mark.asyncio
    async def test_routing_healthcare_query(self):
        """Test routing to healthcare agent."""
        from backend.agents.orchestrator import AgentOrchestrator, AgentType

        with patch('backend.agents.orchestrator.CohereRAGEngine'):
            orchestrator = AgentOrchestrator()
            routing = await orchestrator.analyze_and_route(
                "Parse this clinical note and suggest ICD codes"
            )

            assert routing.primary_agent == AgentType.HEALTHCARE

    @pytest.mark.asyncio
    async def test_multi_domain_detection(self):
        """Test detection of multi-domain queries."""
        from backend.agents.orchestrator import AgentOrchestrator

        with patch('backend.agents.orchestrator.CohereRAGEngine'):
            orchestrator = AgentOrchestrator()
            routing = await orchestrator.analyze_and_route(
                "What are the legal and financial implications of a healthcare data breach?"
            )

            assert routing.requires_multi_agent or len(routing.secondary_agents) > 0


# Test API
class TestAPI:
    """Tests for the FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from backend.api.main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "agents_available" in data

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data

    def test_list_agents(self, client):
        """Test listing agents."""
        response = client.get("/agents")
        assert response.status_code == 200
        agents = response.json()
        assert len(agents) >= 3  # Finance, Legal, Healthcare
