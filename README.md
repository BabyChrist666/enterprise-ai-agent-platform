# 🚀 Enterprise AI Agent Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Cohere](https://img.shields.io/badge/Cohere-Command%20R+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-teal.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Multi-domain AI Agent Platform for Enterprise Applications**

*Finance • Legal • Healthcare • Powered by Cohere*

[Demo](#-quick-start) • [Documentation](#-api-documentation) • [Architecture](#-architecture) • [Contributing](#-contributing)

</div>

---

## 🎯 Overview

Enterprise AI Agent Platform is a production-ready, multi-domain AI system that leverages **Cohere's state-of-the-art LLMs** for specialized enterprise tasks. The platform features intelligent orchestration, domain-specific agents, and a robust RAG pipeline.

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🏦 **Finance Agent** | Portfolio analysis, SEC filing parsing, DCF valuations, risk metrics |
| ⚖️ **Legal Agent** | Contract analysis, compliance checking, NDA review, risk assessment |
| 🏥 **Healthcare Agent** | Clinical note parsing, ICD-10 coding, drug interactions, HIPAA-aware |
| 🧠 **Smart Orchestration** | Automatic routing, multi-agent coordination, response synthesis |
| 🔍 **Advanced RAG** | Cohere Embed v3 + Rerank for precise retrieval |
| ⚡ **Async & Streaming** | Real-time responses with SSE support |
| 🛡️ **Production Ready** | Error handling, rate limiting, comprehensive logging |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Server                          │
│                    (Async, Streaming, OpenAPI)                  │
├─────────────────────────────────────────────────────────────────┤
│                      Agent Orchestrator                         │
│            (Routing • Coordination • Synthesis)                 │
├────────────────┬────────────────┬────────────────┬──────────────┤
│   Finance      │     Legal      │   Healthcare   │    Custom    │
│    Agent       │     Agent      │     Agent      │    Agent     │
│                │                │                │              │
│ • Risk Metrics │ • Contracts    │ • Clinical NLP │ • Your       │
│ • Valuations   │ • Compliance   │ • ICD Coding   │   Domain     │
│ • SEC Filings  │ • NDA Review   │ • Drug Check   │              │
├────────────────┴────────────────┴────────────────┴──────────────┤
│                     Cohere RAG Engine                           │
│         (Embed v3 → Vector Store → Rerank → Generate)          │
├─────────────────────────────────────────────────────────────────┤
│                    Cohere Command R+                            │
│              (Generation • Tool Use • Reasoning)                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Cohere API key ([Get one free](https://dashboard.cohere.com/))

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/enterprise-ai-agent-platform.git
cd enterprise-ai-agent-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your COHERE_API_KEY
```

### Run the Server

```bash
# Development
uvicorn backend.api.main:app --reload --port 8000

# Production
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Try It Out

```bash
# Health check
curl http://localhost:8000/health

# Query the finance agent
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Analyze risk metrics for a portfolio with 40% AAPL, 30% MSFT, 30% GOOGL",
    "agent": "finance"
  }'

# Multi-agent query
curl -X POST http://localhost:8000/multi-agent \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the legal and financial implications of a healthcare data breach?",
    "agents": ["legal", "finance", "healthcare"]
  }'
```

---

## 📖 API Documentation

### Interactive Docs

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Send query to auto-routed agent |
| `/query/stream` | POST | Stream response in real-time |
| `/multi-agent` | POST | Query multiple agents simultaneously |
| `/agents` | GET | List available agents and tools |
| `/documents` | POST | Upload documents to agent knowledge base |

### Domain-Specific Endpoints

#### Finance
| Endpoint | Description |
|----------|-------------|
| `/finance/portfolio-analysis` | Analyze portfolio risk metrics |
| `/finance/valuation` | DCF and comparable valuations |

#### Legal
| Endpoint | Description |
|----------|-------------|
| `/legal/contract-review` | Extract and analyze contract clauses |
| `/legal/compliance-check` | Check GDPR, CCPA, HIPAA compliance |

#### Healthcare
| Endpoint | Description |
|----------|-------------|
| `/healthcare/clinical-parse` | Parse clinical notes to structured data |
| `/healthcare/drug-interactions` | Check medication interactions |
| `/healthcare/icd-codes` | Suggest ICD-10 codes from clinical text |

---

## 🤖 Agent Capabilities

### Finance Agent

```python
# Portfolio Risk Analysis
{
  "query": "Calculate risk metrics for portfolio",
  "tools": [
    "calculate_risk_metrics",    # VaR, Sharpe, Beta, Volatility
    "analyze_financial_ratios",  # P/E, ROE, Debt/Equity
    "parse_sec_filing",          # 10-K, 10-Q, 8-K extraction
    "analyze_earnings",          # Earnings call analysis
    "calculate_dcf_valuation"    # Discounted cash flow
  ]
}
```

### Legal Agent

```python
# Contract Analysis
{
  "query": "Review this NDA for risks",
  "tools": [
    "extract_contract_clauses",  # Clause identification
    "analyze_nda",               # NDA-specific analysis
    "check_compliance",          # GDPR, CCPA, HIPAA
    "compare_contracts",         # Version comparison
    "assess_legal_risk"          # Risk assessment
  ]
}
```

### Healthcare Agent

```python
# Clinical Documentation
{
  "query": "Parse this clinical note",
  "tools": [
    "parse_clinical_note",       # Structured extraction
    "suggest_icd_codes",         # ICD-10/CPT coding
    "check_drug_interactions",   # DDI checking
    "summarize_medical_literature", # Research summaries
    "extract_patient_data",      # PHI extraction (HIPAA-aware)
    "calculate_clinical_scores"  # CHA2DS2-VASc, Wells, NEWS2
  ]
}
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `COHERE_API_KEY` | Your Cohere API key | Required |
| `EMBEDDING_MODEL` | Cohere embedding model | `embed-english-v3.0` |
| `RERANK_MODEL` | Cohere rerank model | `rerank-english-v3.0` |
| `GENERATION_MODEL` | Cohere generation model | `command-r-plus` |
| `MAX_AGENT_ITERATIONS` | Max reasoning steps | `10` |
| `DEBUG` | Enable debug mode | `false` |

### Adding Custom Agents

```python
from backend.core.base_agent import BaseAgent, Tool

class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="CustomAgent",
            description="Your custom domain agent"
        )

    def get_system_prompt(self) -> str:
        return "You are an expert in..."

    def _register_tools(self):
        self.register_tool(Tool(
            name="custom_tool",
            description="Does something useful",
            parameters={...},
            function=self._custom_function
        ))
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=backend --cov-report=html

# Specific test file
pytest tests/test_finance_agent.py -v
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Average Response Time | ~2-5 seconds |
| Streaming First Token | ~500ms |
| Multi-Agent Coordination | Parallel execution |
| RAG Retrieval | Sub-second |

---

## 🛣️ Roadmap

- [x] Core agent framework
- [x] Finance, Legal, Healthcare agents
- [x] Smart orchestration
- [x] RAG with Cohere Embed + Rerank
- [x] FastAPI backend
- [ ] React dashboard frontend
- [ ] Vector store integrations (Pinecone, Weaviate)
- [ ] Authentication & multi-tenancy
- [ ] Agent memory & conversation history
- [ ] Custom agent builder UI

---

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) first.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Cohere](https://cohere.com/) for their incredible LLM APIs
- [FastAPI](https://fastapi.tiangolo.com/) for the blazing-fast framework
- The open-source AI community

---

<div align="center">

**Built with ❤️ for the Cohere community**

[⬆ Back to Top](#-enterprise-ai-agent-platform)

</div>
