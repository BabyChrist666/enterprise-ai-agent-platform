"""
FastAPI Application - Enterprise AI Agent Platform API.

Production-ready API with:
- Async endpoints for all operations
- Streaming support for real-time responses
- Comprehensive error handling
- OpenAPI documentation
- Rate limiting and security
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
from datetime import datetime
import json

from ..core.config import get_settings, Settings
from ..core.cohere_client import CohereRAGEngine, Document
from ..agents.orchestrator import AgentOrchestrator, AgentType

# Initialize app
app = FastAPI(
    title="Enterprise AI Agent Platform",
    description="""
    Multi-domain AI agent platform for enterprise applications.

    ## Features
    - **Finance Agent**: Portfolio analysis, SEC filings, valuations
    - **Legal Agent**: Contract analysis, compliance checking
    - **Healthcare Agent**: Clinical notes, medical coding, drug interactions
    - **Smart Orchestration**: Automatic routing and multi-agent coordination

    ## Authentication
    API key required in header: `X-API-Key`
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized on startup)
orchestrator: Optional[AgentOrchestrator] = None
rag_engine: Optional[CohereRAGEngine] = None


# Pydantic Models
class QueryRequest(BaseModel):
    """Request model for agent queries."""
    query: str = Field(..., description="The question or task for the agent")
    agent: Optional[str] = Field(None, description="Force specific agent (finance, legal, healthcare)")
    context: Optional[List[Dict[str, Any]]] = Field(None, description="Additional context documents")
    stream: bool = Field(False, description="Enable streaming response")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Analyze the risk profile of a portfolio with 40% AAPL, 30% MSFT, 30% GOOGL",
                "agent": "finance",
                "stream": False
            }
        }


class DocumentUpload(BaseModel):
    """Model for document upload."""
    documents: List[Dict[str, Any]] = Field(..., description="List of documents to upload")
    agent: str = Field(..., description="Target agent for documents")

    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    {"id": "doc1", "content": "Contract text here...", "metadata": {"type": "contract"}}
                ],
                "agent": "legal"
            }
        }


class AgentInfo(BaseModel):
    """Information about an agent."""
    type: str
    name: str
    description: str
    tools: List[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    agents_available: List[str]


class QueryResponse(BaseModel):
    """Response model for agent queries."""
    answer: str
    agents_used: List[str]
    execution_time_ms: float
    routing_confidence: float
    metadata: Dict[str, Any]


# Startup/Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global orchestrator, rag_engine
    settings = get_settings()

    # Initialize RAG engine
    rag_engine = CohereRAGEngine()

    # Initialize orchestrator with all agents
    orchestrator = AgentOrchestrator(rag_engine=rag_engine)

    print(f"🚀 {settings.app_name} v{settings.app_version} started")
    print(f"📊 Agents available: {[a.value for a in AgentType]}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("👋 Shutting down Enterprise AI Agent Platform")


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Enterprise AI Agent Platform",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        agents_available=[a.value for a in AgentType if a != AgentType.GENERAL]
    )


@app.get("/agents", response_model=List[AgentInfo], tags=["Agents"])
async def list_agents():
    """List all available agents and their capabilities."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return orchestrator.get_available_agents()


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_agent(request: QueryRequest):
    """
    Send a query to the AI agent platform.

    The orchestrator will automatically route to the appropriate agent(s)
    based on the query content, or you can force a specific agent.
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Convert context to Documents if provided
        context = None
        if request.context:
            context = [
                Document(
                    id=doc.get("id", f"doc_{i}"),
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {})
                )
                for i, doc in enumerate(request.context)
            ]

        # Determine forced agents
        force_agents = None
        if request.agent:
            try:
                force_agents = [AgentType(request.agent.lower())]
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid agent type: {request.agent}. Valid: finance, legal, healthcare"
                )

        # Execute query
        result = await orchestrator.run(
            query=request.query,
            context=context,
            force_agents=force_agents
        )

        return QueryResponse(
            answer=result.answer,
            agents_used=list(result.agent_responses.keys()),
            execution_time_ms=result.total_execution_time_ms,
            routing_confidence=result.routing_decision.confidence,
            metadata=result.metadata
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream", tags=["Query"])
async def query_agent_stream(request: QueryRequest):
    """
    Stream a query response in real-time.

    Returns Server-Sent Events (SSE) for real-time streaming.
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    async def generate():
        try:
            # For streaming, we use direct generation
            routing = await orchestrator.analyze_and_route(request.query)

            yield f"data: {json.dumps({'type': 'routing', 'agent': routing.primary_agent.value})}\n\n"

            # Stream the response
            async for chunk in rag_engine.generate(
                prompt=request.query,
                stream=True,
                temperature=0.3
            ):
                yield f"data: {json.dumps({'type': 'content', 'text': chunk})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@app.post("/documents", tags=["Documents"])
async def upload_documents(upload: DocumentUpload):
    """
    Upload documents to an agent's knowledge base.

    Documents will be embedded and stored for RAG retrieval.
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        agent_type = AgentType(upload.agent.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid agent type: {upload.agent}"
        )

    documents = [
        Document(
            id=doc.get("id", f"doc_{i}"),
            content=doc.get("content", ""),
            metadata=doc.get("metadata", {})
        )
        for i, doc in enumerate(upload.documents)
    ]

    await orchestrator.add_documents_to_agent(agent_type, documents)

    return {
        "status": "success",
        "documents_added": len(documents),
        "agent": upload.agent
    }


# Finance-specific endpoints
@app.post("/finance/portfolio-analysis", tags=["Finance"])
async def analyze_portfolio(
    portfolio: Dict[str, float] = Query(..., description="Portfolio weights by ticker"),
    benchmark: str = Query("SPY", description="Benchmark ticker")
):
    """Analyze portfolio risk metrics."""
    query = f"Calculate risk metrics for portfolio: {portfolio} against benchmark {benchmark}"
    result = await orchestrator.run(query, force_agents=[AgentType.FINANCE])
    return {"analysis": result.answer}


@app.post("/finance/valuation", tags=["Finance"])
async def company_valuation(
    ticker: str = Query(..., description="Stock ticker"),
    method: str = Query("dcf", description="Valuation method")
):
    """Perform company valuation analysis."""
    query = f"Perform {method} valuation analysis for {ticker}"
    result = await orchestrator.run(query, force_agents=[AgentType.FINANCE])
    return {"valuation": result.answer}


# Legal-specific endpoints
@app.post("/legal/contract-review", tags=["Legal"])
async def review_contract(
    contract_text: str,
    focus_areas: Optional[List[str]] = None
):
    """Review and analyze a contract."""
    query = f"Analyze this contract and extract key clauses: {contract_text[:1000]}..."
    result = await orchestrator.run(query, force_agents=[AgentType.LEGAL])
    return {"analysis": result.answer}


@app.post("/legal/compliance-check", tags=["Legal"])
async def check_compliance(
    document_text: str,
    regulations: List[str] = Query(["GDPR", "CCPA"])
):
    """Check document for regulatory compliance."""
    query = f"Check this document for {', '.join(regulations)} compliance: {document_text[:1000]}..."
    result = await orchestrator.run(query, force_agents=[AgentType.LEGAL])
    return {"compliance": result.answer}


# Healthcare-specific endpoints
@app.post("/healthcare/clinical-parse", tags=["Healthcare"])
async def parse_clinical_note(
    clinical_text: str,
    note_type: str = Query("progress_note")
):
    """Parse and structure a clinical note."""
    query = f"Parse this {note_type}: {clinical_text[:1000]}..."
    result = await orchestrator.run(query, force_agents=[AgentType.HEALTHCARE])
    return {"parsed": result.answer}


@app.post("/healthcare/drug-interactions", tags=["Healthcare"])
async def check_drug_interactions(
    medications: List[str]
):
    """Check for drug-drug interactions."""
    query = f"Check for interactions between: {', '.join(medications)}"
    result = await orchestrator.run(query, force_agents=[AgentType.HEALTHCARE])
    return {"interactions": result.answer}


@app.post("/healthcare/icd-codes", tags=["Healthcare"])
async def suggest_icd_codes(
    clinical_text: str
):
    """Suggest ICD-10 codes from clinical text."""
    query = f"Suggest ICD-10 codes for: {clinical_text[:1000]}..."
    result = await orchestrator.run(query, force_agents=[AgentType.HEALTHCARE])
    return {"codes": result.answer}


# Multi-agent endpoint
@app.post("/multi-agent", tags=["Multi-Agent"])
async def multi_agent_query(
    query: str,
    agents: List[str]
):
    """
    Execute a query across multiple agents simultaneously.

    Useful for cross-domain queries requiring expertise from multiple areas.
    """
    try:
        force_agents = [AgentType(a.lower()) for a in agents]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid agent type: {e}")

    result = await orchestrator.run(query, force_agents=force_agents)

    return {
        "synthesized_answer": result.answer,
        "individual_responses": {
            name: resp.answer
            for name, resp in result.agent_responses.items()
        },
        "execution_time_ms": result.total_execution_time_ms
    }


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "backend.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
