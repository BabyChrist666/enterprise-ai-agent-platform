"""
Agent Orchestrator - Multi-agent coordination and task routing.

The orchestrator:
1. Analyzes incoming requests
2. Routes to appropriate specialized agent(s)
3. Coordinates multi-agent workflows
4. Aggregates and synthesizes responses
"""
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime

from ..core.base_agent import BaseAgent, AgentResponse, Document
from ..core.cohere_client import CohereRAGEngine
from .finance_agent import FinanceAgent
from .legal_agent import LegalAgent
from .healthcare_agent import HealthcareAgent


class AgentType(Enum):
    """Available agent types."""
    FINANCE = "finance"
    LEGAL = "legal"
    HEALTHCARE = "healthcare"
    GENERAL = "general"


@dataclass
class RoutingDecision:
    """Result of the routing analysis."""
    primary_agent: AgentType
    secondary_agents: List[AgentType]
    confidence: float
    reasoning: str
    requires_multi_agent: bool


@dataclass
class OrchestratorResponse:
    """Combined response from orchestrator."""
    answer: str
    agent_responses: Dict[str, AgentResponse]
    routing_decision: RoutingDecision
    total_execution_time_ms: float
    metadata: Dict[str, Any]


class AgentOrchestrator:
    """
    Intelligent orchestrator for multi-domain AI agent platform.

    Capabilities:
    - Smart routing based on query analysis
    - Parallel multi-agent execution
    - Response synthesis and aggregation
    - Cross-domain reasoning
    """

    def __init__(self, rag_engine: Optional[CohereRAGEngine] = None):
        self.rag_engine = rag_engine or CohereRAGEngine()

        # Initialize specialized agents
        self.agents: Dict[AgentType, BaseAgent] = {
            AgentType.FINANCE: FinanceAgent(rag_engine=self.rag_engine),
            AgentType.LEGAL: LegalAgent(rag_engine=self.rag_engine),
            AgentType.HEALTHCARE: HealthcareAgent(rag_engine=self.rag_engine),
        }

        # Domain keywords for routing
        self.domain_keywords = {
            AgentType.FINANCE: [
                "stock", "portfolio", "investment", "market", "trading", "financial",
                "earnings", "revenue", "profit", "loss", "SEC", "10-K", "10-Q",
                "valuation", "DCF", "P/E", "ROI", "risk", "hedge", "derivative",
                "bond", "equity", "dividend", "IPO", "M&A", "balance sheet"
            ],
            AgentType.LEGAL: [
                "contract", "agreement", "clause", "legal", "compliance", "GDPR",
                "HIPAA", "liability", "indemnification", "NDA", "confidential",
                "intellectual property", "IP", "trademark", "patent", "lawsuit",
                "litigation", "regulatory", "terms of service", "privacy policy"
            ],
            AgentType.HEALTHCARE: [
                "patient", "clinical", "medical", "diagnosis", "treatment",
                "medication", "drug", "prescription", "ICD", "CPT", "health",
                "disease", "symptom", "vital", "lab", "radiology", "surgery",
                "hospital", "physician", "nurse", "HIPAA", "PHI", "EHR"
            ]
        }

    async def analyze_and_route(self, query: str) -> RoutingDecision:
        """
        Analyze query and determine routing to appropriate agent(s).

        Uses keyword matching and semantic analysis for routing decisions.
        """
        query_lower = query.lower()

        # Score each domain based on keyword matches
        scores: Dict[AgentType, int] = {}
        for agent_type, keywords in self.domain_keywords.items():
            score = sum(1 for kw in keywords if kw.lower() in query_lower)
            scores[agent_type] = score

        # Determine primary agent
        max_score = max(scores.values()) if scores.values() else 0
        if max_score == 0:
            primary = AgentType.GENERAL
            confidence = 0.5
        else:
            primary = max(scores, key=scores.get)
            confidence = min(0.95, 0.5 + (max_score * 0.1))

        # Check for multi-domain queries
        secondary = []
        requires_multi = False
        for agent_type, score in scores.items():
            if agent_type != primary and score > 0:
                secondary.append(agent_type)
                if score >= max_score * 0.5:  # Significant presence
                    requires_multi = True

        reasoning = f"Detected {max_score} keyword matches for {primary.value} domain."
        if secondary:
            reasoning += f" Also found relevance to: {[a.value for a in secondary]}"

        return RoutingDecision(
            primary_agent=primary,
            secondary_agents=secondary,
            confidence=confidence,
            reasoning=reasoning,
            requires_multi_agent=requires_multi
        )

    async def execute_single_agent(
        self,
        agent_type: AgentType,
        query: str,
        context: Optional[List[Document]] = None
    ) -> AgentResponse:
        """Execute query on a single agent."""
        if agent_type == AgentType.GENERAL or agent_type not in self.agents:
            # Use general generation for unrouted queries
            response = await self.rag_engine.generate(
                prompt=query,
                context=context,
                system_prompt="You are a helpful AI assistant. Provide clear, accurate responses."
            )
            return AgentResponse(
                answer=response,
                thoughts=[],
                sources=context or [],
                metadata={"agent": "general"},
                execution_time_ms=0
            )

        agent = self.agents[agent_type]
        return await agent.run(query, context)

    async def execute_multi_agent(
        self,
        query: str,
        agents: List[AgentType],
        context: Optional[List[Document]] = None
    ) -> Dict[str, AgentResponse]:
        """Execute query on multiple agents in parallel."""
        tasks = []
        agent_names = []

        for agent_type in agents:
            if agent_type in self.agents:
                tasks.append(self.execute_single_agent(agent_type, query, context))
                agent_names.append(agent_type.value)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses = {}
        for name, result in zip(agent_names, results):
            if isinstance(result, Exception):
                responses[name] = AgentResponse(
                    answer=f"Error from {name} agent: {str(result)}",
                    thoughts=[],
                    sources=[],
                    metadata={"error": True},
                    execution_time_ms=0
                )
            else:
                responses[name] = result

        return responses

    async def synthesize_responses(
        self,
        query: str,
        responses: Dict[str, AgentResponse]
    ) -> str:
        """Synthesize multiple agent responses into a coherent answer."""
        if len(responses) == 1:
            return list(responses.values())[0].answer

        # Combine responses for synthesis
        combined = "You are synthesizing responses from multiple specialized AI agents.\n\n"
        combined += f"Original Query: {query}\n\n"

        for agent_name, response in responses.items():
            combined += f"=== {agent_name.upper()} AGENT RESPONSE ===\n"
            combined += f"{response.answer}\n\n"

        combined += """
Please synthesize these responses into a coherent, unified answer that:
1. Combines insights from all relevant agents
2. Highlights cross-domain considerations
3. Notes any conflicts or complementary information
4. Provides a clear, actionable summary
"""

        synthesis = await self.rag_engine.generate(
            prompt=combined,
            temperature=0.3,
            max_tokens=2048
        )

        return synthesis

    async def run(
        self,
        query: str,
        context: Optional[List[Document]] = None,
        force_agents: Optional[List[AgentType]] = None
    ) -> OrchestratorResponse:
        """
        Main entry point for the orchestrator.

        Args:
            query: User's question or task
            context: Optional pre-retrieved documents
            force_agents: Force specific agents (bypass routing)

        Returns:
            OrchestratorResponse with synthesized answer
        """
        import time
        start_time = time.time()

        # Route the query
        if force_agents:
            routing = RoutingDecision(
                primary_agent=force_agents[0],
                secondary_agents=force_agents[1:],
                confidence=1.0,
                reasoning="Agents forced by user",
                requires_multi_agent=len(force_agents) > 1
            )
        else:
            routing = await self.analyze_and_route(query)

        # Execute on appropriate agents
        if routing.requires_multi_agent:
            all_agents = [routing.primary_agent] + routing.secondary_agents
            agent_responses = await self.execute_multi_agent(query, all_agents, context)
            final_answer = await self.synthesize_responses(query, agent_responses)
        else:
            response = await self.execute_single_agent(routing.primary_agent, query, context)
            agent_responses = {routing.primary_agent.value: response}
            final_answer = response.answer

        execution_time = (time.time() - start_time) * 1000

        return OrchestratorResponse(
            answer=final_answer,
            agent_responses=agent_responses,
            routing_decision=routing,
            total_execution_time_ms=execution_time,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "agents_used": list(agent_responses.keys()),
                "multi_agent": routing.requires_multi_agent
            }
        )

    async def add_documents_to_agent(
        self,
        agent_type: AgentType,
        documents: List[Document]
    ):
        """Add documents to a specific agent's knowledge base."""
        if agent_type in self.agents:
            await self.agents[agent_type].add_documents(documents)

    def get_available_agents(self) -> List[Dict[str, Any]]:
        """Get information about available agents."""
        return [
            {
                "type": agent_type.value,
                "name": agent.name,
                "description": agent.description,
                "tools": list(agent.tools.keys())
            }
            for agent_type, agent in self.agents.items()
        ]
