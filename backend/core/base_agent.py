"""
Base Agent Framework with Tool Use and Multi-Step Reasoning.
This is the foundation for all domain-specific agents.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio
from datetime import datetime
from .cohere_client import CohereRAGEngine, Document, VectorStore
from .config import get_settings

settings = get_settings()


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING_TOOL = "executing_tool"
    GENERATING = "generating"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class Tool:
    """Represents a tool the agent can use."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema for parameters
    function: Callable[..., Awaitable[Any]]


@dataclass
class AgentThought:
    """A single step in the agent's reasoning chain."""
    step: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentResponse:
    """Complete agent response with reasoning trace."""
    answer: str
    thoughts: List[AgentThought]
    sources: List[Document]
    metadata: Dict[str, Any]
    execution_time_ms: float


class BaseAgent(ABC):
    """
    Abstract base class for domain-specific AI agents.

    Features:
    - Multi-step reasoning with ReAct pattern
    - Tool use with automatic parameter extraction
    - RAG integration for grounded responses
    - Full reasoning trace for transparency
    """

    def __init__(
        self,
        name: str,
        description: str,
        rag_engine: Optional[CohereRAGEngine] = None,
        vector_store: Optional[VectorStore] = None
    ):
        self.name = name
        self.description = description
        self.rag_engine = rag_engine or CohereRAGEngine()
        self.vector_store = vector_store or VectorStore()
        self.tools: Dict[str, Tool] = {}
        self.status = AgentStatus.IDLE
        self.max_iterations = settings.max_agent_iterations

        # Register default tools
        self._register_default_tools()
        # Let subclasses register their tools
        self._register_tools()

    def _register_default_tools(self):
        """Register tools available to all agents."""
        self.register_tool(Tool(
            name="search_knowledge_base",
            description="Search the knowledge base for relevant information. Use this when you need to find specific facts or context.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            },
            function=self._search_knowledge_base
        ))

    @abstractmethod
    def _register_tools(self):
        """Register domain-specific tools. Implemented by subclasses."""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent. Implemented by subclasses."""
        pass

    def register_tool(self, tool: Tool):
        """Register a tool for the agent to use."""
        self.tools[tool.name] = tool

    async def _search_knowledge_base(
        self,
        query: str,
        top_k: int = 5
    ) -> str:
        """Search the vector store and rerank results."""
        # Get query embedding
        query_embedding = await self.rag_engine.embed_query(query)

        # Initial retrieval
        candidates = self.vector_store.similarity_search(query_embedding, top_k=top_k * 2)

        if not candidates:
            return "No relevant documents found in the knowledge base."

        # Rerank for precision
        reranked = await self.rag_engine.rerank(query, candidates, top_n=top_k)

        # Format results
        results = []
        for i, result in enumerate(reranked, 1):
            results.append(f"[{i}] (Score: {result.relevance_score:.3f})\n{result.document.content[:500]}...")

        return "\n\n".join(results)

    def _build_tool_descriptions(self) -> str:
        """Build tool descriptions for the prompt."""
        descriptions = []
        for name, tool in self.tools.items():
            params = json.dumps(tool.parameters, indent=2)
            descriptions.append(f"**{name}**: {tool.description}\nParameters: {params}")
        return "\n\n".join(descriptions)

    async def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Execute a tool and return the result."""
        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'"

        tool = self.tools[tool_name]
        try:
            result = await tool.function(**parameters)
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    async def _reason_and_act(
        self,
        query: str,
        context: Optional[List[Document]] = None
    ) -> AgentResponse:
        """
        Execute multi-step reasoning using ReAct pattern.

        The agent will:
        1. Think about what to do
        2. Decide on an action (tool use or final answer)
        3. Execute the action
        4. Observe the result
        5. Repeat until done or max iterations
        """
        import time
        start_time = time.time()

        thoughts: List[AgentThought] = []
        sources: List[Document] = context or []

        system_prompt = f"""{self.get_system_prompt()}

## Available Tools
{self._build_tool_descriptions()}

## Response Format
You must respond in this exact format:

THOUGHT: [Your reasoning about what to do next]
ACTION: [Tool name to use, or "FINAL_ANSWER" if you have enough information]
ACTION_INPUT: [JSON parameters for the tool, or your final answer if ACTION is FINAL_ANSWER]

Always start with THOUGHT, then ACTION, then ACTION_INPUT.
"""

        conversation_history = f"User Query: {query}\n\n"

        for iteration in range(self.max_iterations):
            self.status = AgentStatus.THINKING

            # Get agent's next step
            response = await self.rag_engine.generate(
                prompt=conversation_history + "What is your next step?",
                system_prompt=system_prompt,
                temperature=0.2
            )

            # Parse response
            thought = self._extract_section(response, "THOUGHT")
            action = self._extract_section(response, "ACTION")
            action_input_raw = self._extract_section(response, "ACTION_INPUT")

            current_thought = AgentThought(
                step=iteration + 1,
                thought=thought,
                action=action
            )

            # Check if we have a final answer
            if action and action.strip().upper() == "FINAL_ANSWER":
                current_thought.observation = action_input_raw
                thoughts.append(current_thought)
                self.status = AgentStatus.COMPLETED

                execution_time = (time.time() - start_time) * 1000
                return AgentResponse(
                    answer=action_input_raw or thought,
                    thoughts=thoughts,
                    sources=sources,
                    metadata={"iterations": iteration + 1},
                    execution_time_ms=execution_time
                )

            # Execute tool
            if action:
                self.status = AgentStatus.EXECUTING_TOOL
                try:
                    action_input = json.loads(action_input_raw) if action_input_raw else {}
                except json.JSONDecodeError:
                    action_input = {"query": action_input_raw} if action_input_raw else {}

                current_thought.action_input = action_input
                observation = await self._execute_tool(action, action_input)
                current_thought.observation = observation

                # Add to conversation
                conversation_history += f"""
THOUGHT: {thought}
ACTION: {action}
ACTION_INPUT: {json.dumps(action_input)}
OBSERVATION: {observation}

"""

            thoughts.append(current_thought)

        # Max iterations reached
        self.status = AgentStatus.COMPLETED
        execution_time = (time.time() - start_time) * 1000

        return AgentResponse(
            answer="I was unable to complete the task within the allowed steps. Here's what I found so far: " +
                   (thoughts[-1].observation if thoughts else "No progress made."),
            thoughts=thoughts,
            sources=sources,
            metadata={"iterations": self.max_iterations, "incomplete": True},
            execution_time_ms=execution_time
        )

    def _extract_section(self, text: str, section: str) -> Optional[str]:
        """Extract a section from the agent's response."""
        import re
        pattern = rf"{section}:\s*(.+?)(?=(?:THOUGHT|ACTION|ACTION_INPUT|$))"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    async def run(
        self,
        query: str,
        context: Optional[List[Document]] = None
    ) -> AgentResponse:
        """
        Run the agent on a query.

        Args:
            query: User's question or task
            context: Optional pre-retrieved documents

        Returns:
            AgentResponse with answer and reasoning trace
        """
        self.status = AgentStatus.THINKING
        try:
            response = await self._reason_and_act(query, context)
            return response
        except Exception as e:
            self.status = AgentStatus.ERROR
            raise

    async def add_documents(self, documents: List[Document]):
        """Add documents to the agent's knowledge base."""
        # Generate embeddings
        texts = [doc.content for doc in documents]
        embeddings = await self.rag_engine.embed_documents(texts)

        # Attach embeddings and store
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding

        self.vector_store.add_documents(documents)
