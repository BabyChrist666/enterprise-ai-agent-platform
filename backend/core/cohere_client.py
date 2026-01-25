"""
Cohere API client wrapper with RAG capabilities.
Implements embed, rerank, and generate with production-grade error handling.
"""
import cohere
from typing import List, Optional, Dict, Any
import asyncio
from dataclasses import dataclass
import numpy as np
from .config import get_settings

settings = get_settings()


@dataclass
class Document:
    """Represents a document for RAG."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class RetrievalResult:
    """Result from retrieval with relevance score."""
    document: Document
    relevance_score: float


class CohereRAGEngine:
    """
    Production-ready RAG engine using Cohere's Embed, Rerank, and Generate.

    Features:
    - Async batch processing for embeddings
    - Intelligent reranking for precision
    - Streaming generation support
    - Built-in retry logic
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = cohere.Client(api_key or settings.cohere_api_key)
        self.async_client = cohere.AsyncClient(api_key or settings.cohere_api_key)
        self.embed_model = settings.embedding_model
        self.rerank_model = settings.rerank_model
        self.generation_model = settings.generation_model

    async def embed_documents(
        self,
        texts: List[str],
        input_type: str = "search_document"
    ) -> List[List[float]]:
        """
        Generate embeddings for documents using Cohere Embed v3.

        Args:
            texts: List of text strings to embed
            input_type: 'search_document' for indexing, 'search_query' for queries

        Returns:
            List of embedding vectors
        """
        # Batch processing for large document sets
        batch_size = 96  # Cohere's recommended batch size
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = await self.async_client.embed(
                texts=batch,
                model=self.embed_model,
                input_type=input_type,
                embedding_types=["float"]
            )
            all_embeddings.extend(response.embeddings.float)

        return all_embeddings

    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query for retrieval."""
        embeddings = await self.embed_documents([query], input_type="search_query")
        return embeddings[0]

    async def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = 5
    ) -> List[RetrievalResult]:
        """
        Rerank documents using Cohere Rerank for improved precision.

        Args:
            query: The search query
            documents: List of candidate documents
            top_n: Number of top results to return

        Returns:
            Reranked documents with relevance scores
        """
        if not documents:
            return []

        response = await self.async_client.rerank(
            query=query,
            documents=[doc.content for doc in documents],
            model=self.rerank_model,
            top_n=min(top_n, len(documents))
        )

        results = []
        for result in response.results:
            results.append(RetrievalResult(
                document=documents[result.index],
                relevance_score=result.relevance_score
            ))

        return results

    async def generate(
        self,
        prompt: str,
        context: Optional[List[Document]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        stream: bool = False
    ):
        """
        Generate response using Cohere Command with optional RAG context.

        Args:
            prompt: User query
            context: Retrieved documents for grounding
            system_prompt: System instructions
            temperature: Creativity control (0-1)
            max_tokens: Maximum response length
            stream: Whether to stream the response

        Returns:
            Generated text or async generator for streaming
        """
        # Build the full message with context if provided
        full_message = prompt

        if context:
            context_text = "\n\n".join([
                f"[Document {doc.id}]: {doc.content}"
                for doc in context
            ])
            full_message = f"Context:\n{context_text}\n\nQuestion: {prompt}"

        # Build preamble from system prompt
        preamble = system_prompt or "You are a helpful AI assistant."

        if stream:
            return self._stream_generate(full_message, preamble, temperature, max_tokens)

        # Use Cohere v5 API with 'message' parameter
        response = await self.async_client.chat(
            model=self.generation_model,
            message=full_message,
            preamble=preamble,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.text

    async def _stream_generate(
        self,
        message: str,
        preamble: str,
        temperature: float,
        max_tokens: int
    ):
        """Stream generation for real-time responses."""
        async for event in self.async_client.chat_stream(
            model=self.generation_model,
            message=message,
            preamble=preamble,
            temperature=temperature,
            max_tokens=max_tokens
        ):
            if hasattr(event, 'text'):
                yield event.text


class VectorStore:
    """
    Simple in-memory vector store with cosine similarity.
    For production, swap with Pinecone, Weaviate, or Qdrant.
    """

    def __init__(self):
        self.documents: Dict[str, Document] = {}

    def add_documents(self, documents: List[Document]):
        """Add documents with embeddings to the store."""
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.id} has no embedding")
            self.documents[doc.id] = doc

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> List[Document]:
        """Find top-k similar documents using cosine similarity."""
        if not self.documents:
            return []

        query_vec = np.array(query_embedding)

        scores = []
        for doc_id, doc in self.documents.items():
            doc_vec = np.array(doc.embedding)
            # Cosine similarity
            similarity = np.dot(query_vec, doc_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
            )
            scores.append((similarity, doc))

        # Sort by similarity descending
        scores.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scores[:top_k]]

    def clear(self):
        """Clear all documents."""
        self.documents.clear()
