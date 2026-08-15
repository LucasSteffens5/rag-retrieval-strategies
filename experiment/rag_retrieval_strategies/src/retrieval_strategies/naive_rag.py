"""Naive RAG com busca vetorial densa."""

from retrieval_strategies.base import RAGRetrievalStrategy
from utils.vector_store import SearchResult
from config import DENSE_CANDIDATES


class NaiveRAG(RAGRetrievalStrategy):
    """RAG de linha de base com busca vetorial densa pura."""

    name = "naive"

    def retrieve(self, query: str) -> list[SearchResult]:
        """Recupera documentos por busca densa com embedding."""
        query_embedding = self.get_query_embedding(query)
        return self.search_dense(query_embedding.dense, top_k=DENSE_CANDIDATES)
