"""Hybrid RAG com busca densa e esparsa."""

from retrieval_strategies.base import RAGRetrievalStrategy
from utils.vector_store import SearchResult
from config import HYBRID_CANDIDATES


class HybridRAG(RAGRetrievalStrategy):
    """Combina busca semantica densa com busca lexical esparsa."""

    name = "hybrid"

    def retrieve(self, query: str) -> list[SearchResult]:
        query_embedding = self.get_query_embedding(query)
        return self.search_hybrid(query_embedding.dense, query_embedding.sparse, top_k=HYBRID_CANDIDATES)

