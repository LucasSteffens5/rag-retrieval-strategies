"""Define a interface base das estratégias RAG."""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from config import (
    MAX_CONTEXTS_PER_DOCUMENT,
    RETRIEVAL_TOP_K,
)
from prompts import (
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT_TEMPLATE,
)
from utils.timer import Timer
from utils.llm_client import OllamaClient
from utils.vector_store import VectorStore, SearchResult
from utils.embedder import EmbeddingManager, EmbeddingResult


@dataclass
class RAGResult:
    """Resultado completo de uma execução RAG."""
    query: str
    answer: str
    context_details: list[dict[str, Any]]
    model_name: str
    retrieval_strategy_name: str
    embedding_model: str
    embedding_ms: float = 0.0
    vector_search_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    routing_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

class RAGRetrievalStrategy(ABC):
    """Interface base para estratégias de recuperação RAG."""

    name: str = "base"

    def __init__(
        self,
        llm: OllamaClient,
        store: VectorStore,
        embedder: EmbeddingManager,
        collection: str,
        top_k: int = RETRIEVAL_TOP_K,
    ):
        """Inicializa dependências compartilhadas e acumuladores de latência."""
        self.llm = llm
        self.store = store
        self.embedder = embedder
        self.collection = collection
        self.top_k = top_k
        self._embedding_ms = 0.0
        self._vector_search_ms = 0.0

    def get_query_embedding(self, query: str) -> EmbeddingResult:
        """Gera embedding online e acumula sua latencia."""
        start = time.perf_counter()
        result = self.embedder.embed_query(query)
        self._embedding_ms += (time.perf_counter() - start) * 1000
        return result

    def reset_retrieval_timing(self) -> None:
        """Reinicia acumuladores de latencia da etapa de recuperacao."""
        self._embedding_ms = 0.0
        self._vector_search_ms = 0.0

    def search_dense(self, query_vector: list[float], top_k: int | None = None) -> list[SearchResult]:
        """Executa busca densa e acumula somente o tempo de consulta ao Qdrant."""
        start = time.perf_counter()
        results = self.store.search_dense(
            collection=self.collection,
            query_vector=query_vector,
            top_k=top_k or self.top_k,
        )
        self._vector_search_ms += (time.perf_counter() - start) * 1000
        return results

    def search_hybrid(
        self,
        dense_vector: list[float],
        sparse_vector: dict[str, Any],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Executa busca hibrida e acumula somente o tempo de consulta ao Qdrant."""
        start = time.perf_counter()
        results = self.store.search_hybrid(
            collection=self.collection,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=top_k or self.top_k,
        )
        self._vector_search_ms += (time.perf_counter() - start) * 1000
        return results

    @abstractmethod
    def retrieve(self, query: str) -> list[SearchResult]:
        """Recupera documentos relevantes para a consulta."""
        pass

    def build_prompt(self, query: str, contexts: list[SearchResult]) -> str:
        """Monta o prompt RAG com consulta e contextos recuperados."""
        context_text = "\n\n---\n\n".join(self.format_context(context) for context in contexts)
        return RAG_USER_PROMPT_TEMPLATE.format(context=context_text, query=query)

    def format_context(self, context: SearchResult) -> str:
        """Formata evidencia usando apenas fonte e texto original."""
        metadata_lines = [
            f"[Fonte: {context.metadata.get('filename', 'unknown')}]",
            context.text,
        ]
        return "\n".join(str(line) for line in metadata_lines if str(line).strip())

    def select_contexts(self, contexts: list[SearchResult]) -> list[SearchResult]:
        """Aplica diversidade documental conservadora ao ranking final."""
        selected: list[SearchResult] = []
        counts: dict[str, int] = {}
        for context in contexts:
            document_id = str(context.metadata.get("document_id") or context.doc_id)
            if counts.get(document_id, 0) >= MAX_CONTEXTS_PER_DOCUMENT:
                continue
            selected.append(context)
            counts[document_id] = counts.get(document_id, 0) + 1
            if len(selected) >= self.top_k:
                return selected

        selected_ids = {context.doc_id for context in selected}
        for context in contexts:
            if context.doc_id in selected_ids:
                continue
            selected.append(context)
            if len(selected) >= self.top_k:
                break
        return selected

    def build_context_details(self, contexts: list[SearchResult]) -> list[dict[str, Any]]:
        """Monta metadados auditaveis mantendo a ordem de recuperacao."""
        details = []
        for rank, context in enumerate(contexts, start=1):
            details.append({
                "rank": rank,
                "score": context.score,
                "doc_id": context.doc_id,
                "filename": context.metadata.get("filename", "unknown"),
                "document_id": context.metadata.get("document_id"),
                "document_level": context.metadata.get("document_level"),
                "chunk_id": context.metadata.get("chunk_id"),
                "chunk_index": context.metadata.get("chunk_index"),
                "token_count": context.metadata.get("token_count"),
                "text": context.text,
            })
        return details

    def make_result(
        self,
        query: str,
        answer: str,
        contexts: list[SearchResult],
        retrieval_ms: float,
        generation_ms: float,
        total_ms: float,
        tokens_generated: int,
        tokens_per_second: float,
        reranking_ms: float = 0.0,
        routing_ms: float = 0.0,
        embedding_ms: float | None = None,
        vector_search_ms: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> RAGResult:
        """Monta o resultado a todas as estratégias."""
        return RAGResult(
            query=query,
            answer=answer,
            context_details=self.build_context_details(contexts),
            model_name=self.llm.model_name,
            retrieval_strategy_name=self.name,
            embedding_model=self.embedder.model_key,
            embedding_ms=round(self._embedding_ms if embedding_ms is None else embedding_ms, 2),
            vector_search_ms=round(self._vector_search_ms if vector_search_ms is None else vector_search_ms, 2),
            retrieval_ms=round(retrieval_ms, 2),
            reranking_ms=round(reranking_ms, 2),
            routing_ms=round(routing_ms, 2),
            generation_ms=round(generation_ms, 2),
            total_ms=round(total_ms, 2),
            tokens_generated=tokens_generated,
            tokens_per_second=round(tokens_per_second, 2),
            extra=extra or {},
        )

    def run(self, query: str) -> RAGResult:
        """Executa o pipeline RAG completo: recuperar e gerar."""
        total_start = time.perf_counter()
        self.reset_retrieval_timing()

        contexts = self.select_contexts(self.retrieve(query))
        retrieval_ms = self._embedding_ms + self._vector_search_ms

        prompt = self.build_prompt(query, contexts)

        with Timer("generation") as t_gen:
            llm_response = self.llm.generate(prompt, system_prompt=RAG_SYSTEM_PROMPT)
        generation_ms = t_gen.elapsed_ms

        total_ms = (time.perf_counter() - total_start) * 1000

        return self.make_result(
            query=query,
            answer=llm_response.text,
            contexts=contexts,
            retrieval_ms=round(retrieval_ms, 2),
            generation_ms=round(generation_ms, 2),
            total_ms=round(total_ms, 2),
            tokens_generated=llm_response.tokens_generated,
            tokens_per_second=llm_response.tokens_per_second,
        )
