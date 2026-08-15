"""Implementa RAG com reranqueamento por cross-encoder."""
import time
import gc
import inspect

from retrieval_strategies.base import RAGRetrievalStrategy, RAGResult
from utils.vector_store import SearchResult
from utils.timer import Timer
from config import RERANKING_CANDIDATES, RERANKER_MODEL, RERANKER_MAX_LENGTH
from prompts import RAG_SYSTEM_PROMPT


class RerankingRAG(RAGRetrievalStrategy):
    """RAG com cross-encoder para maior precisao."""

    name = "reranking"

    def __init__(self, *args, **kwargs):
        """Inicializa o reranqueador sob demanda."""
        super().__init__(*args, **kwargs)
        self._reranker = None

    def _load_reranker(self):
        """Carrega o reranker sob demanda."""
        if self._reranker is None:
            import torch
            from FlagEmbedding import FlagReranker

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA indisponivel para o reranker.")

            self._reranker = FlagReranker(
                RERANKER_MODEL,
                use_fp16=True,
                device="cuda:0",
            )

    def warmup(self) -> None:
        """Aquece o reranker antes das consultas medidas."""
        self._load_reranker()
        self._reranker.compute_score(
            [["consulta de aquecimento", "documento de aquecimento"]],
            normalize=True,
        )

    def unload(self) -> None:
        """Libera a VRAM ocupada pelo reranker."""
        if self._reranker is not None:
            del self._reranker
            self._reranker = None
            gc.collect()
            import torch
            torch.cuda.empty_cache()

    def retrieve(self, query: str) -> list[SearchResult]:
        """Recupera candidatos por busca densa com embedding."""
        query_embedding = self.get_query_embedding(query)
        return self.search_dense(query_embedding.dense, top_k=RERANKING_CANDIDATES)

    def rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        """Reordena candidatos com cross-encoder."""
        self._load_reranker()

        pairs = [[query, c.text] for c in candidates]
        kwargs = {"normalize": True}
        if "max_length" in inspect.signature(self._reranker.compute_score).parameters:
            kwargs["max_length"] = RERANKER_MAX_LENGTH
        scores = self._reranker.compute_score(pairs, **kwargs)

        if isinstance(scores, float):
            scores = [scores]

        for i, candidate in enumerate(candidates):
            candidate.score = scores[i]

        reranked = sorted(candidates, key=lambda x: x.score, reverse=True)
        return self.select_contexts(reranked)

    def run(self, query: str) -> RAGResult:
        """Executa o pipeline com latencia de reranking separada."""
        total_start = time.perf_counter()
        self.reset_retrieval_timing()

        candidates = self.retrieve(query)
        with Timer("reranking") as t_rerank:
            contexts = self.rerank(query, candidates)

        prompt = self.build_prompt(query, contexts)
        with Timer("generation") as t_gen:
            llm_response = self.llm.generate(prompt, system_prompt=RAG_SYSTEM_PROMPT)

        total_ms = (time.perf_counter() - total_start) * 1000

        return self.make_result(
            query=query,
            answer=llm_response.text,
            contexts=contexts,
            retrieval_ms=self._embedding_ms + self._vector_search_ms,
            reranking_ms=t_rerank.elapsed_ms,
            generation_ms=t_gen.elapsed_ms,
            total_ms=total_ms,
            tokens_generated=llm_response.tokens_generated,
            tokens_per_second=llm_response.tokens_per_second,
        )
