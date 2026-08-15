"""HyDE RAG com documentos hipoteticos."""
import time

from retrieval_strategies.base import RAGRetrievalStrategy, RAGResult
from utils.vector_store import SearchResult
from utils.timer import Timer
from prompts import HYDE_GENERATION_PROMPT, RAG_SYSTEM_PROMPT
from config import DENSE_CANDIDATES


class HyDERAG(RAGRetrievalStrategy):
    name = "hyde"

    def warmup(self) -> None:
        """Aquece o embedder em CPU antes das consultas medidas."""
        self.embedder.embed_query("documento hipotetico de aquecimento")

    def retrieve(self, query: str) -> list[SearchResult]:
        """Recupera candidatos densos a partir do embedding da consulta original."""
        qe = self.get_query_embedding(query)
        return self.search_dense(qe.dense, top_k=DENSE_CANDIDATES)

    def run(self, query: str) -> RAGResult:
        """Executa HyDE com geração hipotética, recuperação densa e resposta final."""
        total_start = time.perf_counter()
        self.reset_retrieval_timing()
        with Timer("hyde") as t_hyde:
            hyde_resp = self.llm.generate(HYDE_GENERATION_PROMPT.format(query=query))
        with Timer("emb") as t_emb:
            hyde_emb = self.embedder.embed_query(hyde_resp.text)
        self._embedding_ms += t_emb.elapsed_ms
        contexts = self.select_contexts(self.search_dense(hyde_emb.dense, top_k=DENSE_CANDIDATES))
        prompt = self.build_prompt(query, contexts)
        with Timer("gen") as t_gen:
            resp = self.llm.generate(prompt, system_prompt=RAG_SYSTEM_PROMPT)
        total_ms = (time.perf_counter() - total_start) * 1000

        total_tokens = hyde_resp.tokens_generated + resp.tokens_generated
        total_gen_ms = t_hyde.elapsed_ms + t_gen.elapsed_ms
        effective_tps = (total_tokens / (total_gen_ms / 1000)) if total_gen_ms > 0 else 0

        return self.make_result(
            query=query,
            answer=resp.text,
            contexts=contexts,
            retrieval_ms=self._embedding_ms + self._vector_search_ms,
            generation_ms=total_gen_ms,
            total_ms=total_ms,
            tokens_generated=total_tokens,
            tokens_per_second=effective_tps,
            extra={"hyde_ms": round(t_hyde.elapsed_ms, 2),
                   "hyde_tokens": hyde_resp.tokens_generated,
                   "answer_tokens": resp.tokens_generated,
                   "hyde_emb_ms": round(t_emb.elapsed_ms, 2),
                   "hyp_doc": hyde_resp.text[:500]},
        )
