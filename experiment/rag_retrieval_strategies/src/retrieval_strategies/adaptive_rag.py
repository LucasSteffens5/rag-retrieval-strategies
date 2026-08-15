"""Adaptive RAG com roteamento por consulta."""
import time
from typing import Any

from retrieval_strategies.base import RAGRetrievalStrategy, RAGResult
from retrieval_strategies.naive_rag import NaiveRAG
from retrieval_strategies.hybrid_rag import HybridRAG
from retrieval_strategies.hyde_rag import HyDERAG
from retrieval_strategies.reranking_rag import RerankingRAG
from utils.vector_store import SearchResult
from utils.timer import Timer
from prompts import ADAPTIVE_ROUTER_PROMPT


class AdaptiveRAG(RAGRetrievalStrategy):
    """Roteia consultas entre estrategias com o mesmo LLM avaliado."""
    name = "adaptive"
    routable_routes: tuple[str, ...] = ("naive", "hybrid", "reranking", "hyde")
    default_route = "naive"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Inicializa as subestratégias roteáveis com as mesmas dependências."""
        super().__init__(*args, **kwargs)
        self._naive = NaiveRAG(*args, **kwargs)
        self._hybrid = HybridRAG(*args, **kwargs)
        self._reranking = RerankingRAG(*args, **kwargs)
        self._hyde = HyDERAG(*args, **kwargs)

    def retrieve(self, query: str) -> list[SearchResult]:
        """Mantém uma rota densa padrão para compatibilidade com a interface base."""
        return self._naive.retrieve(query)

    def warmup(self) -> None:
        """Aquece subcomponentes fora das consultas medidas."""
        self._reranking.warmup()
        self._hyde.warmup()

    def unload(self) -> None:
        """Libera recursos da rota com reranking."""
        self._reranking.unload()

    def route_query(self, query: str) -> tuple[str, int]:
        """Seleciona a rota e retorna os tokens do roteador."""
        import json
        prompt = ADAPTIVE_ROUTER_PROMPT.format(query=query)
        schema = {
            "type": "object",
            "properties": {
                "route": {
                    "type": "string",
                    "enum": list(self.routable_routes)
                }
            },
            "required": ["route"]
        }
        resp = self.llm.generate(prompt, format=schema)
        try:
            data = json.loads(resp.text)
            route = str(data.get("route") or "").strip().lower()
            if route in self.routable_routes:
                return route, resp.tokens_generated
        except Exception:
            pass

        return self.default_route, resp.tokens_generated

    def run(self, query: str) -> RAGResult:
        """Roteia a consulta e executa a estratégia selecionada."""
        total_start = time.perf_counter()
        with Timer("routing") as t_route:
            selected_route, routing_tokens = self.route_query(query)
        route_map = {
            "naive": self._naive,
            "hybrid": self._hybrid,
            "reranking": self._reranking,
            "hyde": self._hyde,
        }
        strategy = route_map.get(selected_route, self._naive)
        result = strategy.run(query)
        total_ms = (time.perf_counter() - total_start) * 1000
        routing_ms = t_route.elapsed_ms
        routing_generation_ms = routing_ms
        route_tokens = result.tokens_generated
        route_generation_ms = result.generation_ms
        total_generation_ms = routing_generation_ms + route_generation_ms
        total_tokens = routing_tokens + route_tokens
        tokens_per_second = (
            total_tokens / (total_generation_ms / 1000)
            if total_generation_ms > 0
            else 0.0
        )

        result.retrieval_strategy_name = f"adaptive->{strategy.name}"
        result.total_ms = round(total_ms, 2)
        result.routing_ms = round(routing_ms, 2)
        result.generation_ms = round(total_generation_ms, 2)
        result.tokens_generated = total_tokens
        result.tokens_per_second = round(tokens_per_second, 2)
        result.extra["selected_route"] = selected_route
        result.extra["routed_to"] = strategy.name
        result.extra["routing_tokens"] = routing_tokens
        result.extra["route_tokens"] = route_tokens
        result.extra["route_generation_ms"] = round(route_generation_ms, 2)
        if strategy.name != "hyde":
            result.extra["answer_tokens"] = route_tokens
            result.extra["answer_generation_ms"] = round(route_generation_ms, 2)
        result.extra["routing_generation_ms"] = round(routing_generation_ms, 2)
        result.extra["total_generation_ms_for_throughput"] = round(total_generation_ms, 2)
        result.extra["generation_ms_definition"] = "routing_generation_ms + route_generation_ms"
        result.extra["throughput_definition"] = "tokens_generated / generation_ms"
        return result
