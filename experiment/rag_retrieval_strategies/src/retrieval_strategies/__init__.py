from retrieval_strategies.adaptive_rag import AdaptiveRAG
from retrieval_strategies.hybrid_rag import HybridRAG
from retrieval_strategies.hyde_rag import HyDERAG
from retrieval_strategies.naive_rag import NaiveRAG
from retrieval_strategies.reranking_rag import RerankingRAG

RETRIEVAL_STRATEGY_REGISTRY = {
    "naive": NaiveRAG,
    "hybrid": HybridRAG,
    "reranking": RerankingRAG,
    "hyde": HyDERAG,
    "adaptive": AdaptiveRAG,
}
