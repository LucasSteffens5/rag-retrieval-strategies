"""Centraliza as configurações experimentais."""
import os
from pathlib import Path

EMBEDDING_MODELS = {
    "bge-m3": {
        "model_name": "BAAI/bge-m3",
        "dimensions": 1024,
        "max_tokens": 8192,
        "supports_sparse": True,
    },
}

# Mantém o chunking fixo para isolar a estratégia de recuperação.
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " "]

LLM_MODELS = [
    "qwen3:1.7b-q4_K_M",
    "qwen3:4b-instruct-2507-q4_K_M",
    "qwen3:8b-q4_K_M",
]
LLM_TEMPERATURE = 0.0
LLM_SEED = 42
LLM_NUM_CTX = 4096
LLM_NUM_GPU = -1  # Solicita uso máximo da GPU ao Ollama.
LLM_TOP_K = 1
LLM_TOP_P = 0.9
LLM_THINKING_ENABLED = False

RETRIEVAL_STRATEGIES = ["naive", "hybrid", "reranking", "hyde", "adaptive"]
RETRIEVAL_TOP_K = 5
DENSE_CANDIDATES = 20
RERANKING_CANDIDATES = 20
HYBRID_CANDIDATES = 20
MAX_CONTEXTS_PER_DOCUMENT = 2

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_MAX_LENGTH = 1024

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))


def get_collection_name(embedding_key: str) -> str:
    """Retorna o nome determinístico da collection por embedding."""
    return f"rag_{embedding_key.replace('-', '_')}"


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parents[1] if len(PROJECT_DIR.parents) > 1 else PROJECT_DIR


def docker_or_local_path(container_path: str, local_path: Path) -> str:
    """Resolve caminho no Docker e aplica alternativa local quando necessario."""
    if os.path.exists(container_path):
        return container_path
    return str(local_path)


CORPUS_DIR = os.getenv(
    "CORPUS_DIR",
    docker_or_local_path("/app/corpus", WORKSPACE_DIR / "database" / "corpus_processado"),
)
BENCHMARK_PATH = os.getenv(
    "BENCHMARK_PATH",
    docker_or_local_path(
        "/app/benchmark/perguntas_benchmark_edital_docente_ufmt.json",
        PROJECT_DIR / "data" / "benchmark" / "perguntas_benchmark_edital_docente_ufmt.json",
    ),
)
RESULTS_DIR = os.getenv(
    "RESULTS_DIR",
    docker_or_local_path("/app/results", PROJECT_DIR / "results"),
)

CORPUS_LEVELS = {
    "A": "nivel_A_principal",
    "B": "nivel_B_suporte",
    "C": "nivel_C_periferico",
}

EXPERIMENT_ID = "rag_retrieval_strategies"
EXPERIMENT_VERSION = "1.0"
RANDOM_SEED = 42

DEFAULT_RUNS = 3  # Estima a variância da latência.
