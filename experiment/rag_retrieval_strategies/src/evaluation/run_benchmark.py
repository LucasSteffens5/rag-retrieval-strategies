"""Executa a matriz experimental do benchmark RAG."""
from __future__ import annotations

import json
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from config import (
    RETRIEVAL_STRATEGIES,
    BENCHMARK_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CORPUS_DIR,
    DEFAULT_RUNS,
    DENSE_CANDIDATES,
    EMBEDDING_MODELS,
    EXPERIMENT_ID,
    EXPERIMENT_VERSION,
    LLM_MODELS,
    LLM_NUM_CTX,
    LLM_NUM_GPU,
    LLM_SEED,
    LLM_THINKING_ENABLED,
    LLM_TEMPERATURE,
    LLM_TOP_K,
    LLM_TOP_P,
    MAX_CONTEXTS_PER_DOCUMENT,
    RANDOM_SEED,
    RERANKER_MAX_LENGTH,
    RESULTS_DIR,
    RETRIEVAL_TOP_K,
    get_collection_name,
)
from evaluation.benchmark_validation import (
    compute_corpus_hash,
    validate_benchmark,
)
from evaluation.metrics.common import sha256_file

QUERY_MAX_RETRIES = 2
QUERY_RETRY_SLEEP_SECONDS = 2.0


def get_hardware_info() -> dict[str, Any]:
    """Coleta metadados de GPU quando ``nvidia-smi`` esta disponivel."""
    info: dict[str, Any] = {"gpu": "unknown", "vram_total_mb": 0, "driver": "unknown"}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            info["gpu"] = parts[0]
            info["vram_total_mb"] = int(parts[1])
            info["driver"] = parts[2]
    except Exception:
        pass
    return info


def run_query_with_retries(strategy: Any, query_item: dict[str, Any]) -> dict[str, Any]:
    """Executa uma pergunta do benchmark com retentativas auditáveis."""
    attempt_errors: list[dict[str, Any]] = []
    total_attempts = QUERY_MAX_RETRIES + 1

    for attempt in range(1, total_attempts + 1):
        try:
            result = strategy.run(query_item["query"])
            retry_attempts = attempt - 1
            extra = dict(result.extra)
            extra.update(
                {
                    "retry_attempts": retry_attempts,
                    "max_retries": QUERY_MAX_RETRIES,
                    "attempt_errors": attempt_errors,
                }
            )
            return {
                "query_id": query_item["query_id"],
                "query": query_item["query"],
                "expected_answer": query_item["expected_answer"],
                "generated_answer": result.answer,
                "context_details": result.context_details,
                "source_documents": query_item.get("source_documents", []),
                "embedding_ms": result.embedding_ms,
                "vector_search_ms": result.vector_search_ms,
                "retrieval_ms": result.retrieval_ms,
                "reranking_ms": result.reranking_ms,
                "routing_ms": result.routing_ms,
                "generation_ms": result.generation_ms,
                "total_ms": result.total_ms,
                "tokens_generated": result.tokens_generated,
                "tokens_per_second": result.tokens_per_second,
                "extra": extra,
            }
        except Exception as exc:
            attempt_errors.append(
                {
                    "attempt": attempt,
                    "exception_type": exc.__class__.__name__,
                    "message": str(exc),
                }
            )
            if attempt < total_attempts:
                time.sleep(QUERY_RETRY_SLEEP_SECONDS)

    return {
        "query_id": query_item["query_id"],
        "query": query_item.get("query"),
        "expected_answer": query_item.get("expected_answer"),
        "source_documents": query_item.get("source_documents", []),
        "embedding_ms": None,
        "vector_search_ms": None,
        "retrieval_ms": None,
        "reranking_ms": None,
        "routing_ms": None,
        "generation_ms": None,
        "total_ms": None,
        "error": attempt_errors[-1]["message"] if attempt_errors else "unknown error",
        "retry_attempts": QUERY_MAX_RETRIES,
        "max_retries": QUERY_MAX_RETRIES,
        "attempt_errors": attempt_errors,
    }


def get_model_quantization(llm: Any, model_name: str) -> str:
    """Consulta o Ollama para obter a quantização real do modelo."""
    try:
        import requests
        show_resp = requests.post(
            f"{llm.host}/api/show",
            json={"name": model_name},
            timeout=10,
        )
        if show_resp.status_code == 200:
            details = show_resp.json().get("details", {})
            return details.get("quantization_level", "unknown")
    except Exception:
        pass
    return "unknown"


def build_run_output(
    strategy_name: str,
    model_name: str,
    emb_key: str,
    model_quantization: str,
    run_num: int,
    emb_warmup_ms: float,
    embedder_runtime: str,
    loaded_model_info: dict[str, Any],
    benchmark: dict[str, Any],
    benchmark_hash: str,
    validation_report: Any,
    corpus_hash: str,
    collection: str,
    run_info: dict[str, Any],
    hardware_info: dict[str, Any],
    run_results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Monta o dicionário estruturado para exportação do resultado da run."""
    config = {
        "retrieval_strategy": strategy_name,
        "llm_model": model_name,
        "embedding_model": emb_key,
        "quantization": model_quantization,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": RETRIEVAL_TOP_K,
        "dense_candidates": DENSE_CANDIDATES,
        "max_contexts_per_document": MAX_CONTEXTS_PER_DOCUMENT,
        "reranker_max_length": RERANKER_MAX_LENGTH,
        "temperature": LLM_TEMPERATURE,
        "seed": LLM_SEED,
        "num_ctx": LLM_NUM_CTX,
        "num_gpu": LLM_NUM_GPU,
        "llm_gpu_only": True,
        "llm_size_bytes": loaded_model_info.get("size"),
        "llm_size_vram_bytes": loaded_model_info.get("size_vram"),
        "llm_thinking_enabled": LLM_THINKING_ENABLED,
        "top_k_llm": LLM_TOP_K,
        "top_p": LLM_TOP_P,
        "benchmark_version": benchmark.get("version", "unknown"),
        "query_embedding_mode": "online",
        "embedding_runtime": embedder_runtime,
        "embedding_warmup_excluded": True,
        "embedding_warmup_ms": round(emb_warmup_ms, 2),
    }
    return {
        "experiment_id": EXPERIMENT_ID,
        "version": EXPERIMENT_VERSION,
        "run_id": f"run_{run_num:03d}",
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "benchmark": {
            "version": benchmark.get("version", "unknown"),
            "hash": benchmark_hash,
            "validation": validation_report.to_dict(),
        },
        "corpus": {
            "corpus_hash": corpus_hash,
        },
        "vector_store": {
            "collection": collection,
            "collection_points_count": run_info.get("points_count"),
            "collection_status": run_info.get("status"),
        },
        "hardware": hardware_info,
        "results": run_results,
        "summary": summary,
    }


def mean_field(rows: list[dict[str, Any]], field: str) -> float:
    """Calcula média de campo numérico em linhas bem-sucedidas."""
    return sum(float(row[field]) for row in rows) / len(rows)


def summarize_run(run_results: list[dict[str, Any]], total_queries: int) -> dict[str, Any]:
    """Resume uma execução sem alterar métricas primárias por consulta."""
    valid = [result for result in run_results if "error" not in result]
    summary: dict[str, Any] = {
        "total_queries": total_queries,
        "successful": len(valid),
        "errors": len(run_results) - len(valid),
    }
    if valid:
        summary.update({
            "avg_latency_ms": mean_field(valid, "total_ms"),
            "avg_embedding_ms": mean_field(valid, "embedding_ms"),
            "avg_vector_search_ms": mean_field(valid, "vector_search_ms"),
            "avg_tokens_per_second": mean_field(valid, "tokens_per_second"),
        })
    return summary


def run_benchmark() -> None:
    """Executa a matriz experimental"""
    from rich.console import Console

    from retrieval_strategies import RETRIEVAL_STRATEGY_REGISTRY
    from utils.embedder import EmbeddingManager
    from utils.llm_client import OllamaClient
    from utils.vector_store import VectorStore

    retrieval_strategies = RETRIEVAL_STRATEGIES
    models = LLM_MODELS
    embeddings = list(EMBEDDING_MODELS.keys())
    runs = DEFAULT_RUNS
    output_dir = Path(RESULTS_DIR) / "raw"

    console = Console()
    console.rule("[bold blue]Benchmark de Estratégias de Recuperação RAG[/]")

    benchmark_path = Path(BENCHMARK_PATH)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark_hash = sha256_file(benchmark_path)
    validation_report = validate_benchmark(benchmark)
    if not validation_report.valid:
        failures = "\n".join(validation_report.errors[:25])
        raise RuntimeError(f"Benchmark inválido para execução científica:\n{failures}")

    corpus_hash = compute_corpus_hash(CORPUS_DIR)
    queries = benchmark["queries"]
    hardware_info = get_hardware_info()
    output_dir.mkdir(parents=True, exist_ok=True)
    store = VectorStore()

    total_configs = len(retrieval_strategies) * len(models) * len(embeddings) * runs
    total_queries = total_configs * len(queries)

    console.print(f"Benchmark: [cyan]{benchmark.get('version', 'unknown')}[/]")
    console.print("Validação do benchmark: [green]OK[/]")
    console.print(f"Queries no benchmark: [cyan]{len(queries)}[/]")
    console.print(f"Configurações: [cyan]{total_configs}[/]")
    console.print(f"Total execuções: [cyan]{total_queries}[/]")

    config_idx = 0
    for emb_key in embeddings:
        collection = get_collection_name(emb_key)
        try:
            info = store.get_collection_info(collection)
        except Exception as exc:
            raise RuntimeError(f"Collection {collection} não encontrada. Execute ingest.py primeiro.") from exc
        console.print(f"\n[green]Collection {collection}: {info['points_count']} pontos[/]")

        embedder = EmbeddingManager(emb_key, device="cpu")
        console.print("[dim]Warmup do embedding BGE-M3 em CPU...[/]")
        emb_warmup_start = time.perf_counter()
        embedder.warmup()
        emb_warmup_ms = (time.perf_counter() - emb_warmup_start) * 1000
        console.print(f"[green]Embedding CPU aquecido em {emb_warmup_ms:.0f}ms[/]")

        for model_name in models:
            llm = OllamaClient(model_name)
            if not llm.health_check():
                raise RuntimeError(f"Modelo {model_name} não disponível no Ollama.")

            OllamaClient.unload_loaded_models()

            console.print(f"  [dim]Warmup: carregando {model_name} na VRAM...[/]")
            try:
                llm.generate("Olá, responda apenas: ok")
                loaded_model_info = llm.assert_gpu_only()
                console.print("  [dim]Modelo aquecido[/]")
                console.print(
                    "  [dim]GPU-only validado: "
                    f"size_vram={loaded_model_info.get('size_vram')} "
                    f"size={loaded_model_info.get('size')}[/]"
                )
            except Exception:
                llm.unload()
                raise

            model_quantization = get_model_quantization(llm, model_name)
            if model_quantization != "unknown":
                console.print(f"  [dim]Quantização real: {model_quantization}[/]")

            for strategy_name in retrieval_strategies:
                strategy_class = RETRIEVAL_STRATEGY_REGISTRY[strategy_name]
                strategy = strategy_class(
                    llm=llm,
                    store=store,
                    embedder=embedder,
                    collection=collection,
                )
                if hasattr(strategy, "warmup"):
                    console.print(f"  [dim]Warmup da estratégia {strategy_name}...[/]")
                    strategy.warmup()
                    console.print("  [dim]Estratégia aquecida[/]")

                for run_num in range(1, runs + 1):
                    config_idx += 1
                    console.print(
                        f"\n[bold]Config [{config_idx}/{total_configs}] "
                        f"{strategy_name} + {model_name} + {emb_key} | run {run_num}/{runs}[/]"
                    )

                    shuffled_queries = list(queries)
                    random.seed(RANDOM_SEED + run_num)
                    random.shuffle(shuffled_queries)

                    run_results = []
                    for query_item in shuffled_queries:
                        query_result = run_query_with_retries(strategy, query_item)
                        run_results.append(query_result)

                    summary = summarize_run(run_results, len(queries))

                    run_info = store.get_collection_info(collection)
                    output = build_run_output(
                        strategy_name=strategy_name,
                        model_name=model_name,
                        emb_key=emb_key,
                        model_quantization=model_quantization,
                        run_num=run_num,
                        emb_warmup_ms=emb_warmup_ms,
                        embedder_runtime=embedder.runtime,
                        loaded_model_info=loaded_model_info,
                        benchmark=benchmark,
                        benchmark_hash=benchmark_hash,
                        validation_report=validation_report,
                        corpus_hash=corpus_hash,
                        collection=collection,
                        run_info=run_info,
                        hardware_info=hardware_info,
                        run_results=run_results,
                        summary=summary,
                    )

                    safe_model_name = model_name.replace("/", "-").replace("\\", "-").replace(":", "-")
                    filename = f"{strategy_name}_{safe_model_name}_{emb_key}_run{run_num:03d}.json"
                    output_path = output_dir / filename
                    with output_path.open("w", encoding="utf-8") as handle:
                        json.dump(output, handle, ensure_ascii=False, indent=2)

                    console.print(f"  [green]Saved: {filename}[/]")
                    if summary["successful"]:
                        console.print(
                            f"  Summary: success={summary['successful']}/{summary['total_queries']} "
                            f"Lat={summary['avg_latency_ms']:.0f}ms | Avg Speed={summary['avg_tokens_per_second']:.1f} tok/s"
                        )

                if hasattr(strategy, "unload"):
                    strategy.unload()

            llm.unload()

    console.print(f"\n[bold green]Benchmark concluído: {config_idx} configurações executadas.[/]")


if __name__ == "__main__":
    run_benchmark()
