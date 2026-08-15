"""Estatísticas de latência para resultados brutos RAG."""
from __future__ import annotations

import math
import os
from statistics import stdev
from typing import Any, Sequence

from config import RESULTS_DIR
from evaluation.metrics.common import (
    write_json,
    prepare_metric_input,
    base_report as common_base_report,
)

LATENCY_RESULTS_DIR = os.getenv(
    "LATENCY_RESULTS_DIR",
    os.path.join(RESULTS_DIR, "metrics", "latency"),
)

LATENCY_FIELDS: tuple[str, ...] = (
    "total_ms",
    "embedding_ms",
    "vector_search_ms",
    "retrieval_ms",
    "reranking_ms",
    "routing_ms",
    "generation_ms",
)

OPTIONAL_FIELDS: frozenset[str] = frozenset({"reranking_ms", "routing_ms"})

LATENCY_PARAMETERS = {
    "latency_fields": list(LATENCY_FIELDS),
    "optional_fields": sorted(OPTIONAL_FIELDS),
    "percentile_method": "linear_interpolation",
    "std_method": "sample_ddof_1",
    "filter": "successful_executions_only",
}


def evaluate_raw_file(
    raw_result_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] = LATENCY_RESULTS_DIR,
) -> dict[str, Any]:
    """Avalia um JSON bruto e grava um arquivo auxiliar de métricas de latência."""
    metric_input = prepare_metric_input(raw_result_path, output_dir, "latency_metrics")
    results, warnings = build_latency_rows(metric_input.raw.get("results", []))
    report = common_base_report(
        raw_path=metric_input.raw_path,
        raw_hash=metric_input.raw_hash,
        raw=metric_input.raw,
        metric_family="latency",
        parameters=LATENCY_PARAMETERS,
        status="completed",
        extra_keys=("config", "benchmark", "corpus", "hardware"),
    )
    report.update({"results": results, "summary": summarize(results), "warnings": warnings})
    return write_json(metric_input.sidecar_path, report)


def build_latency_rows(
    raw_results: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Monta linhas de latência com status de sucesso ou erro."""
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for item in raw_results:
        query_id = item.get("query_id")
        base: dict[str, Any] = {
            "query_id": query_id,
        }

        if "error" in item:
            base.update({"status": "error", "error": item["error"]})
        else:
            base["status"] = "success"
            for f in LATENCY_FIELDS:
                val = item.get(f)
                if val is None and f not in OPTIONAL_FIELDS:
                    warnings.append(
                        f"query_id={query_id}: campo obrigatório '{f}' ausente "
                        f"em execução bem-sucedida."
                    )
                base[f] = val if val is not None else 0.0
            base["tokens_generated"] = item.get("tokens_generated")
            base["tokens_per_second"] = item.get("tokens_per_second")
        rows.append(base)

    return rows, warnings


def compute_latency_stats(values: Sequence[float]) -> dict[str, float | None]:
    """Calcula média, desvio-padrão amostral e percentis para uma série."""
    if not values:
        return {k: None for k in ("mean", "std", "p50", "p95", "p99")}

    n = len(values)
    sorted_vals = sorted(values)

    def get_p(q: float) -> float:
        pos = (q / 100.0) * (n - 1)
        idx = int(pos)
        return sorted_vals[idx] if n == 1 else sorted_vals[idx] + (pos - idx) * (sorted_vals[idx + 1] - sorted_vals[idx])

    return {
        "mean": round(math.fsum(values) / n, 4),
        "std": round(stdev(values) if n > 1 else 0.0, 4),
        "p50": round(get_p(50.0), 4),
        "p95": round(get_p(95.0), 4),
        "p99": round(get_p(99.0), 4),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Agrega contagens e estatísticas de latência sobre execuções bem-sucedidas."""
    total = len(results)
    successful = [r for r in results if r.get("status") == "success"]
    errors = total - len(successful)

    def rate(count: int) -> float:
        return round(count / total, 6) if total > 0 else 0.0

    return {
        "total_queries": total,
        "successful": len(successful),
        "errors": errors,
        "success_rate": rate(len(successful)),
        "error_rate": rate(errors),
        "latency": {
            f: compute_latency_stats([float(r[f]) for r in successful if r.get(f) is not None])
            for f in LATENCY_FIELDS
        },
    }
