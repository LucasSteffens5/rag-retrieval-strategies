"""Métricas clássicas de recuperação para resultados brutos RAG."""
from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from config import RESULTS_DIR, RETRIEVAL_TOP_K
from evaluation.metrics.common import (
    base_report as common_base_report,
    context_details,
    mean_summary,
    prepare_metric_input,
    write_json,
)

RETRIEVAL_RESULTS_DIR = os.getenv(
    "RETRIEVAL_RESULTS_DIR",
    os.path.join(RESULTS_DIR, "metrics", "retrieval"),
)

RETRIEVAL_K = RETRIEVAL_TOP_K
RELEVANCE_UNIT = "document_id"
METRIC_NAMES = (
    f"recall_at_{RETRIEVAL_K}",
    f"ndcg_at_{RETRIEVAL_K}",
)


def evaluate_raw_file(
    raw_result_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] = RETRIEVAL_RESULTS_DIR,
) -> dict[str, Any]:
    """Avalia um JSON bruto e grava um arquivo auxiliar de métricas de recuperação."""
    metric_input = prepare_metric_input(raw_result_path, output_dir, "retrieval_metrics")
    results = evaluate_results(metric_input.raw.get("results", []))
    report = common_base_report(
        raw_path=metric_input.raw_path,
        raw_hash=metric_input.raw_hash,
        raw=metric_input.raw,
        metric_family="retrieval",
        parameters=retrieval_parameters(),
        status="completed",
        extra_keys=("config", "benchmark", "corpus"),
    )
    report.update(
        {
            "results": results,
            "summary": summarize_results(results),
        }
    )
    return write_json(metric_input.sidecar_path, report)


def evaluate_results(
    raw_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Calcula métricas de recuperação para cada item bruto."""
    evaluated: list[dict[str, Any]] = []
    for item in raw_results:
        expected_document_ids = extract_expected_document_ids(item)
        retrieved_document_ids = [] if "error" in item else extract_retrieved_document_ids(item)
        base = {
            "query_id": item.get("query_id"),
            "expected_document_ids": expected_document_ids,
            f"retrieved_document_ids_at_{RETRIEVAL_K}": retrieved_document_ids,
        }

        if "error" in item:
            evaluated.append({**base, "status": "skipped_generation_error"})
            continue

        if not expected_document_ids:
            evaluated.append({**base, "status": "skipped_no_reference_documents"})
            continue

        scores = compute_query_metrics(
            query_id=str(item.get("query_id", "")),
            expected_document_ids=expected_document_ids,
            retrieved_document_ids=retrieved_document_ids,
        )
        evaluated.append({**base, "status": "completed", **scores})
    return evaluated


def extract_expected_document_ids(item: Mapping[str, Any]) -> list[str]:
    """Extrai documentos relevantes anotados no benchmark."""
    source_documents = item.get("source_documents", [])
    if not isinstance(source_documents, list):
        return []
    values = [
        str(source_doc.get("document_id", "")).strip()
        for source_doc in source_documents
        if isinstance(source_doc, Mapping)
    ]
    return unique_preserving_order([value for value in values if value])


def extract_retrieved_document_ids(item: Mapping[str, Any]) -> list[str]:
    """Extrai ranking recuperado em nível de documento."""
    ordered_contexts = sorted(context_details(item), key=context_rank)
    document_ids = [
        str(context.get("document_id", "")).strip()
        for context in ordered_contexts
        if str(context.get("document_id", "")).strip()
    ]
    return unique_preserving_order(document_ids)[:RETRIEVAL_K]


def context_rank(context: Mapping[str, Any]) -> int:
    """Retorna a posição inteira com alternativa estável."""
    try:
        rank = int(context.get("rank", 0))
    except (TypeError, ValueError):
        return 10**9
    return rank if rank > 0 else 10**9


def unique_preserving_order(values: Sequence[str]) -> list[str]:
    """Remove duplicatas preservando a primeira ocorrência."""
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def compute_query_metrics(
    query_id: str,
    expected_document_ids: Sequence[str],
    retrieved_document_ids: Sequence[str],
) -> dict[str, float]:
    """Calcula cobertura e qualidade de ordenacao para uma pergunta."""
    import ir_measures
    from ir_measures import R, nDCG

    qrels = [
        ir_measures.Qrel(query_id, document_id, 1)
        for document_id in expected_document_ids
    ]
    run = [
        ir_measures.ScoredDoc(query_id, document_id, float(RETRIEVAL_K - index))
        for index, document_id in enumerate(retrieved_document_ids)
    ]
    measures = [R @ RETRIEVAL_K, nDCG @ RETRIEVAL_K]
    metric_values = {
        str(metric.measure): float(metric.value)
        for metric in ir_measures.iter_calc(measures, qrels, run)
    }
    return {
        f"recall_at_{RETRIEVAL_K}": round(metric_values.get(f"R@{RETRIEVAL_K}", 0.0), 6),
        f"ndcg_at_{RETRIEVAL_K}": round(metric_values.get(f"nDCG@{RETRIEVAL_K}", 0.0), 6),
    }


def summarize_results(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Agrega médias e contagens das métricas de recuperação."""
    total_queries = len(results)
    completed = [item for item in results if item.get("status") == "completed"]
    skipped_generation_error = sum(
        1 for item in results if item.get("status") == "skipped_generation_error"
    )
    skipped_no_reference_documents = sum(
        1 for item in results if item.get("status") == "skipped_no_reference_documents"
    )
    skipped_total = total_queries - len(completed)
    metrics: dict[str, dict[str, float | int | None]] = {}
    for metric_name in METRIC_NAMES:
        values = [
            float(item[metric_name])
            for item in completed
            if isinstance(item.get(metric_name), (int, float))
        ]
        metrics[metric_name] = mean_summary(values, skipped=skipped_total)
    return {
        "total_queries": total_queries,
        "evaluated_queries": len(completed),
        "skipped_queries": skipped_total,
        "skipped_generation_error": skipped_generation_error,
        "skipped_no_reference_documents": skipped_no_reference_documents,
        "metrics": metrics,
    }


def retrieval_parameters() -> dict[str, Any]:
    """Registra parâmetros fixos da avaliação de recuperação."""
    return {
        "k": RETRIEVAL_K,
        "relevance_unit": RELEVANCE_UNIT,
        "library": "ir-measures",
        "library_measures": {
            f"recall_at_{RETRIEVAL_K}": f"R@{RETRIEVAL_K}",
            f"ndcg_at_{RETRIEVAL_K}": f"nDCG@{RETRIEVAL_K}",
        },
    }
