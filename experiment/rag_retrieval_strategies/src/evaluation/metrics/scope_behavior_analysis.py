"""Analise complementar de comportamento de escopo para resultados RAG."""
from __future__ import annotations

import os
import unicodedata
from typing import Any

from config import RESULTS_DIR
from evaluation.metrics.common import (
    base_report as common_base_report,
    prepare_metric_input,
    write_json,
)

SCOPE_BEHAVIOR_RESULTS_DIR = os.getenv(
    "SCOPE_BEHAVIOR_RESULTS_DIR",
    os.path.join(RESULTS_DIR, "analysis", "scope_behavior"),
)

REFUSAL_PATTERNS: list[str] = [
    "Informação não encontrada no contexto fornecido.",
]


def normalize_text(text: str) -> str:
    """Normaliza texto para comparar a frase de recusa."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFD", text.lower())
    clean_bytes = normalized.encode("ascii", "ignore")
    return " ".join(clean_bytes.decode("utf-8").split())


def classify_scope_behavior(is_out_of_scope: bool, refused: bool) -> str:
    """Classifica descritivamente o comportamento por escopo."""
    if is_out_of_scope and refused:
        return "correct_refusal_without_evidence"
    if is_out_of_scope and not refused:
        return "improper_answer_without_evidence"
    if not is_out_of_scope and refused:
        return "refusal_with_reference_documents"
    return "answered_with_reference_documents"


def evaluate_raw_file(
    raw_result_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] = SCOPE_BEHAVIOR_RESULTS_DIR,
) -> dict[str, Any]:
    """Gera contagens descritivas do comportamento por escopo."""
    metric_input = prepare_metric_input(raw_result_path, output_dir, "scope_behavior")

    results: list[dict[str, Any]] = []
    expected_refusal = normalize_text(REFUSAL_PATTERNS[0]).strip()

    for item in metric_input.raw.get("results", []):
        query_id = item.get("query_id")

        if "error" in item:
            results.append(
                {
                    "query_id": query_id,
                    "status": "skipped_generation_error",
                }
            )
            continue

        is_answerable = len(item.get("source_documents", [])) > 0
        is_out_of_scope = not is_answerable
        generated_answer = normalize_text(item.get("generated_answer", "")).strip()
        refused = generated_answer == expected_refusal
        behavior = classify_scope_behavior(is_out_of_scope, refused)

        results.append(
            {
                "query_id": query_id,
                "status": "success",
                "has_reference_documents": is_answerable,
                "is_out_of_scope": is_out_of_scope,
                "refused": refused,
                "behavior": behavior,
            }
        )

    valid = [row for row in results if row["status"] == "success"]
    out_of_scope = [row for row in valid if row["is_out_of_scope"]]
    answerable = [row for row in valid if not row["is_out_of_scope"]]

    behavior_counts = {
        "correct_refusal_without_evidence": sum(
            1 for row in out_of_scope if row["refused"]
        ),
        "improper_answer_without_evidence": sum(
            1 for row in out_of_scope if not row["refused"]
        ),
        "refusal_with_reference_documents": sum(
            1 for row in answerable if row["refused"]
        ),
        "answered_with_reference_documents": sum(
            1 for row in answerable if not row["refused"]
        ),
    }

    parameters = {
        "refusal_patterns": sorted(REFUSAL_PATTERNS),
        "out_of_scope_rule": "source_documents == []",
        "interpretation": "descriptive_counts_only",
    }
    report = common_base_report(
        raw_path=metric_input.raw_path,
        raw_hash=metric_input.raw_hash,
        raw=metric_input.raw,
        metric_family="scope_behavior",
        parameters=parameters,
        status="completed",
        extra_keys=("config", "benchmark", "corpus", "hardware"),
    )
    report["results"] = results
    report["summary"] = {
        "total_queries": len(results),
        "successful": len(valid),
        "skipped_generation_error": len(results) - len(valid),
        "out_of_scope_count": len(out_of_scope),
        "answerable_count": len(answerable),
        **behavior_counts,
    }

    return write_json(metric_input.sidecar_path, report)
