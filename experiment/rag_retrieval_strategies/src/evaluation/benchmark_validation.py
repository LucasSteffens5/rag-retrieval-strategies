"""Valida a estrutura do benchmark simplificado."""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationReport:
    """Resultado serializável da validação estrutural do benchmark."""
    valid: bool
    errors: list[str]
    warnings: list[str]
    query_count: int

    def to_dict(self) -> dict[str, Any]:
        """Retorna o relatório em formato compatível com JSON."""
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "query_count": self.query_count,
        }


def compute_corpus_hash(corpus_dir: str | os.PathLike[str]) -> str:
    """Calcula hash determinístico dos textos do corpus processado."""
    hasher = hashlib.sha256()
    corpus_path = Path(corpus_dir)
    for path in sorted(corpus_path.glob("nivel_*/*.txt")):
        hasher.update(path.relative_to(corpus_path).as_posix().encode("utf-8"))
        with open(path, "rb") as handle:
            hasher.update(handle.read())
    return hasher.hexdigest()


def validate_benchmark(benchmark: dict[str, Any]) -> ValidationReport:
    """Valida o esquema mínimo do benchmark humano revisado sem taxonomia."""
    errors: list[str] = []
    warnings: list[str] = []

    if not str(benchmark.get("version", "")).strip():
        errors.append("benchmark: missing version")

    queries = benchmark.get("queries")
    if not isinstance(queries, list) or not queries:
        return ValidationReport(
            valid=False,
            errors=[*errors, "benchmark: queries must be a non-empty list"],
            warnings=warnings,
            query_count=0,
        )

    seen_ids: set[str] = set()
    for index, query in enumerate(queries):
        prefix = f"queries[{index}]"
        query_id = str(query.get("query_id", "")).strip()
        expected_query_id = f"Q{index + 1:03d}"
        if not query_id:
            errors.append(f"{prefix}: missing query_id")
        elif query_id in seen_ids:
            errors.append(f"{prefix}: duplicate query_id {query_id}")
        elif not re.fullmatch(r"Q\d{3}", query_id):
            errors.append(f"{prefix}: query_id must match Q###, got {query_id}")
        elif query_id != expected_query_id:
            errors.append(f"{prefix}: query_id must be sequential {expected_query_id}, got {query_id}")
        seen_ids.add(query_id)

        if "category" in query:
            errors.append(f"{prefix}: category must be absent in question-only benchmark")

        for field in ("query", "expected_answer"):
            if not str(query.get(field, "")).strip():
                errors.append(f"{prefix}: missing {field}")

        source_documents = query.get("source_documents")
        if not isinstance(source_documents, list):
            errors.append(f"{prefix}: source_documents must be a list")
            continue

        for doc_index, source_doc in enumerate(source_documents):
            doc_prefix = f"{prefix}.source_documents[{doc_index}]"
            if not isinstance(source_doc, dict):
                errors.append(f"{doc_prefix}: source document must be an object")
                continue

            for field in ("document_id", "document_level"):
                if not str(source_doc.get(field, "")).strip():
                    errors.append(f"{doc_prefix}: missing {field}")

            relevant_sections = source_doc.get("relevant_sections")
            if not isinstance(relevant_sections, list) or not relevant_sections:
                errors.append(f"{doc_prefix}: relevant_sections must be a non-empty list")

    return ValidationReport(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        query_count=len(queries),
    )
