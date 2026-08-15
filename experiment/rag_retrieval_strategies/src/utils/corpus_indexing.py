"""Utilitarios deterministicos para chunking simples do corpus RAG."""
from __future__ import annotations

import hashlib
import statistics
from typing import Any, Callable


def corpus_quality_summary(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Agrega estatísticas dos fragmentos gerados."""
    token_counts = sorted(int(chunk["metadata"]["token_count"]) for chunk in chunks)
    duplicate_hashes: dict[str, int] = {}
    for chunk in chunks:
        digest = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
        duplicate_hashes[digest] = duplicate_hashes.get(digest, 0) + 1

    if not token_counts:
        return {
            "total_chunks": 0,
            "min_tokens": 0,
            "median_tokens": 0,
            "p75_tokens": 0,
            "max_tokens": 0,
            "chunks_lt_80_tokens": 0,
            "duplicate_chunk_hashes": 0,
            "duplicate_chunk_instances": 0,
            "summary_statistics_version": "descriptive_statistics_v2",
            "quantile_method": "inclusive",
        }

    median_tokens = statistics.median(token_counts)
    p75_tokens = (
        token_counts[0]
        if len(token_counts) == 1
        else statistics.quantiles(token_counts, n=4, method="inclusive")[2]
    )
    return {
        "total_chunks": len(chunks),
        "min_tokens": token_counts[0],
        "median_tokens": median_tokens,
        "p75_tokens": p75_tokens,
        "max_tokens": token_counts[-1],
        "chunks_lt_80_tokens": sum(1 for count in token_counts if count < 80),
        "duplicate_chunk_hashes": sum(1 for count in duplicate_hashes.values() if count > 1),
        "duplicate_chunk_instances": sum(count for count in duplicate_hashes.values() if count > 1),
        "summary_statistics_version": "descriptive_statistics_v2",
        "quantile_method": "inclusive",
    }


def token_budgeted_chunks(
    text: str,
    splitter: Any,
    count_tokens: Callable[[str], int],
) -> list[str]:
    """Divide o texto e remove fragmentos vazios."""
    chunks: list[str] = []
    for chunk in splitter.split_text(text):
        clean_chunk = chunk.strip()
        if clean_chunk and count_tokens(clean_chunk) > 0:
            chunks.append(clean_chunk)
    return chunks
