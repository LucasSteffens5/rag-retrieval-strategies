"""Modelos de dados do processamento determinístico do corpus."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CorpusLevel = Literal["A", "B", "C"]


@dataclass(frozen=True)
class DocumentRule:
    """Regra declarativa que associa uma fonte bruta a uma saída do corpus. """

    rule_id: str
    exact_name: str
    output_name: str
    level: CorpusLevel


@dataclass(frozen=True)
class ExtractionResult:
    """Texto extraído de uma fonte documental."""

    text: str
    warnings: tuple[str, ...] = ()
