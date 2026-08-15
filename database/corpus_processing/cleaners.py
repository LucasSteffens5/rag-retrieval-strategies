"""Limpeza textual do corpus documental."""
from __future__ import annotations

import re
import unicodedata

HEADER_LINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*MINIST[ÉE]RIO DA EDUCA[ÇC][ÃA]O\s*$", re.IGNORECASE),
    re.compile(r"^\s*UNIVERSIDADE FEDERAL DE MATO GROSSO\s*$", re.IGNORECASE),
    re.compile(r"^\s*PR[ÓO]-REITORIA DE GEST[ÃA]O DE PESSOAS\s*$", re.IGNORECASE),
    re.compile(r"^\s*SUPERVIS[ÃA]O DE CONCURSOS\s*$", re.IGNORECASE),
    re.compile(r"^\s*Di[áa]rio Oficial da Uni[ãa]o\s*$", re.IGNORECASE),
    re.compile(r"^\s*Imprensa Nacional\s*$", re.IGNORECASE),
    re.compile(r"^\s*Se[çc][ãa]o\s+\d+\s+ISSN\s+\d{4}-\d{4}\s*$", re.IGNORECASE),
)


def normalize_text(text: str) -> str:
    """Normaliza unicode e quebras de linha preservando parágrafos. """
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", normalized)
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n[ \t]+", "\n", normalized)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    return normalized.strip()


def remove_repeated_headers(text: str) -> str:
    """Remove cabeçalhos institucionais recorrentes quando aparecem isolados."""
    kept_lines: list[str] = []
    for line in text.split("\n"):
        if any(pattern.match(line) for pattern in HEADER_LINE_PATTERNS):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def clean_text(text: str) -> str:
    """Remove artefatos recorrentes e independentes do documento."""
    cleaned = remove_repeated_headers(normalize_text(text))
    cleaned = re.sub(r"×\nDiário Oficial da União\nImprensa Nacional\nBAIXAR.*?Ver\n", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"Documento assinado digitalmente.*?ICP-Brasil\.", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"Este documento pode ser verificado.*?pelo código \d+", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\d+/\d+/\d+, \d+:\d+\n.*?https://www\.planalto\.gov\.br.*?\n", "", cleaned)
    cleaned = re.sub(r"https?://\S+\n?", "", cleaned)
    cleaned = re.sub(r"ISSN \d+-\d+\n", "", cleaned)
    cleaned = re.sub(r"Nº \d+, \w+-\w+, \d+ de \w+ de \d+\n", "", cleaned)
    cleaned = re.sub(r"^\s*\d+\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def count_words(text: str) -> int:
    return len(text.split())
