from __future__ import annotations

import json
import os
from pathlib import Path

from .models import DocumentRule

PACKAGE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = PACKAGE_DIR.parent
DEFAULT_BASE_DIR = Path(os.getenv("CORPUS_BASE_DIR", DATABASE_DIR / "concurso docente"))
DEFAULT_OUTPUT_DIR = Path(os.getenv("CORPUS_DIR", DATABASE_DIR / "corpus_processado"))

ALLOWED_EXTENSIONS = (".pdf", ".docx", ".txt")

LEVEL_DIRECTORIES = {
    "A": "nivel_A_principal",
    "B": "nivel_B_suporte",
    "C": "nivel_C_periferico",
}

try:
    with open(PACKAGE_DIR / "document_rules.json", "r", encoding="utf-8") as f:
        _rules_data = json.load(f)
except FileNotFoundError as exc:
    raise FileNotFoundError(
        f"Arquivo de regras obrigatório não encontrado: {PACKAGE_DIR / 'document_rules.json'}. "
        "Verifique se o arquivo foi excluído ou movido acidentalmente do repositório."
    ) from exc
except json.JSONDecodeError as exc:
    raise ValueError(
        f"Arquivo de regras {PACKAGE_DIR / 'document_rules.json'} está corrompido ou não é um JSON válido."
    ) from exc

DOCUMENT_RULES: tuple[DocumentRule, ...] = tuple(
    DocumentRule(
        rule_id=r["rule_id"],
        exact_name=r["exact_name"],
        output_name=r["output_name"],
        level=r["level"],
    )
    for r in _rules_data
)
