"""Extração textual de arquivos PDF."""
from __future__ import annotations

from pathlib import Path

from ..models import ExtractionResult
import fitz

def extract_pdf(path: Path) -> ExtractionResult:
    """Extrai texto de PDFs com PyMuPDF em ordem visual ordenada."""

    with path.open("rb") as file_obj:
        pdf_bytes = file_obj.read()
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        pages = [page.get_text("text", sort=True) for page in document]
        text = "\n\n".join(page_text.strip() for page_text in pages if page_text.strip())
        warnings: tuple[str, ...] = ()
        if not text.strip():
            warnings = ("PDF sem camada textual extraível por PyMuPDF.",)
        return ExtractionResult(text=text, warnings=warnings)
