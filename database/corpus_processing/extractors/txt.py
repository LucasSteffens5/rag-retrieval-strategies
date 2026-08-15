from __future__ import annotations

from pathlib import Path

from ..models import ExtractionResult


def extract_txt(path: Path) -> ExtractionResult:
    text = path.read_text(encoding="utf-8-sig")
    warnings: tuple[str, ...] = ()
    if not text.strip():
        warnings = ("TXT curado sem conteúdo textual.",)
    return ExtractionResult(text=text, warnings=warnings)
