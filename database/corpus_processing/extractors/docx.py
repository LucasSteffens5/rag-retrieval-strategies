from __future__ import annotations

from pathlib import Path
from typing import Iterator

from docx import Document
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph

from ..models import ExtractionResult


def extract_docx(path: Path) -> ExtractionResult:
    """Extrai texto do corpo principal de DOCX.    """
    document = Document(path)
    parts: list[str] = []
    for block in _iter_body_blocks(document):
        if isinstance(block, Paragraph):
            paragraph = _paragraph_text(block)
            if paragraph:
                parts.append(paragraph)
        elif isinstance(block, Table):
            table = _table_text(block)
            if table:
                parts.append(table)
    text = "\n\n".join(part for part in parts if part.strip())
    warnings: tuple[str, ...] = ()
    if not text.strip():
        warnings = ("DOCX sem texto extraível no XML principal.",)
    return ExtractionResult(text=text, warnings=warnings)


def _iter_body_blocks(document: DocxDocument) -> Iterator[Paragraph | Table]:
    """Itera blocos do corpo principal preservando ordem de documento."""
    body = document.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def _table_text(table: Table) -> str:
    """Serializa células de tabela em linhas textuais estáveis."""
    rows: list[str] = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            text = _cell_text(cell)
            if text:
                cells.append(text)
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _cell_text(cell: _Cell) -> str:
    """Consolida parágrafos textuais de uma célula de tabela."""
    paragraphs = [_paragraph_text(paragraph) for paragraph in cell.paragraphs]
    return " ".join(paragraph for paragraph in paragraphs if paragraph)


def _paragraph_text(paragraph: Paragraph) -> str:
    """Retorna o texto normalizado de um parágrafo."""
    return paragraph.text.strip()
