"""Pipeline principal de preparação do corpus documental."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .cleaners import clean_text, count_words
from .config import ALLOWED_EXTENSIONS, DOCUMENT_RULES, LEVEL_DIRECTORIES
from .extractors.docx import extract_docx
from .extractors.pdf import extract_pdf
from .extractors.txt import extract_txt
from .models import DocumentRule, ExtractionResult


class CorpusPreparationPipeline:
    """Prepara o corpus documental para ingestão RAG."""

    def __init__(self, base_dir: str | Path, output_dir: str | Path) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dirs = {
            level: self.output_dir / directory for level, directory in LEVEL_DIRECTORIES.items()
        }
        self.extractors: dict[str, Callable[[Path], ExtractionResult]] = {
            ".pdf": extract_pdf,
            ".docx": extract_docx,
            ".txt": extract_txt,
        }

    def run(self) -> dict[str, int]:
        """Executa todas as etapas determinísticas de preparação."""
        self._validate_inputs()
        self._ensure_output_dirs()
        sources = self._discover_sources()
        records = self._process_sources(sources)
        self._validate_coverage(sources, records)
        return {"total_sources": len(sources), "processed_documents": len(records)}

    def _validate_inputs(self) -> None:
        """Valida a existência do diretório bruto antes da extração."""
        if not self.base_dir.is_dir():
            raise FileNotFoundError(f"Diretório bruto do corpus não encontrado: {self.base_dir}")

    def _ensure_output_dirs(self) -> None:
        """Cria os diretórios de saída por nível documental."""
        for directory in self.output_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

    def _discover_sources(self) -> list[Path]:
        """Descobre fontes com extensões aceitas em ordem determinística."""
        return [
            path
            for path in sorted(
                self.base_dir.rglob("*"),
                key=lambda item: str(item.relative_to(self.base_dir)).casefold(),
            )
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
        ]

    def _process_sources(self, sources: list[Path]) -> list[Path]:
        """Aplica regras documentais e rejeita fontes não catalogadas."""
        by_name = {source.name.casefold(): source for source in sources}
        matched: set[Path] = set()
        outputs: list[Path] = []
        for rule in DOCUMENT_RULES:
            source = by_name.get(rule.exact_name.casefold())
            if source is None:
                raise FileNotFoundError(f"Fonte obrigatória não encontrada para a regra {rule.rule_id}.")
            outputs.append(self._process_one(source, rule))
            matched.add(source)

        unmatched = [source for source in sources if source not in matched]
        if unmatched:
            names = ", ".join(self._relative(source) for source in unmatched)
            raise RuntimeError(f"Fontes documentais sem regra explícita de processamento: {names}")
        return outputs

    def _process_one(self, source: Path, rule: DocumentRule) -> Path:
        """Extrai, limpa e grava uma fonte conforme sua regra documental."""
        extraction = self._extract(source)
        output_path = self.output_dirs[rule.level] / rule.output_name
        final_text = clean_text(extraction.text).strip()
        if count_words(final_text) == 0:
            raise RuntimeError(f"Documento RAG sem texto substantivo: {self._relative(source)}")
        if extraction.warnings:
            warnings = "; ".join(extraction.warnings)
            raise RuntimeError(f"Aviso de extração em documento RAG {self._relative(source)}: {warnings}")
        output_path.write_text(final_text + "\n", encoding="utf-8", newline="\n")
        return output_path

    def _extract(self, source: Path) -> ExtractionResult:
        """Seleciona o extrator apropriado pela extensão da fonte."""
        return self.extractors[source.suffix.lower()](source)

    def _relative(self, source: Path) -> str:
        """Retorna caminho relativo estável para mensagens de auditoria."""
        return str(source.relative_to(self.base_dir)).replace("\\", "/")

    def _validate_coverage(self, sources: list[Path], records: list[Path]) -> None:
        """Confirma que toda fonte descoberta gerou exatamente uma saída."""
        if len(sources) != len(records):
            raise RuntimeError(f"Cobertura inconsistente: {len(sources)} fontes e {len(records)} registros.")
