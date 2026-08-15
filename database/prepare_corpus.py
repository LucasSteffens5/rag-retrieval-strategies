"""Inicia a preparação do corpus UFMT RAG."""
from __future__ import annotations

from corpus_processing.config import DEFAULT_BASE_DIR, DEFAULT_OUTPUT_DIR
from corpus_processing.pipeline import CorpusPreparationPipeline


def main() -> None:
    summary = CorpusPreparationPipeline(
        base_dir=DEFAULT_BASE_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
    ).run()
    print(
        "Corpus processado: "
        f"{summary['total_sources']} fontes, "
        f"{summary['processed_documents']} documentos RAG."
    )


if __name__ == "__main__":
    main()
