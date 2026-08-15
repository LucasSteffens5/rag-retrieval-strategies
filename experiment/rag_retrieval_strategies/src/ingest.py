"""Executa a ingestão do corpus no Qdrant."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from functools import lru_cache
from typing import Any

from config import (
    CORPUS_DIR, CORPUS_LEVELS, EMBEDDING_MODELS,
    CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS,
    get_collection_name,
)
from utils.vector_store import VectorStore
from utils.embedder import EmbeddingManager
from utils.timer import Timer
from utils.corpus_indexing import (
    corpus_quality_summary,
    token_budgeted_chunks,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.table import Table

console = Console()


def load_corpus(corpus_dir: str | Path) -> list[dict[str, Any]]:
    """Carrega todos os arquivos .txt dos subdiretórios do corpus com metadados."""
    documents: list[dict[str, Any]] = []
    corpus_path = Path(corpus_dir)
    for level, subdir in CORPUS_LEVELS.items():
        level_dir = corpus_path / subdir
        if not level_dir.is_dir():
            console.print(f"[yellow]Warning: {level_dir} not found, skipping[/]")
            continue

        for path in sorted(level_dir.iterdir(), key=lambda item: item.name.casefold()):
            if not path.is_file() or path.suffix != ".txt":
                continue
            text = path.read_text(encoding="utf-8")
            document_id = path.stem

            documents.append({
                "text": text,
                "metadata": {
                    "document_id": document_id,
                    "filename": path.name,
                    "document_level": level,
                },
            })

    return documents


def chunk_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aplica a divisão recursiva token-aware em todos os documentos."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        length_function=count_tokens,
    )

    chunks: list[dict[str, Any]] = []
    for doc in documents:
        document_id = doc["metadata"]["document_id"]
        doc_chunks: list[dict[str, Any]] = []
        for chunk_text in token_budgeted_chunks(doc["text"], splitter, count_tokens):
            token_count = count_tokens(chunk_text)
            chunk_metadata = {
                **doc["metadata"],
                "token_count": token_count,
            }
            doc_chunks.append({"text": chunk_text, "metadata": chunk_metadata})

        for i, chunk in enumerate(doc_chunks):
            chunk_id = f"{doc['metadata']['document_level']}_{document_id}_chunk_{i:04d}"
            chunk["metadata"]["chunk_id"] = chunk_id
            chunk["metadata"]["chunk_index"] = i
            chunk["metadata"]["total_chunks"] = len(doc_chunks)
            chunks.append({
                "text": chunk["text"],
                "metadata": chunk["metadata"],
            })

    return chunks


def save_chunk_registry(chunks: list[dict[str, Any]], output_dir: str | Path) -> Path:
    """Salva os mapeamentos dos chunks e o resumo de qualidade."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    registry = [
        {
            "chunk_id": chunk["metadata"]["chunk_id"],
            "document_id": chunk["metadata"]["document_id"],
            "filename": chunk["metadata"]["filename"],
            "document_level": chunk["metadata"]["document_level"],
            "chunk_index": chunk["metadata"]["chunk_index"],
            "total_chunks": chunk["metadata"]["total_chunks"],
            "token_count": chunk["metadata"]["token_count"],
            "sha256": hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest(),
        }
        for chunk in chunks
    ]
    registry_path = output_path / "chunk_registry.json"
    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    quality_path = output_path / "chunk_quality_summary.json"
    with quality_path.open("w", encoding="utf-8") as f:
        json.dump(corpus_quality_summary(chunks), f, ensure_ascii=False, indent=2)
    return registry_path


@lru_cache(maxsize=1)
def get_chunk_tokenizer() -> Any:
    """Carrega o tokenizador do BGE-M3."""
    from transformers import AutoTokenizer
    model_name = EMBEDDING_MODELS["bge-m3"]["model_name"]
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def count_tokens(text: str) -> int:
    """Conta tokens usando o tokenizador do BGE-M3."""
    tokenizer = get_chunk_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def print_distribution_table(documents: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> None:
    """Imprime a tabela de distribuição de chunks por nível."""
    table = Table(title="Distribuição de chunks por nível")
    table.add_column("Nível", style="cyan")
    table.add_column("Documentos", justify="right")
    table.add_column("Chunks", justify="right")
    for level in ["A", "B", "C"]:
        n_docs = sum(d["metadata"]["document_level"] == level for d in documents)
        n_chunks = sum(c["metadata"]["document_level"] == level for c in chunks)
        table.add_row(level, str(n_docs), str(n_chunks))
    table.add_row("[bold]Total[/]", f"[bold]{len(documents)}[/]", f"[bold]{len(chunks)}[/]")
    console.print(table)


def ingest() -> None:
    """Executa a pipeline de ingestão do corpus no Qdrant."""
    console.rule("[bold blue]Pipeline de Ingestão — Estratégias de Recuperação RAG[/]")

    console.print("\n[bold]1. Carregando corpus...[/]")
    with Timer("load_corpus") as t:
        documents = load_corpus(CORPUS_DIR)
    console.print(f"   {len(documents)} documentos carregados em {t.elapsed_ms:.0f}ms")

    console.print("\n[bold]2. Aplicando chunking...[/]")
    with Timer("chunking") as t:
        chunks = chunk_documents(documents)
    console.print(f"   {len(chunks)} chunks gerados em {t.elapsed_ms:.0f}ms")
    registry_path = save_chunk_registry(chunks, Path(CORPUS_DIR) / "metadata")
    console.print(f"   Registro de chunks salvo em: {registry_path}")

    print_distribution_table(documents, chunks)

    texts = [c["text"] for c in chunks]
    metadata_list = [c["metadata"] for c in chunks]
    store = VectorStore()

    for emb_key in EMBEDDING_MODELS.keys():
        emb_config = EMBEDDING_MODELS[emb_key]
        collection_name = get_collection_name(emb_key)

        console.print(f"\n[bold]3. Embedding com {emb_key}...[/]")
        console.print(f"   Model: {emb_config['model_name']} | Sparse: {emb_config['supports_sparse']}")

        embedder = EmbeddingManager(emb_key, device="cuda")
        with Timer("embedding") as t:
            emb_results = embedder.embed_documents(texts)
        console.print(f"   {len(emb_results)} embeddings gerados em {t.elapsed_ms:.0f}ms")

        console.print(f"   Indexando no Qdrant ({collection_name})...")
        dense_vectors = [r.dense for r in emb_results]
        sparse_vectors = [r.sparse for r in emb_results] if emb_config["supports_sparse"] else None

        store.create_collection(
            name=collection_name,
            dense_dim=emb_config["dimensions"],
            enable_sparse=emb_config["supports_sparse"],
        )

        with Timer("indexing") as t:
            n_indexed = store.upsert_documents(
                collection=collection_name,
                texts=texts,
                dense_vectors=dense_vectors,
                sparse_vectors=sparse_vectors,
                metadata_list=metadata_list,
            )
        console.print(f"   {n_indexed} pontos indexados em {t.elapsed_ms:.0f}ms")

        info = store.get_collection_info(collection_name)
        console.print(f"   Collection status: {info['status']} | Points: {info['points_count']}")
        if info["points_count"] != len(chunks):
            raise RuntimeError(
                f"Ingestão inconsistente: {len(chunks)} chunks gerados, "
                f"mas {info['points_count']} pontos no Qdrant."
            )

    console.print(f"\n[bold green]✓ Ingestão concluída![/]")


if __name__ == "__main__":
    ingest()
