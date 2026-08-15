"""Adapta o Qdrant para vetores densos e esparsos."""
import uuid
import hashlib
from typing import Any
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
)

from config import QDRANT_HOST, QDRANT_PORT, HYBRID_CANDIDATES


@dataclass
class SearchResult:
    """Resultado de busca no Qdrant."""
    text: str
    score: float
    metadata: dict[str, Any]
    doc_id: str


class VectorStore:
    """Adaptador de operacoes no Qdrant."""

    def __init__(self, host: str = QDRANT_HOST, port: int = QDRANT_PORT):
        """Inicializa o cliente Qdrant com timeout fixo para o experimento."""
        self.client = QdrantClient(host=host, port=port, timeout=60)

    def create_collection(
        self,
        name: str,
        dense_dim: int,
        enable_sparse: bool = False,
    ):
        """Cria colecao com vetores densos e, opcionalmente, esparsos."""
        vectors_config = {
            "dense": VectorParams(size=dense_dim, distance=Distance.COSINE)
        }
        sparse_vectors_config = None
        if enable_sparse:
            sparse_vectors_config = {
                "sparse": SparseVectorParams(index=SparseIndexParams())
            }

        try:
            self.client.delete_collection(name)
        except Exception:
            pass

        self.client.create_collection(
            collection_name=name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )

    def upsert_documents(
        self,
        collection: str,
        texts: list[str],
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict[str, Any]] | None = None,
        metadata_list: list[dict[str, Any]] | None = None,
    ) -> int:
        """Insere documentos em lotes com IDs reprodutiveis e sem colisao."""
        points = []
        for i, text in enumerate(texts):
            payload = {"text": text, "chunk_index": i}
            if metadata_list and i < len(metadata_list):
                payload.update(metadata_list[i])
            point_id = self._make_point_id(text=text, metadata=payload)

            vectors = {"dense": dense_vectors[i]}
            sparse_vec = None
            if sparse_vectors and i < len(sparse_vectors):
                sv = sparse_vectors[i]
                sparse_vec = {"sparse": SparseVector(
                    indices=sv["indices"],
                    values=sv["values"],
                )}

            point = PointStruct(
                id=point_id,
                vector=vectors,
                payload=payload,
            )
            if sparse_vec:
                point.vector.update(sparse_vec)

            points.append(point)

        batch_size = 100
        for start in range(0, len(points), batch_size):
            batch = points[start:start + batch_size]
            self.client.upsert(collection_name=collection, points=batch)

        return len(points)

    def _make_point_id(self, text: str, metadata: dict[str, Any]) -> str:
        """Gera UUID deterministico e preserva chunks duplicados."""
        stable_identity = "|".join(
            [
                str(metadata.get("document_level", "")),
                str(metadata.get("filename", "")),
                str(metadata.get("chunk_id", metadata.get("chunk_index", ""))),
                text,
            ]
        )
        content_hash = hashlib.sha256(stable_identity.encode("utf-8")).hexdigest()
        return str(uuid.UUID(content_hash[:32]))

    def search_dense(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Executa busca vetorial apenas densa."""
        results = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            using="dense",
            limit=top_k,
            with_payload=True,
        )

        return [self._point_to_search_result(point) for point in results.points]

    def _search_sparse(
        self,
        collection: str,
        sparse_vector: dict[str, Any],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Executa busca apenas esparsa (estilo BM25)."""
        results = self.client.query_points(
            collection_name=collection,
            query=SparseVector(
                indices=sparse_vector["indices"],
                values=sparse_vector["values"],
            ),
            using="sparse",
            limit=top_k,
            with_payload=True,
        )

        return [self._point_to_search_result(point) for point in results.points]

    def _point_to_search_result(self, point: Any) -> SearchResult:
        """Converte um ponto retornado pelo Qdrant para o DTO de busca."""
        payload = point.payload or {}
        return SearchResult(
            text=payload.get("text", ""),
            score=point.score,
            metadata={key: value for key, value in payload.items() if key != "text"},
            doc_id=str(point.id),
        )

    def search_hybrid(
        self,
        collection: str,
        dense_vector: list[float],
        sparse_vector: dict[str, Any],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Executa busca híbrida com fusão recíproca de posições."""
        candidates_k = HYBRID_CANDIDATES
        dense_results = self.search_dense(collection, dense_vector, candidates_k)
        sparse_results = self._search_sparse(collection, sparse_vector, candidates_k)

        rrf_k = 60
        scores: dict[str, float] = {}
        doc_map: dict[str, SearchResult] = {}

        for rank, result in enumerate(dense_results):
            key = result.doc_id
            scores[key] = scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
            doc_map[key] = result

        for rank, result in enumerate(sparse_results):
            key = result.doc_id
            scores[key] = scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
            if key not in doc_map:
                doc_map[key] = result

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]

        return [
            SearchResult(
                text=doc_map[did].text,
                score=scores[did],
                metadata=doc_map[did].metadata,
                doc_id=did,
            )
            for did in sorted_ids
        ]

    def get_collection_info(self, collection: str) -> dict[str, Any]:
        """Retorna estatisticas da colecao."""
        info = self.client.get_collection(collection)
        return {
            "name": collection,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": str(info.status),
        }
