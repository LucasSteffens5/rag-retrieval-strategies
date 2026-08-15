"""Adaptador de embedding para BGE-M3 (denso+esparso)."""
import contextlib
import io
from dataclasses import dataclass
from typing import Any

from config import EMBEDDING_MODELS


@dataclass
class EmbeddingResult:
    """Resultado de embedding de um texto."""
    dense: list[float]
    sparse: dict[str, Any] | None = None


class EmbeddingManager:
    """Gerencia o BGE-M3 no dispositivo definido pelo chamador."""

    def __init__(self, model_key: str, device: str = "cpu") -> None:
        if model_key not in EMBEDDING_MODELS:
            raise ValueError(f"Unknown embedding model: {model_key}. Available: {list(EMBEDDING_MODELS.keys())}")

        self.model_key = model_key
        self.model_config = EMBEDDING_MODELS[model_key]
        self.supports_sparse = self.model_config["supports_sparse"]
        self.device = "cuda:0" if device == "cuda" else device
        self.runtime = "cuda" if self.device.startswith("cuda") else "cpu"
        self._model: Any | None = None

    def _load_model(self) -> None:
        """Carrega o BGE-M3 uma unica vez no dispositivo configurado."""
        if self._model is not None:
            return
        if self.runtime == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA foi solicitado para embeddings, mas nao esta disponivel.")

        from FlagEmbedding import BGEM3FlagModel

        self._model = BGEM3FlagModel(
            self.model_config["model_name"],
            use_fp16=False,
            device=self.device,
        )
        self._model.target_devices = [self.device]
        print(f"[{self.runtime.upper()}] Modelo de embedding carregado: {self.model_config['model_name']}")

    def embed_documents(self, texts: list[str]) -> list[EmbeddingResult]:
        """Gera embeddings do corpus no dispositivo configurado."""
        output = self._encode(texts, batch_size=32)
        return [
            EmbeddingResult(
                dense=output["dense_vecs"][i].tolist(),
                sparse=self._convert_sparse(output["lexical_weights"][i]),
            )
            for i in range(len(texts))
        ]

    def embed_query(self, query: str) -> EmbeddingResult:
        """Gera embedding online de uma consulta no dispositivo configurado."""
        output = self._encode([query], batch_size=2)
        return EmbeddingResult(
            dense=output["dense_vecs"][0].tolist(),
            sparse=self._convert_sparse(output["lexical_weights"][0]),
        )

    def _encode(self, texts: list[str], batch_size: int) -> dict[str, Any]:
        """Executa FlagEmbedding sem barras tqdm no console experimental."""
        self._load_model()
        with contextlib.redirect_stderr(io.StringIO()):
            return self._model.encode_single_device(
                texts,
                batch_size=batch_size,
                max_length=self.model_config["max_tokens"],
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
                device=self.device,
            )

    def _convert_sparse(self, lexical_weights: dict[str, Any]) -> dict[str, list[int] | list[float]]:
        """Converte pesos lexicais do BGE-M3 para vetor esparso do Qdrant."""
        if not lexical_weights:
            return {"indices": [], "values": []}
        indices = []
        values = []
        for token_id, weight in lexical_weights.items():
            indices.append(int(token_id))
            values.append(float(weight))
        return {"indices": indices, "values": values}

    def warmup(self) -> None:
        """Aquece o modelo antes das métricas."""
        self.embed_query("consulta de aquecimento para o modelo de embedding")
