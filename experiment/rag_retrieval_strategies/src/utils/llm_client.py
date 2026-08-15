"""Adapta o Ollama para geração de texto instrumentada."""
import requests
import time
from dataclasses import dataclass
from typing import Any

from config import (
    OLLAMA_HOST,
    LLM_TEMPERATURE,
    LLM_SEED,
    LLM_NUM_CTX,
    LLM_NUM_GPU,
    LLM_TOP_K,
    LLM_TOP_P,
    LLM_THINKING_ENABLED,
)


@dataclass
class LLMResponse:
    """Resultado de uma chamada ao LLM."""
    text: str
    model: str
    tokens_generated: int
    tokens_per_second: float
    generation_ms: float
    thinking_enabled: bool
    thinking_text: str


class OllamaClient:
    """Cliente para o Ollama com instrumentacao de latencia."""

    def __init__(
        self,
        model_name: str,
        temperature: float = LLM_TEMPERATURE,
        seed: int = LLM_SEED,
        num_ctx: int = LLM_NUM_CTX,
        host: str = OLLAMA_HOST,
    ):
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.options = {
            "temperature": temperature,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_gpu": LLM_NUM_GPU,
            "top_k": LLM_TOP_K,
            "top_p": LLM_TOP_P,
        }

    def generate(self, prompt: str, system_prompt: str = "", format: Any = None) -> LLMResponse:
        """Gera texto usando a API do Ollama."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "think": LLM_THINKING_ENABLED,
            "options": self.options,
        }
        if format:
            payload["format"] = format
        if system_prompt:
            payload["system"] = system_prompt

        start = time.perf_counter()
        resp = requests.post(
            f"{self.host}/api/generate",
            json=payload,
            timeout=600,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        data = resp.json()
        thinking_text = str(data.get("thinking") or "").strip()
        if not LLM_THINKING_ENABLED and thinking_text:
            raise RuntimeError(
                "O Ollama retornou conteudo de thinking apesar de think=false. "
                f"model={self.model_name}. Abortando para preservar metricas de tokens."
            )

        eval_count = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 1)
        tokens_per_second = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else 0

        return LLMResponse(
            text=data.get("response", "").strip(),
            model=self.model_name,
            tokens_generated=eval_count,
            tokens_per_second=round(tokens_per_second, 2),
            generation_ms=round(elapsed_ms, 2),
            thinking_enabled=LLM_THINKING_ENABLED,
            thinking_text=thinking_text,
        )

    @staticmethod
    def normalize_model_name(name: str) -> str:
        """Normaliza identificadores Ollama para comparacao exata e reprodutivel."""
        clean_name = name.strip()
        if clean_name.endswith(":latest"):
            return clean_name[: -len(":latest")]
        return clean_name

    def assert_gpu_only(self) -> dict[str, Any]:
        """Falha se o Ollama nao reportar o modelo integralmente em VRAM."""
        resp = requests.get(f"{self.host}/api/ps", timeout=10)
        resp.raise_for_status()
        expected_name = self.normalize_model_name(self.model_name)
        model_info = next(
            (
                model
                for model in resp.json().get("models", [])
                if expected_name
                in {
                    self.normalize_model_name(model.get("name") or ""),
                    self.normalize_model_name(model.get("model") or ""),
                }
            ),
            None,
        )
        if not model_info:
            raise RuntimeError(f"Modelo {self.model_name} nao aparece em /api/ps apos warmup.")

        size = int(model_info.get("size") or 0)
        size_vram = int(model_info.get("size_vram") or 0)
        if not size or size_vram < size:
            raise RuntimeError(
                "LLM carregado parcialmente fora da GPU: "
                f"model={self.model_name}, size_vram={size_vram}, size={size}. "
                "Abortando para cumprir a regra experimental: LLM somente em GPU."
            )
        return model_info

    def unload(self) -> None:
        """Solicita ao Ollama descarregar o modelo da memoria."""
        requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model_name, "keep_alive": 0},
            timeout=30,
        )

    @classmethod
    def unload_loaded_models(cls, host: str = OLLAMA_HOST) -> None:
        """Descarrega modelos para evitar heranca de VRAM entre LLMs."""
        clean_host = host.rstrip("/")
        resp = requests.get(f"{clean_host}/api/ps", timeout=10)
        resp.raise_for_status()
        for model_info in resp.json().get("models", []):
            model_name = model_info.get("name") or model_info.get("model")
            if model_name:
                requests.post(
                    f"{clean_host}/api/generate",
                    json={"model": model_name, "keep_alive": 0},
                    timeout=30,
                )

    def health_check(self) -> bool:
        """Verifica se o Ollama responde e se o modelo esta disponivel."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=10)
            if resp.status_code != 200:
                return False
            expected_name = self.normalize_model_name(self.model_name)
            models = {
                self.normalize_model_name(m.get("name") or "")
                for m in resp.json().get("models", [])
            }
            return expected_name in models
        except requests.RequestException:
            return False

    def __repr__(self):
        return f"OllamaClient(model={self.model_name}, host={self.host})"
