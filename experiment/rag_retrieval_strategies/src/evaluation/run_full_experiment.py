"""Executa uma rodada completa e isolada no Docker."""
from __future__ import annotations

import os
import re
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from config import EMBEDDING_MODELS, LLM_MODELS, OLLAMA_HOST, RERANKER_MODEL


def parse_bool_env(name: str, default: str = "0") -> bool:
    """Interpreta variaveis booleanas de ambiente usadas na orquestracao."""
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "sim"}


def make_experiment_run_id() -> str:
    """Retorna o identificador da rodada ou gera um timestamp UTC estavel."""
    requested = os.getenv("EXPERIMENT_RUN_ID", "").strip()
    run_id = requested or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError(
            "EXPERIMENT_RUN_ID deve conter apenas letras, numeros, ponto, "
            f"underscore ou hifen: {run_id!r}"
        )
    return run_id


def repository_root() -> Path:
    """Resolve a raiz do repositorio tanto no Docker quanto no workspace local."""
    return Path(__file__).resolve().parents[4]


def safe_remove_tree(path: Path, expected_name: str) -> None:
    """Remove uma arvore transitória somente quando o alvo esperado confere."""
    resolved = path.resolve()
    if resolved.name != expected_name:
        raise RuntimeError(
            f"Recusando remover caminho inesperado para {expected_name}: {resolved}"
        )
    if resolved.exists():
        shutil.rmtree(resolved)


def safe_clean_directory(path: Path, expected_name: str) -> None:
    """Esvazia um diretorio transitorio preservando o ponto de montagem Docker."""
    resolved = path.resolve()
    if resolved.name != expected_name:
        raise RuntimeError(
            f"Recusando limpar caminho inesperado para {expected_name}: {resolved}"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    for child in resolved.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def build_child_env(run_dir: Path, corpus_dir: Path, benchmark_path: Path) -> dict[str, str]:
    """Monta ambiente reprodutivel compartilhado pelas etapas subprocessadas."""
    root = repository_root()
    src_dir = root / "experiment" / "rag_retrieval_strategies" / "src"
    database_dir = root / "database"
    child_env = os.environ.copy()
    python_paths = [str(src_dir), str(database_dir)]
    if child_env.get("PYTHONPATH"):
        python_paths.append(child_env["PYTHONPATH"])
    child_env.update(
        {
            "PYTHONPATH": os.pathsep.join(python_paths),
            "CORPUS_DIR": str(corpus_dir),
            "BENCHMARK_PATH": str(benchmark_path),
            "RESULTS_DIR": str(run_dir),
            "PYTHONUNBUFFERED": "1",
        }
    )
    return child_env


def run_step(label: str, script_path: Path, env: dict[str, str]) -> None:
    """Executa uma etapa Python e falha preservando o codigo de retorno."""
    print(f"\n=== {label} ===", flush=True)
    subprocess.run([sys.executable, str(script_path)], check=True, env=env)


def wait_for_ollama(timeout_seconds: int = 120) -> None:
    """Aguarda o Ollama responder antes de solicitar modelos."""
    deadline = time.monotonic() + timeout_seconds
    url = f"{OLLAMA_HOST.rstrip('/')}/api/tags"
    while time.monotonic() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"Ollama indisponivel apos {timeout_seconds}s em {url}.")


def ensure_ollama_models() -> None:
    """Garante que os LLMs configurados estejam presentes no Ollama."""
    wait_for_ollama()
    host = OLLAMA_HOST.rstrip("/")
    for model_name in LLM_MODELS:
        print(f"\n=== Modelo Ollama: {model_name} ===", flush=True)
        response = requests.post(
            f"{host}/api/pull",
            json={"name": model_name, "stream": True},
            stream=True,
            timeout=1800,
        )
        response.raise_for_status()
        last_status = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            data = json.loads(line)
            status = str(data.get("status") or "").strip()
            if status and status != last_status:
                print(f"  {status}", flush=True)
                last_status = status
            if data.get("error"):
                raise RuntimeError(f"Falha ao baixar {model_name}: {data['error']}")


def ensure_huggingface_models() -> None:
    """Garante cache local dos modelos Hugging Face usados no experimento."""
    print("\n=== Modelos Hugging Face ===", flush=True)
    from FlagEmbedding import BGEM3FlagModel, FlagReranker
    from transformers import AutoTokenizer

    embedding_model = EMBEDDING_MODELS["bge-m3"]["model_name"]
    AutoTokenizer.from_pretrained(embedding_model, use_fast=True)
    BGEM3FlagModel(embedding_model, use_fp16=False, device="cpu")
    FlagReranker(RERANKER_MODEL, use_fp16=False, device="cpu")
    print("  Cache Hugging Face validado.", flush=True)


def prepare_run_directory(results_base_dir: Path, run_id: str) -> Path:
    """Cria o diretorio isolado da rodada conforme a politica clean-run."""
    run_dir = results_base_dir / "runs" / run_id
    force_clean = parse_bool_env("FORCE_CLEAN_RUN")
    if run_dir.exists():
        if not force_clean:
            raise FileExistsError(
                f"A rodada {run_id} ja existe em {run_dir}. "
                "Use FORCE_CLEAN_RUN=1 para substituir apenas esta rodada."
            )
        safe_remove_tree(run_dir, expected_name=run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main() -> int:
    """Executa preparacao, ingestao, benchmark, metricas locais e RAGAS opcional."""
    root = repository_root()
    results_base_dir = Path(
        os.getenv(
            "RESULTS_BASE_DIR",
            root / "experiment" / "rag_retrieval_strategies" / "results",
        )
    )
    corpus_dir = Path(os.getenv("CORPUS_DIR", root / "database" / "corpus_processado"))
    benchmark_path = Path(
        os.getenv(
            "BENCHMARK_PATH",
            root
            / "experiment"
            / "rag_retrieval_strategies"
            / "data"
            / "benchmark"
            / "perguntas_benchmark_edital_docente_ufmt.json",
        )
    )
    run_id = make_experiment_run_id()
    run_dir = prepare_run_directory(results_base_dir, run_id)
    safe_clean_directory(corpus_dir, expected_name="corpus_processado")

    env = build_child_env(run_dir=run_dir, corpus_dir=corpus_dir, benchmark_path=benchmark_path)
    scripts = {
        "prepare_corpus": root / "database" / "prepare_corpus.py",
        "ingest": root / "experiment" / "rag_retrieval_strategies" / "src" / "ingest.py",
        "benchmark": root
        / "experiment"
        / "rag_retrieval_strategies"
        / "src"
        / "evaluation"
        / "run_benchmark.py",
        "metrics": root
        / "experiment"
        / "rag_retrieval_strategies"
        / "src"
        / "evaluation"
        / "run_metrics.py",
        "ragas": root
        / "experiment"
        / "rag_retrieval_strategies"
        / "src"
        / "evaluation"
        / "ragas_evaluator.py",
    }

    ensure_ollama_models()
    ensure_huggingface_models()
    run_step("Preparacao do corpus", scripts["prepare_corpus"], env)
    run_step("Ingestao vetorial", scripts["ingest"], env)
    run_step("Benchmark", scripts["benchmark"], env)
    run_step("Metricas locais", scripts["metrics"], env)

    if parse_bool_env("RUN_RAGAS") and env.get("GOOGLE_API_KEY", "").strip():
        run_step("RAGAS opcional", scripts["ragas"], env)
    elif parse_bool_env("RUN_RAGAS"):
        print("RUN_RAGAS=1 definido, mas GOOGLE_API_KEY ausente; RAGAS ignorado.", flush=True)

    print(f"\nRodada concluida: {run_id}", flush=True)
    print(f"Resultados: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
