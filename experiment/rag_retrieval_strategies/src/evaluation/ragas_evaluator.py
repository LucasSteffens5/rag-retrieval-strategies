"""Avaliacao RAGAS pos-processada para JSONs brutos do benchmark."""
from __future__ import annotations

import math
import os
import time
from typing import Any, Mapping

from config import RESULTS_DIR  # noqa: E402
from evaluation.metrics.common import (  # noqa: E402
    base_report as common_base_report,
    collect_raw_files,
    context_texts,
    mean_summary,
    prepare_metric_input,
    write_json,
)

API_KEY_ENV_VAR = "GOOGLE_API_KEY"
RAGAS_RESULTS_DIR = os.getenv("RAGAS_RESULTS_DIR", os.path.join(RESULTS_DIR, "ragas"))
RAGAS_GOOGLE_MODEL = os.getenv("RAGAS_GOOGLE_MODEL", "gemini-2.5-flash-lite")
RAGAS_GOOGLE_EMBEDDING_MODEL = os.getenv("RAGAS_GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
RAGAS_MAX_WORKERS = max(1, int(os.getenv("RAGAS_MAX_WORKERS", "1")))
RAGAS_TIMEOUT_SECONDS = max(1, int(os.getenv("RAGAS_TIMEOUT_SECONDS", "600")))
METRICS = (
    "faithfulness",
    "context_precision",
    "context_recall",
    "answer_relevancy",
    "answer_correctness",
)


def evaluate_raw_file(
    raw_result_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] = RAGAS_RESULTS_DIR,
) -> dict[str, Any]:
    """Avalia um JSON bruto com RAGAS e grava um arquivo auxiliar auditavel."""
    if not os.getenv(API_KEY_ENV_VAR, "").strip():
        raise RuntimeError(f"{API_KEY_ENV_VAR} deve estar definida para executar RAGAS.")

    metric_input = prepare_metric_input(raw_result_path, output_dir, "ragas")
    report = common_base_report(
        raw_path=metric_input.raw_path,
        raw_hash=metric_input.raw_hash,
        raw=metric_input.raw,
        metric_family="ragas",
        parameters=ragas_parameters(),
        status="pending",
        extra_keys=("config", "benchmark", "corpus", "hardware"),
    )
    try:
        report.update(run_ragas(metric_input.raw))
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = safe_error(exc)
        report["summary"] = summary(
            metric_input.raw,
            evaluated=0,
            skipped=sum(1 for item in metric_input.raw.get("results", []) if "error" in item),
            metrics={},
        )
    return write_json(metric_input.sidecar_path, report)


def run_ragas(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Executa RAGAS usando a API oficial ``evaluate``."""
    rows, meta, skipped = build_rows(raw)
    if not rows:
        return {
            "status": "completed",
            "results": skipped,
            "summary": summary(raw, evaluated=0, skipped=len(skipped), metrics={}),
        }

    import asyncio

    from google import genai
    from ragas.embeddings import GoogleEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    client = genai.Client(api_key=os.environ[API_KEY_ENV_VAR])
    llm = build_ragas_llm(llm_factory=llm_factory)
    embeddings = GoogleEmbeddings(client=client, model=RAGAS_GOOGLE_EMBEDDING_MODEL)
    metrics = [
        ("faithfulness", Faithfulness(llm=llm), ("user_input", "response", "retrieved_contexts")),
        ("context_precision", ContextPrecision(llm=llm), ("user_input", "reference", "retrieved_contexts")),
        ("context_recall", ContextRecall(llm=llm), ("user_input", "retrieved_contexts", "reference")),
        ("answer_relevancy", AnswerRelevancy(llm=llm, embeddings=embeddings), ("user_input", "response")),
        ("answer_correctness", AnswerCorrectness(llm=llm, embeddings=embeddings), ("user_input", "response", "reference")),
    ]

    async def score_rows() -> list[dict[str, Any]]:
        """Executa metricas RAGAS modernas linha a linha."""
        semaphore = asyncio.Semaphore(RAGAS_MAX_WORKERS)

        async def score_one(item_meta: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
            scores: dict[str, Any] = {}
            metric_errors: dict[str, Any] = {}
            async with semaphore:
                for metric_name, metric, fields in metrics:
                    try:
                        metric_result = await asyncio.wait_for(
                            metric.ascore(**{field: row[field] for field in fields}),
                            timeout=RAGAS_TIMEOUT_SECONDS,
                        )
                        scores[metric_name] = json_safe(metric_result.value)
                    except Exception as exc:
                        scores[metric_name] = None
                        metric_errors[metric_name] = safe_error(exc)
                result_row = {**item_meta, "status": "completed", "scores": scores}
                if metric_errors:
                    result_row["metric_errors"] = metric_errors
                return result_row

        return await asyncio.gather(*(score_one(item_meta, row) for item_meta, row in zip(meta, rows)))

    started = time.perf_counter()
    results = asyncio.run(score_rows())
    missing = sum(score is None for item in results for score in item["scores"].values())
    report: dict[str, Any] = {
        "status": "failed" if missing else "completed",
        "duration_seconds": round(time.perf_counter() - started, 3),
        "results": [*results, *skipped],
        "summary": summary(
            raw,
            evaluated=len(results),
            skipped=len(skipped),
            metrics=summarize_scores(results),
            missing=missing,
        ),
    }
    if missing:
        report["error"] = {
            "exception_type": "RagasMissingScore",
            "message": f"{missing} metric scores were returned as null.",
        }
    return report


def build_ragas_llm(llm_factory: Any) -> Any:
    """Constrói o juiz RAGAS com Gemini."""
    from openai import AsyncOpenAI

    compatible_client = AsyncOpenAI(
        api_key=os.environ[API_KEY_ENV_VAR],
        base_url=GEMINI_BASE_URL,
    )
    llm = llm_factory(RAGAS_GOOGLE_MODEL, provider="openai", client=compatible_client)
    if hasattr(llm, "model_args") and isinstance(llm.model_args, dict):
        llm.model_args["max_tokens"] = 8192
    return llm


def build_rows(raw: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Converte resultados brutos em linhas RAGAS e itens ignorados."""
    rows: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in raw.get("results", []):
        base = {"query_id": item.get("query_id")}
        if "error" in item:
            skipped.append({**base, "status": "skipped_generation_error", "scores": {}})
            continue
        rows.append({
            "user_input": str(item.get("query", "")),
            "response": str(item.get("generated_answer", "")),
            "retrieved_contexts": context_texts(item),
            "reference": str(item.get("expected_answer", "")),
        })
        meta.append(base)
    return rows, meta, skipped


def ragas_parameters() -> dict[str, Any]:
    """Retorna parametros de reprodutibilidade da avaliacao RAGAS."""
    return {
        "api_key_env_var": API_KEY_ENV_VAR,
        "llm_provider": "google_gemini",
        "gemini_base_url": GEMINI_BASE_URL,
        "google_model": RAGAS_GOOGLE_MODEL,
        "google_embedding_model": RAGAS_GOOGLE_EMBEDDING_MODEL,
        "max_workers": RAGAS_MAX_WORKERS,
        "timeout_seconds": RAGAS_TIMEOUT_SECONDS,
        "metrics": list(METRICS),
    }


def summary(
    raw: Mapping[str, Any],
    evaluated: int,
    skipped: int,
    metrics: Mapping[str, Any],
    missing: int = 0,
) -> dict[str, Any]:
    """Resume o estado da avaliacao RAGAS."""
    return {
        "total_queries": len(raw.get("results", [])),
        "evaluated_queries": evaluated,
        "skipped_generation_error": skipped,
        "metrics": dict(metrics),
        "missing_scores": missing,
    }


def summarize_scores(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Calcula medias compactas por metrica para evitar  arquivos auxiliar verbosos."""
    return {
        metric: mean_summary([
            float(item["scores"][metric])
            for item in results
            if item["scores"].get(metric) is not None
        ])
        for metric in METRICS
    }


def json_safe(value: Any) -> Any:
    """Converte NaN em null para serializacao JSON valida."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def safe_error(exc: Exception) -> dict[str, str]:
    """Serializa erro sem vazar a chave de API."""
    message = str(exc)
    api_key = os.getenv(API_KEY_ENV_VAR, "")
    if api_key:
        message = message.replace(api_key, "<redacted>")
    return {"exception_type": exc.__class__.__name__, "message": message}


def main() -> int:
    """Executa RAGAS para todos os JSONs brutos da rodada corrente."""
    input_dir = os.path.join(RESULTS_DIR, "raw")
    raw_files = collect_raw_files(input_dir)
    if not raw_files:
        raise FileNotFoundError(f"Nenhum JSON bruto encontrado em {input_dir}.")
    print(f"Arquivos brutos para RAGAS: {len(raw_files)}")
    for index, raw_file in enumerate(raw_files, start=1):
        report = evaluate_raw_file(raw_file)
        sidecar = report.get("output", {}).get("sidecar_path", "")
        print(f"[{index}/{len(raw_files)}] {raw_file.name} -> {report.get('status')} ({sidecar})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
