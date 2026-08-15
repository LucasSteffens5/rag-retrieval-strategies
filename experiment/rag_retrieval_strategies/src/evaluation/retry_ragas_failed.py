"""Repete avaliações RAGAS com métricas ausentes."""
from __future__ import annotations

import asyncio
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from config import RESULTS_DIR  # noqa: E402
from evaluation.metrics.common import context_texts  # noqa: E402
from evaluation.ragas_evaluator import (  # noqa: E402
    API_KEY_ENV_VAR,
    METRICS,
    RAGAS_GOOGLE_EMBEDDING_MODEL,
    RAGAS_GOOGLE_MODEL,
    RAGAS_MAX_WORKERS,
    RAGAS_RESULTS_DIR,
    RAGAS_TIMEOUT_SECONDS,
    build_ragas_llm,
    json_safe,
    safe_error,
    summarize_scores,
    summary,
)

INTER_FILE_DELAY_SECONDS = 10.0

@dataclass(frozen=True)
class FailedQuery:
    """Consulta com métricas ausentes."""
    query_id: str
    missing_metrics: list[str]


@dataclass
class FailedFile:
    """Arquivo com consultas não avaliadas."""
    sidecar_path: Path
    raw_filename: str
    failed_queries: list[FailedQuery]
    total_queries: int = 0
    total_missing_scores: int = 0



def scan_failed_queries(ragas_dir: Path) -> list[FailedFile]:
    """Localiza consultas com métricas ausentes."""
    failed_files: list[FailedFile] = []

    sidecar_files = sorted(ragas_dir.glob("*_ragas.json"))
    if not sidecar_files:
        return failed_files

    for sidecar_path in sidecar_files:
        try:
            with open(sidecar_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        status = data.get("status")
        if status == "completed":
            continue

        results = data.get("results", [])
        raw_filename = data.get("source", {}).get("raw_filename", "")
        total_queries = len(results)

        failed_queries: list[FailedQuery] = []
        total_missing = 0

        for item in results:
            query_id = item.get("query_id")
            if not query_id:
                continue

            if item.get("status") == "skipped_generation_error":
                continue

            scores = item.get("scores", {})
            missing = [m for m in METRICS if m not in scores or scores[m] is None]
            if missing:
                failed_queries.append(FailedQuery(query_id=query_id, missing_metrics=missing))
                total_missing += len(missing)

        if failed_queries:
            failed_files.append(FailedFile(
                sidecar_path=sidecar_path,
                raw_filename=raw_filename,
                failed_queries=failed_queries,
                total_queries=total_queries,
                total_missing_scores=total_missing,
            ))

    return failed_files



def find_raw_for_sidecar(raw_filename: str, raw_dir: Path) -> Path | None:
    """Localiza o arquivo bruto correspondente."""
    raw_path = raw_dir / raw_filename
    if raw_path.exists():
        return raw_path
    return None



def extract_query_row(raw: Mapping[str, Any], query_id: str) -> dict[str, Any] | None:
    """Extrai os dados RAGAS de uma consulta."""
    for item in raw.get("results", []):
        if item.get("query_id") == query_id:
            if "error" in item:
                return None
            return {
                "user_input": str(item.get("query", "")),
                "response": str(item.get("generated_answer", "")),
                "retrieved_contexts": context_texts(item),
                "reference": str(item.get("expected_answer", "")),
            }
    return None



async def retry_single_query(
    query_id: str,
    row: dict[str, Any],
    missing_metrics: list[str],
    metric_instances: dict[str, tuple[Any, tuple[str, ...]]],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Repete apenas as métricas ausentes de uma consulta."""
    max_attempts = 3
    scores: dict[str, Any] = {}
    metric_errors: dict[str, Any] = {}

    async with semaphore:
        for m_name in missing_metrics:
            metric, fields = metric_instances[m_name]
            backoff = 5.0

            for attempt in range(1, max_attempts + 1):
                try:
                    metric_result = await asyncio.wait_for(
                        metric.ascore(**{field: row[field] for field in fields}),
                        timeout=RAGAS_TIMEOUT_SECONDS,
                    )
                    scores[m_name] = json_safe(metric_result.value)
                    break
                except Exception as exc:
                    if attempt == max_attempts:
                        scores[m_name] = None
                        metric_errors[m_name] = safe_error(exc)
                    else:
                        exc_name = type(exc).__name__
                        print(
                            f"        ⚠ [Tentativa {attempt}/{max_attempts}] Query {query_id} "
                            f"falhou em '{m_name}' ({exc_name}). Aguardando {backoff}s...",
                            flush=True,
                        )
                        await asyncio.sleep(backoff)
                        backoff *= 2.0

    return {"scores": scores, "metric_errors": metric_errors}



def update_sidecar(
    sidecar_path: Path,
    sidecar_data: dict[str, Any],
    retry_results: dict[str, dict[str, Any]],
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    """Atualiza o JSON com as novas métricas."""
    results = sidecar_data.get("results", [])

    for item in results:
        query_id = item.get("query_id")
        if query_id not in retry_results:
            continue

        new_data = retry_results[query_id]
        existing_scores = item.get("scores", {})
        existing_errors = item.get("metric_errors", {})

        for m_name, score_val in new_data["scores"].items():
            existing_scores[m_name] = score_val

        for m_name in new_data["scores"]:
            if m_name in new_data["metric_errors"]:
                existing_errors[m_name] = new_data["metric_errors"][m_name]
            else:
                existing_errors.pop(m_name, None)

        item["scores"] = existing_scores
        if existing_errors:
            item["metric_errors"] = existing_errors
        elif "metric_errors" in item:
            del item["metric_errors"]

    all_scored = [item for item in results if item.get("status") == "completed"]
    missing = sum(
        score is None
        for item in results
        for score in item.get("scores", {}).values()
    )

    sidecar_data["status"] = "failed" if missing else "completed"
    sidecar_data["summary"] = summary(
        raw,
        evaluated=len(all_scored),
        skipped=sum(1 for item in results if item.get("status") == "skipped_generation_error"),
        metrics=summarize_scores(all_scored),
        missing=missing,
    )

    if missing:
        sidecar_data["error"] = {
            "exception_type": "RagasMissingScore",
            "message": f"{missing} metric scores were returned as null.",
        }
    elif "error" in sidecar_data:
        del sidecar_data["error"]

    sidecar_data["output"] = {"sidecar_path": str(sidecar_path)}
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar_data, f, ensure_ascii=False, indent=2)

    return sidecar_data



def print_scan_report(failed_files: list[FailedFile], ragas_dir: Path) -> None:
    """Exibe o relatório de métricas ausentes."""
    total_sidecars = len(list(ragas_dir.glob("*_ragas.json")))
    total_failed_files = len(failed_files)
    total_failed_queries = sum(len(ff.failed_queries) for ff in failed_files)
    total_missing_scores = sum(ff.total_missing_scores for ff in failed_files)

    print(f"\n{'=' * 65}")
    print(f"  RAGAS Retry — Scan de Queries Falhadas")
    print(f"{'=' * 65}")
    print(f"  Total de sidecars encontrados:  {total_sidecars}")
    print(f"  Sidecars com falhas:            {total_failed_files}")
    print(f"  Total de queries com null:      {total_failed_queries}")
    print(f"  Total de scores null (métricas):{total_missing_scores}")
    print(f"{'=' * 65}\n")

    if not failed_files:
        print("  ✓ Nenhuma query falhada encontrada. Nada a fazer.\n")
        return

    for ff in failed_files:
        print(f"  📄 {ff.sidecar_path.name}")
        print(f"     Raw: {ff.raw_filename}")
        print(f"     Queries com falha: {len(ff.failed_queries)}/{ff.total_queries}")
        for fq in ff.failed_queries:
            metrics_str = ", ".join(fq.missing_metrics)
            print(f"       → {fq.query_id}: [{metrics_str}]")
        print()



def build_metric_instances() -> dict[str, tuple[Any, tuple[str, ...]]]:
    """Instancia métricas RAGAS uma única vez."""
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

    return {
        "faithfulness": (Faithfulness(llm=llm), ("user_input", "response", "retrieved_contexts")),
        "context_precision": (ContextPrecision(llm=llm), ("user_input", "reference", "retrieved_contexts")),
        "context_recall": (ContextRecall(llm=llm), ("user_input", "retrieved_contexts", "reference")),
        "answer_relevancy": (AnswerRelevancy(llm=llm, embeddings=embeddings), ("user_input", "response")),
        "answer_correctness": (AnswerCorrectness(llm=llm, embeddings=embeddings), ("user_input", "response", "reference")),
    }



def run(raw_dir: str, ragas_dir: str, dry_run: bool = False) -> int:
    """Repete avaliações RAGAS incompletas."""
    ragas_path = Path(ragas_dir)
    raw_path = Path(raw_dir)

    if not ragas_path.exists():
        print(f"Diretório de sidecars RAGAS não encontrado: {ragas_dir}")
        return 1

    failed_files = scan_failed_queries(ragas_path)
    print_scan_report(failed_files, ragas_path)

    if not failed_files:
        return 0

    if dry_run:
        print("  --dry-run ativado. Nenhuma query será reprocessada.\n")
        return 0

    print("  Instanciando métricas RAGAS...", flush=True)
    metric_instances = build_metric_instances()
    print("  ✓ Métricas instanciadas.\n", flush=True)

    total_files = len(failed_files)
    total_retried = 0
    total_fixed = 0
    total_still_failed = 0
    total_duration = 0.0

    for file_idx, ff in enumerate(failed_files, start=1):
        print(
            f"  [{file_idx}/{total_files}] Retrying {ff.sidecar_path.name} "
            f"({len(ff.failed_queries)} queries)...",
            flush=True,
        )

        raw_file = find_raw_for_sidecar(ff.raw_filename, raw_path)
        if raw_file is None:
            print(f"    ✗ Raw não encontrado: {ff.raw_filename}. Pulando.\n", flush=True)
            continue

        with open(raw_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        with open(ff.sidecar_path, "r", encoding="utf-8") as f:
            sidecar_data = json.load(f)

        file_start = time.perf_counter()
        semaphore = asyncio.Semaphore(RAGAS_MAX_WORKERS)

        async def process_file_queries() -> dict[str, dict[str, Any]]:
            results: dict[str, dict[str, Any]] = {}
            for q_idx, fq in enumerate(ff.failed_queries, start=1):
                row = extract_query_row(raw, fq.query_id)
                if row is None:
                    print(
                        f"    ⚠ Query {fq.query_id} não encontrada no raw ou tem erro de geração. Pulando.",
                        flush=True,
                    )
                    continue

                q_start = time.perf_counter()
                result = await retry_single_query(
                    query_id=fq.query_id,
                    row=row,
                    missing_metrics=fq.missing_metrics,
                    metric_instances=metric_instances,
                    semaphore=semaphore,
                )
                q_elapsed = time.perf_counter() - q_start

                any_still_null = any(v is None for v in result["scores"].values())
                status_char = "✓" if not any_still_null else "⚠"
                metrics_str = ", ".join(f"{m}={result['scores'].get(m)}" for m in fq.missing_metrics)

                print(
                    f"    [{q_idx}/{len(ff.failed_queries)}] {fq.query_id} "
                    f"{status_char} em {q_elapsed:.1f}s → {metrics_str}",
                    flush=True,
                )

                results[fq.query_id] = result
            return results

        retry_results = asyncio.run(process_file_queries())
        file_elapsed = time.perf_counter() - file_start
        total_duration += file_elapsed

        for qid, res in retry_results.items():
            total_retried += 1
            still_null = sum(1 for v in res["scores"].values() if v is None)
            if still_null == 0:
                total_fixed += 1
            else:
                total_still_failed += 1

        updated = update_sidecar(ff.sidecar_path, sidecar_data, retry_results, raw)
        new_status = updated.get("status", "unknown")
        print(
            f"  [{file_idx}/{total_files}] Concluído em {file_elapsed:.1f}s → status: {new_status}\n",
            flush=True,
        )

        gc.collect()

        if file_idx < total_files:
            print(
                f"  ⏳ Aguardando {INTER_FILE_DELAY_SECONDS:.0f}s antes do próximo arquivo...",
                flush=True,
            )
            time.sleep(INTER_FILE_DELAY_SECONDS)

    print(f"\n{'=' * 65}")
    print(f"  RAGAS Retry — Resumo Final")
    print(f"{'=' * 65}")
    print(f"  Arquivos processados:       {total_files}")
    print(f"  Queries retried:            {total_retried}")
    print(f"  Queries corrigidas:         {total_fixed}")
    print(f"  Queries ainda com falha:    {total_still_failed}")
    print(f"  Tempo total:                {total_duration:.1f}s ({total_duration / 60:.1f}min)")
    print(f"{'=' * 65}")

    if total_still_failed > 0:
        print(f"\n  ⚠ {total_still_failed} query(ies) ainda com scores null.")
        print("  Execute novamente para tentar mais retries.\n")
        return 1

    print("\n  ✓ Todas as queries falhadas foram corrigidas!\n")
    return 0



def main() -> int:
    """Resolve diretórios e opções da linha de comando."""
    dry_run = "--dry-run" in sys.argv

    raw_dir = os.path.join(RESULTS_DIR, "raw")
    ragas_dir = os.getenv("RAGAS_RESULTS_DIR", RAGAS_RESULTS_DIR)

    if not dry_run and not os.getenv("GOOGLE_API_KEY", "").strip():
        print("ERRO: GOOGLE_API_KEY não definida. Defina antes de executar.")
        return 1

    print(f"Diretório raw:   {raw_dir}")
    print(f"Diretório ragas: {ragas_dir}")
    if dry_run:
        print("Modo: --dry-run (apenas exibir scan)")

    return run(raw_dir, ragas_dir, dry_run=dry_run)


if __name__ == "__main__":
    sys.exit(main())
