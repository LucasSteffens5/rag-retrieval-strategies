"""Pos-processamento padrao das metricas locais e analises auxiliares."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable

from config import RESULTS_DIR  # noqa: E402
from evaluation.metrics.common import collect_raw_files
from evaluation.metrics.latency_metrics import evaluate_raw_file as evaluate_latency_raw_file
from evaluation.metrics.retrieval_metrics import evaluate_raw_file as evaluate_retrieval_raw_file
from evaluation.metrics.scope_behavior_analysis import evaluate_raw_file as evaluate_scope_behavior_raw_file

MetricEvaluator = Callable[[Path], dict[str, Any]]

LOCAL_METRICS: tuple[tuple[str, MetricEvaluator], ...] = (
    ("retrieval", evaluate_retrieval_raw_file),
    ("latency", evaluate_latency_raw_file),
)

AUXILIARY_ANALYSES: tuple[tuple[str, MetricEvaluator], ...] = (
    ("scope_behavior", evaluate_scope_behavior_raw_file),
)


def main() -> int:
    """Executa metricas locais e analises auxiliares sobre os JSONs brutos."""
    input_dir = os.path.join(RESULTS_DIR, "raw")
    raw_files = collect_raw_files(input_dir)
    if not raw_files:
        raise FileNotFoundError(f"Nenhum JSON bruto encontrado em {input_dir}.")

    print(f"Arquivos brutos: {len(raw_files)}")
    for family, evaluator in LOCAL_METRICS:
        counts: dict[str, int] = {}
        print(f"Familia de metrica: {family}")
        for index, raw_file in enumerate(raw_files, start=1):
            report = evaluator(raw_file)
            status = str(report.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
            sidecar = Path(report.get("output", {}).get("sidecar_path", "")).name
            print(f"[{index}/{len(raw_files)}] {raw_file.name} -> {status} ({sidecar})")
        print(f"Resumo {family}: {counts}")

    for analysis_name, evaluator in AUXILIARY_ANALYSES:
        counts: dict[str, int] = {}
        print(f"Analise complementar: {analysis_name}")
        for index, raw_file in enumerate(raw_files, start=1):
            report = evaluator(raw_file)
            status = str(report.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
            sidecar = Path(report.get("output", {}).get("sidecar_path", "")).name
            print(f"[{index}/{len(raw_files)}] {raw_file.name} -> {status} ({sidecar})")
        print(f"Resumo {analysis_name}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
