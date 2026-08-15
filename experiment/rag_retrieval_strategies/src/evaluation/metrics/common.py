"""Utilitarios compartilhados para arquivos auxiliar de metricas."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from statistics import stdev
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class MetricInput:
    """Entrada normalizada para avaliadores de metricas pos-processadas."""

    raw_path: Path
    raw_hash: str
    raw: dict[str, Any]
    sidecar_path: Path


def collect_raw_files(input_dir: str, pattern: str = "*.json") -> list[Path]:
    """Retorna os JSONs brutos de um diretorio."""
    directory = Path(input_dir)
    return sorted(directory.glob(pattern)) if directory.exists() else []


def read_json(path: str | Path) -> dict[str, Any]:
    """Carrega JSON UTF-8."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: dict[str, Any]) -> dict[str, Any]:
    """Grava JSON UTF-8 e registra o caminho do arquivo auxiliar."""
    output_path = Path(path)
    data["output"] = {"sidecar_path": str(output_path)}
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return data


def prepare_metric_input(
    raw_result_path: str | Path,
    output_dir: str | Path,
    sidecar_suffix: str,
) -> MetricInput:
    """Carrega o JSON bruto e prepara o caminho deterministico do arquivo auxiliar."""
    raw_path = Path(raw_result_path)
    sidecar_path = Path(output_dir) / f"{raw_path.stem}_{sidecar_suffix}.json"
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    return MetricInput(
        raw_path=raw_path,
        raw_hash=sha256_file(raw_path),
        raw=read_json(raw_path),
        sidecar_path=sidecar_path,
    )


def sha256_file(path: str | Path) -> str:
    """Calcula SHA-256 de arquivo para rastreabilidade."""
    hasher = sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def source_metadata(
    raw_path: Path,
    raw_hash: str,
    raw: Mapping[str, Any],
    extra_keys: tuple[str, ...] = ("config", "benchmark", "corpus", "hardware"),
) -> dict[str, Any]:
    """Monta metadados de origem comuns aos arquivos auxiliares."""
    source = {
        "raw_filename": raw_path.name,
        "raw_sha256": raw_hash,
        "experiment_id": raw.get("experiment_id"),
        "run_id": raw.get("run_id"),
    }
    source.update({key: raw.get(key, {}) for key in extra_keys})
    return source


def base_report(
    raw_path: Path,
    raw_hash: str,
    raw: Mapping[str, Any],
    metric_family: str,
    parameters: dict[str, Any],
    status: str = "completed",
    extra_keys: tuple[str, ...] = ("config", "benchmark", "corpus", "hardware"),
) -> dict[str, Any]:
    """Monta a estrutura comum dos relatórios de métricas."""
    import time
    return {
        "schema_version": "1.0",
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": source_metadata(raw_path, raw_hash, raw, extra_keys),
        "metric_family": metric_family,
        "parameters": parameters,
        "results": [],
        "summary": {},
    }


def context_details(item: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Retorna os contextos auditaveis."""
    details = item.get("context_details")
    if not isinstance(details, list):
        raise ValueError(
            "Resultado bruto sem 'context_details'. "
        )
    return [context for context in details if isinstance(context, Mapping)]


def context_texts(item: Mapping[str, Any]) -> list[str]:
    """Extrai textos recuperados a partir de ``context_details``."""
    texts: list[str] = []
    for context in context_details(item):
        text = str(context.get("text", "")).strip()
        if text:
            texts.append(text)
    return texts


def count_status(rows: Sequence[Mapping[str, Any]], status: str) -> int:
    """Conta quantos itens possuem determinado status."""
    return sum(1 for row in rows if row.get("status") == status)


def mean_summary(
    values: Sequence[float],
    digits: int = 6,
    skipped: int | None = None,
) -> dict[str, float | int | None]:
    """Resume uma serie numerica por contagem e media."""
    summary: dict[str, float | int | None] = {
        "count": len(values),
        "mean": round(math.fsum(values) / len(values), digits) if values else None,
    }
    if skipped is not None:
        summary["skipped"] = skipped
    return summary


def mean_std_summary(values: Sequence[float], digits: int = 6) -> dict[str, float | int | None]:
    """Resume uma serie numerica por contagem, media e desvio-padrao amostral."""
    summary = mean_summary(values, digits)
    summary["std"] = round(stdev(values), digits) if len(values) > 1 else (0.0 if values else None)
    return summary
