"""Gera tabelas e figuras finais reproduzíveis a partir dos JSONs do experimento."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any, Final, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


RAGAS_METRICS: Final[tuple[str, ...]] = (
    "faithfulness",
    "context_precision",
    "context_recall",
    "answer_relevancy",
    "answer_correctness",
)
RETRIEVAL_METRICS: Final[tuple[str, ...]] = ("recall_at_5", "ndcg_at_5")
STRATEGIES: Final[tuple[str, ...]] = (
    "naive", "hybrid", "reranking", "hyde", "adaptive",
)
STRATEGY_COLORS: Final[dict[str, str]] = {
    "naive": "#4C78A8",
    "hybrid": "#59C3E1",
    "reranking": "#EF6375",
    "hyde": "#268A35",
    "adaptive": "#D4BF3F",
}
MODEL_MARKERS: Final[tuple[str, ...]] = ("o", "s", "^")
MEDIAN_COLUMNS: Final[dict[str, str]] = {
    "faithfulness": "faith_median",
    "context_precision": "ctx_prec_median",
    "context_recall": "ctx_recall_median",
    "answer_relevancy": "ans_rel_median",
    "answer_correctness": "ac_median",
    "recall_at_5": "recall5_median",
    "ndcg_at_5": "ndcg5_median",
}
MEAN_COLUMNS: Final[dict[str, str]] = {
    "faithfulness": "faith_mean",
    "context_precision": "ctx_prec_mean",
    "context_recall": "ctx_recall_mean",
    "answer_relevancy": "ans_rel_mean",
    "answer_correctness": "ac_mean",
    "recall_at_5": "recall5_mean",
    "ndcg_at_5": "ndcg5_mean",
}
REFUSAL_TEXT: Final[str] = "Informação não encontrada no contexto fornecido."
EXPECTED_FILES: Final[int] = 45
EXPECTED_FULL_ROWS: Final[int] = 50
EXPECTED_RETRIEVAL_ROWS: Final[int] = 46
EXPECTED_FILTERED_ROWS: Final[int] = 46
EXPECTED_RUNS: Final[int] = 3
EXPECTED_MODELS: Final[int] = 3
EXPECTED_CONFIGURATIONS: Final[int] = len(STRATEGIES) * EXPECTED_MODELS
EXPECTED_ROWS_PER_CONFIG: Final[int] = EXPECTED_FILTERED_ROWS * EXPECTED_RUNS
EXPECTED_ROWS_PER_STRATEGY: Final[int] = EXPECTED_ROWS_PER_CONFIG * EXPECTED_MODELS
DPI: Final[int] = 300



def parse_args() -> argparse.Namespace:
    default_run = (
        Path(__file__).resolve().parents[1]
        / "results" / "runs" / "20260608T004938Z"
    )
    parser = argparse.ArgumentParser(
        description="Gera tabelas e figuras finais agregadas por mediana."
    )
    parser.add_argument("--run-root", type=Path, default=default_run)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Raiz JSON inválida: {path}")
    return payload


def json_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.json"))
    if len(files) != EXPECTED_FILES:
        raise ValueError(
            f"Esperados {EXPECTED_FILES} JSONs em {directory}. "
            f"encontrados {len(files)}."
        )
    return files


def metadata(payload: Mapping[str, Any], family: str, path: Path) -> dict[str, str]:
    source = payload if family == "raw" else payload["source"]
    config = source["config"]
    return {
        "experiment_id": str(source["experiment_id"]),
        "run_id": str(source["run_id"]),
        "model": str(config["llm_model"]),
        "strategy": str(config["retrieval_strategy"]),
        "source_file": path.name,
    }


def finite(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.nan
    numeric = float(value)
    return numeric if math.isfinite(numeric) else math.nan


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return re.sub(r"\s+", " ", text)


def validate_key(frame: pd.DataFrame, family: str) -> None:
    key = ["experiment_id", "run_id", "model", "strategy", "query_id"]
    if frame.duplicated(key, keep=False).any():
        raise ValueError(f"Chaves duplicadas em {family}.")


def model_size(model: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)b", str(model))
    return float(match.group(1)) if match else math.inf


def short_model(model: str) -> str:
    size = model_size(model)
    return f"Qwen3 {size:g}B" if math.isfinite(size) else str(model)


def order_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    models = sorted(result["model"].dropna().unique(), key=model_size)
    result["model"] = pd.Categorical(result["model"], models, ordered=True)
    result["strategy"] = pd.Categorical(result["strategy"], STRATEGIES, ordered=True)
    return result


def load_raw(directory: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in json_files(directory):
        payload = read_json(path)
        meta = metadata(payload, "raw", path)
        results = payload.get("results")
        if not isinstance(results, list) or len(results) != EXPECTED_FULL_ROWS:
            raise ValueError(f"Raw deve possuir 50 resultados: {path}")
        for result in results:
            source_docs = result.get("source_documents", [])
            no_reference = isinstance(source_docs, list) and len(source_docs) == 0
            generated_answer = normalize_text(result.get("generated_answer"))
            correct_refusal = no_reference and (
                generated_answer == normalize_text(result.get("expected_answer"))
            )
            extra = result.get("extra", {})
            selected_route = (
                extra.get("selected_route") if isinstance(extra, Mapping) else None
            )
            rows.append({
                **meta,
                "query_id": str(result["query_id"]),
                "no_reference_document": no_reference,
                "correct_refusal": correct_refusal,
                "refused": generated_answer == normalize_text(REFUSAL_TEXT),
                "selected_route": selected_route,
            })
    frame = pd.DataFrame(rows)
    validate_key(frame, "raw")
    return frame


def load_ragas(directory: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in json_files(directory):
        payload = read_json(path)
        meta = metadata(payload, "ragas", path)
        results = payload.get("results")
        if not isinstance(results, list) or len(results) != EXPECTED_FULL_ROWS:
            raise ValueError(f"RAGAS deve possuir 50 resultados: {path}")
        for result in results:
            scores = result.get("scores", {})
            row: dict[str, Any] = {**meta, "query_id": str(result["query_id"])}
            for metric in RAGAS_METRICS:
                value = finite(scores.get(metric))
                row[metric] = min(value, 1.0) if math.isfinite(value) else math.nan
            rows.append(row)
    frame = pd.DataFrame(rows)
    validate_key(frame, "RAGAS")
    return frame


def load_latency(directory: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in json_files(directory):
        payload = read_json(path)
        meta = metadata(payload, "latency", path)
        results = payload.get("results")
        if not isinstance(results, list) or len(results) != EXPECTED_FULL_ROWS:
            raise ValueError(f"Latência deve possuir 50 resultados: {path}")
        for result in results:
            rows.append({
                **meta,
                "query_id": str(result["query_id"]),
                "total_ms": finite(result.get("total_ms")),
                "routing_ms": finite(result.get("routing_ms")),
            })
    frame = pd.DataFrame(rows)
    validate_key(frame, "latência")
    return frame


def load_retrieval(directory: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in json_files(directory):
        payload = read_json(path)
        meta = metadata(payload, "retrieval", path)
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError(f"Resultados de recuperação inválidos: {path}")
        valid = [
            r for r in results
            if r.get("status") == "completed"
            and all(
                math.isfinite(finite(r.get(m))) for m in RETRIEVAL_METRICS
            )
        ]
        if len(valid) != EXPECTED_RETRIEVAL_ROWS:
            raise ValueError(f"Esperadas 46 perguntas avaliáveis: {path}")
        for result in valid:
            rows.append({
                **meta,
                "query_id": str(result["query_id"]),
                **{m: finite(result.get(m)) for m in RETRIEVAL_METRICS},
            })
    frame = pd.DataFrame(rows)
    validate_key(frame, "recuperação")
    return frame



def validate_denominators(
    raw: pd.DataFrame,
    ragas: pd.DataFrame,
    retrieval: pd.DataFrame,
    latency: pd.DataFrame,
) -> None:
    group = ["model", "strategy", "run_id"]
    expected = (
        (raw, 50, "raw"),
        (ragas, 50, "RAGAS"),
        (retrieval, 46, "recuperação"),
        (latency, 50, "latência"),
    )
    for frame, count, label in expected:
        observed = frame.groupby(group, observed=True).size()
        if set(observed) != {count}:
            raise ValueError(
                f"Denominadores inválidos em {label}: {observed.to_dict()}"
            )
    no_ref = raw[raw["no_reference_document"]].groupby(group, observed=True).size()
    if set(no_ref) != {4}:
        raise ValueError(
            f"Esperadas 4 perguntas sem referência por arquivo: {no_ref.to_dict()}"
        )


def validate_filtered_counts(
    ragas_f: pd.DataFrame,
    retrieval_f: pd.DataFrame,
    latency_f: pd.DataFrame,
) -> None:
    """Confirma 46 consultas por execução após a filtragem."""
    group = ["model", "strategy", "run_id"]
    for frame, label in [
        (ragas_f, "RAGAS filtrado"),
        (retrieval_f, "Recuperação filtrada"),
        (latency_f, "Latência filtrada"),
    ]:
        observed = frame.groupby(group, observed=True).size()
        if set(observed) != {EXPECTED_FILTERED_ROWS}:
            raise ValueError(
                f"Denominadores inválidos em {label} após filtragem: "
                f"{observed.to_dict()}"
            )
    print("  - Denominadores validados: 46 queries/run em todas as famílias.")


def cross_validate_with_raw_json(
    run_root: Path,
    ragas_filtered: pd.DataFrame,
    latency_filtered: pd.DataFrame,
) -> None:
    """Amostra JSONs originais e verifica correspondência com os DataFrames."""
    print("\n-- Validação cruzada contra JSONs originais --")
    
    ragas_dir = run_root / "ragas"
    latency_dir = run_root / "metrics" / "latency"
    
    errors = []
    
    for path in list(json_files(ragas_dir))[:3]:
        payload = read_json(path)
        meta = metadata(payload, "ragas", path)
        model, strategy, run_id = meta["model"], meta["strategy"], meta["run_id"]
        
        results = payload["results"]
        for r in results:
            qid = str(r["query_id"])
            mask = (
                (ragas_filtered["model"].astype(str) == model)
                & (ragas_filtered["strategy"].astype(str) == strategy)
                & (ragas_filtered["run_id"].astype(str) == run_id)
                & (ragas_filtered["query_id"] == qid)
            )
            in_filtered = mask.any()
            score = r.get("scores", {}).get("answer_correctness")
            if in_filtered and score is not None:
                df_value = float(
                    ragas_filtered.loc[mask, "answer_correctness"].iloc[0]
                )
                json_value = min(float(score), 1.0) if math.isfinite(float(score)) else math.nan
                if abs(df_value - json_value) > 1e-10:
                    errors.append(
                        f"Divergência em {path.name} query {qid}: "
                        f"JSON={json_value:.6f} vs DF={df_value:.6f}"
                    )
    
    for path in list(json_files(latency_dir))[:3]:
        payload = read_json(path)
        meta = metadata(payload, "latency", path)
        model, strategy, run_id = meta["model"], meta["strategy"], meta["run_id"]
        
        results = payload["results"]
        for r in results:
            qid = str(r["query_id"])
            mask = (
                (latency_filtered["model"].astype(str) == model)
                & (latency_filtered["strategy"].astype(str) == strategy)
                & (latency_filtered["run_id"].astype(str) == run_id)
                & (latency_filtered["query_id"] == qid)
            )
            in_filtered = mask.any()
            total_ms = r.get("total_ms")
            if in_filtered and total_ms is not None:
                df_value = float(latency_filtered.loc[mask, "total_ms"].iloc[0])
                json_value = float(total_ms)
                if abs(df_value - json_value) > 1e-10:
                    errors.append(
                        f"Divergência latência em {path.name} query {qid}: "
                        f"JSON={json_value:.2f} vs DF={df_value:.2f}"
                    )
    
    if errors:
        for e in errors:
            print(f"  [ERROR] {e}")
        raise ValueError(f"Encontradas {len(errors)} divergências na validação cruzada.")
    
    print("  - Validação cruzada OK: dados nos DataFrames correspondem aos JSONs originais.")



def build_consolidated_table(
    ragas: pd.DataFrame,
    retrieval: pd.DataFrame,
    latency: pd.DataFrame,
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """Constrói uma linha por modelo×estratégia com medianas e médias."""
    keys = ["model", "strategy"]
    summary = build_summary_table(ragas, retrieval, latency, keys)
    
    refusals = raw.groupby(keys, observed=True).agg(
        correct_refusals=("correct_refusal", "sum"),
        no_ref_questions=("no_reference_document", "sum"),
    )
    refusals["refusal_rate"] = refusals["correct_refusals"] / refusals["no_ref_questions"]
    
    result = (
        summary.set_index(keys)
        .join(refusals[["refusal_rate"]])
        .reset_index()
    )
    result["model_label"] = result["model"].apply(lambda m: short_model(str(m)))
    return result.sort_values(["model", "strategy"]).reset_index(drop=True)


def build_median_table(
    ragas: pd.DataFrame,
    retrieval: pd.DataFrame,
    latency: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    """Agrega qualidade, recuperação e latência pela mediana."""
    return build_summary_table(ragas, retrieval, latency, keys, include_means=False)


def build_summary_table(
    ragas: pd.DataFrame,
    retrieval: pd.DataFrame,
    latency: pd.DataFrame,
    keys: list[str],
    include_means: bool = True,
) -> pd.DataFrame:
    """Agrega métricas por mediana e, opcionalmente, por média."""
    ragas_group = ragas.groupby(keys, observed=True)
    retrieval_group = retrieval.groupby(keys, observed=True)
    latency_group = latency.groupby(keys, observed=True)

    quality = ragas_group[list(RAGAS_METRICS)].median().rename(
        columns={metric: MEDIAN_COLUMNS[metric] for metric in RAGAS_METRICS}
    )
    quality["n_quality"] = ragas_group.size()

    retrieval_medians = retrieval_group[list(RETRIEVAL_METRICS)].median().rename(
        columns={metric: MEDIAN_COLUMNS[metric] for metric in RETRIEVAL_METRICS}
    )
    retrieval_medians["n_retrieval"] = retrieval_group.size()

    latency_medians = latency_group[["total_ms"]].median().rename(
        columns={"total_ms": "lat_median_ms"}
    )
    latency_medians["n_latency"] = latency_group.size()

    result = (
        quality.join(retrieval_medians)
        .join(latency_medians)
        .reset_index()
    )

    if not include_means:
        return result

    quality_means = ragas_group[list(RAGAS_METRICS)].mean().rename(
        columns={metric: MEAN_COLUMNS[metric] for metric in RAGAS_METRICS}
    )
    retrieval_means = retrieval_group[list(RETRIEVAL_METRICS)].mean().rename(
        columns={metric: MEAN_COLUMNS[metric] for metric in RETRIEVAL_METRICS}
    )
    latency_means = latency_group[["total_ms"]].mean().rename(
        columns={"total_ms": "lat_mean_ms"}
    )
    means = quality_means.join(retrieval_means).join(latency_means).reset_index()

    return result.merge(means, on=keys, how="left")


def build_strategy_summary(
    ragas: pd.DataFrame,
    retrieval: pd.DataFrame,
    latency: pd.DataFrame,
) -> pd.DataFrame:
    """Constrói estatísticas agregadas por estratégia para sustentar o manuscrito."""
    result = build_summary_table(ragas, retrieval, latency, ["strategy"])
    result["strategy"] = pd.Categorical(
        result["strategy"], STRATEGIES, ordered=True
    )
    return result.sort_values("strategy").reset_index(drop=True)


def build_operational_summary(
    raw: pd.DataFrame,
    latency: pd.DataFrame,
) -> pd.DataFrame:
    """Constrói uma tabela longa de roteamento e abstenção."""
    adaptive_raw = raw[raw["strategy"].astype(str) == "adaptive"]
    adaptive_latency = latency[latency["strategy"].astype(str) == "adaptive"]
    route_counts = adaptive_raw["selected_route"].dropna().astype(str).value_counts()
    routing_values = (
        pd.to_numeric(adaptive_latency["routing_ms"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if int(route_counts.sum()) != len(adaptive_raw):
        raise ValueError("Há decisões adaptativas sem rota selecionada.")
    if len(routing_values) != len(adaptive_latency):
        raise ValueError("Há decisões adaptativas sem latência de roteamento.")

    decisions = int(route_counts.sum())
    rows: list[dict[str, Any]] = [{
        "analysis": "adaptive_routing",
        "group": "all",
        "metric": "routing_ms",
        "aggregation": "median",
        "numerator": math.nan,
        "denominator": decisions,
        "value": float(routing_values.median()),
    }]
    rows.extend({
        "analysis": "adaptive_routing",
        "group": str(route),
        "metric": "selected_route_rate",
        "aggregation": "proportion",
        "numerator": int(count),
        "denominator": decisions,
        "value": float(count) / decisions,
    } for route, count in route_counts.sort_index().items())

    models = sorted(raw["model"].dropna().unique(), key=model_size)
    for model in models:
        model_rows = raw[raw["model"].astype(str) == str(model)]
        out_of_scope = model_rows[model_rows["no_reference_document"]]
        answerable = model_rows[~model_rows["no_reference_document"]]
        exact_refusals = int(out_of_scope["correct_refusal"].sum())
        false_refusals = int(answerable["refused"].sum())
        rows.extend([
            {
                "analysis": "abstention",
                "group": short_model(str(model)),
                "metric": "exact_refusal_rate",
                "aggregation": "proportion",
                "numerator": exact_refusals,
                "denominator": int(len(out_of_scope)),
                "value": float(exact_refusals) / len(out_of_scope),
            },
            {
                "analysis": "abstention",
                "group": short_model(str(model)),
                "metric": "false_refusal_rate",
                "aggregation": "proportion",
                "numerator": false_refusals,
                "denominator": int(len(answerable)),
                "value": float(false_refusals) / len(answerable),
            },
        ])

    result = pd.DataFrame(rows)
    result["numerator"] = result["numerator"].astype("Int64")
    return result


def validate_final_outputs(
    consolidated: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    operational_summary: pd.DataFrame,
) -> None:
    """Valida denominadores e agregações finais usadas no manuscrito."""
    for label, frame, expected_rows in (
        ("tabela consolidada", consolidated, EXPECTED_CONFIGURATIONS),
        ("tabela por estratégia", strategy_summary, len(STRATEGIES)),
        ("tabela operacional", operational_summary, 11),
    ):
        median_columns = [
            column for column in frame.columns if "median" in column.casefold()
        ]
        if len(frame) != expected_rows:
            raise ValueError(
                f"{label} deveria possuir {expected_rows} linhas. Possui {len(frame)}."
            )
        if label != "tabela operacional":
            if not median_columns or frame[median_columns].isna().any().any():
                raise ValueError(f"{label} contém medianas ausentes.")
            mean_columns = [
                column for column in frame.columns if "mean" in column.casefold()
            ]
            if not mean_columns or frame[mean_columns].isna().any().any():
                raise ValueError(f"{label} contém médias ausentes.")

    count_columns = ("n_quality", "n_retrieval", "n_latency")
    for column in count_columns:
        if set(consolidated[column]) != {EXPECTED_ROWS_PER_CONFIG}:
            raise ValueError(f"Denominador inválido em consolidated.{column}.")
        if set(strategy_summary[column]) != {EXPECTED_ROWS_PER_STRATEGY}:
            raise ValueError(f"Denominador inválido em strategy_summary.{column}.")

    if set(operational_summary["aggregation"]) != {"median", "proportion"}:
        raise ValueError("A tabela operacional possui agregações inesperadas.")
    if operational_summary["value"].isna().any():
        raise ValueError("A tabela operacional contém valores ausentes.")
    if (operational_summary["denominator"] <= 0).any():
        raise ValueError("A tabela operacional contém denominadores não positivos.")

    proportions = operational_summary[
        operational_summary["aggregation"] == "proportion"
    ]
    expected_values = (
        proportions["numerator"].astype(float)
        / proportions["denominator"].astype(float)
    )
    if not np.allclose(proportions["value"], expected_values, rtol=0.0, atol=1e-12):
        raise ValueError("As proporções operacionais não correspondem às contagens.")


def configure_style() -> None:
    plt.rcParams.update({
        "savefig.dpi": DPI,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.titlesize": 13,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig: Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_01_heatmap(consolidated: pd.DataFrame, path: Path) -> None:
    """Figura 1: Heatmap da mediana de correção por modelo × estratégia."""
    matrix = consolidated.pivot(
        index="strategy", columns="model_label", values="ac_median",
    )
    strategy_order = [s for s in STRATEGIES if s in matrix.index]
    model_order = sorted(matrix.columns, key=lambda m: model_size(m.lower()))
    matrix = matrix.reindex(strategy_order)[model_order]
    
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(
        matrix.to_numpy(dtype=float),
        cmap="YlOrRd", vmin=0, vmax=1, aspect="auto",
    )
    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns)
    ax.set_yticks(
        np.arange(matrix.shape[0]),
        [s.title() for s in matrix.index],
    )
    ax.set_xlabel("Modelo (escala de parâmetros)")
    ax.set_ylabel("Estratégia de recuperação")
    ax.set_title(
        "Mediana de Answer Correctness",
        fontweight="bold",
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            color = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                     fontsize=10, fontweight="bold", color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mediana de Answer Correctness (0–1)")
    fig.tight_layout()
    save_figure(fig, path)


def figure_03_latency_boxplot(latency: pd.DataFrame, path: Path) -> None:
    """Gera boxplots horizontais da latência por modelo e estratégia.

    Parameters
    ----------
    latency:
        Observações individuais contendo as colunas ``model``, ``strategy`` e
        ``total_ms``. Cada distribuição é representada por mediana, intervalo
        interquartil, extremos e valores atípicos segundo o boxplot de Tukey.
    path:
        Caminho de saída da figura em formato raster.

    Notes
    -----
    Os painéis são empilhados e usam escalas independentes de latência. Essa
    composição preserva rótulos horizontais legíveis quando a figura é reduzida
    à largura de uma página em documentos LaTeX.
    """
    models = [str(m) for m in latency["model"].dropna().unique()]
    strategy_labels = {
        "naive": "Naive",
        "hybrid": "Hybrid",
        "reranking": "Reranking",
        "hyde": "HyDE",
        "adaptive": "Adaptive",
    }
    fig, axes = plt.subplots(
        len(models),
        1,
        figsize=(9.0, 8.5),
        sharex=False,
        squeeze=False,
    )
    
    for ax, model in zip(axes[:, 0], models, strict=True):
        subset = latency[latency["model"].astype(str) == model]
        series = [
            subset.loc[
                subset["strategy"].astype(str) == s, "total_ms"
            ].dropna().to_numpy()
            for s in STRATEGIES
        ]
        boxes = ax.boxplot(
            series,
            tick_labels=[strategy_labels[s] for s in STRATEGIES],
            vert=False,
            patch_artist=True,
            widths=0.62,
            medianprops={"color": "#111827", "linewidth": 2.0},
            flierprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "#6B7280",
                "markersize": 3.5,
                "alpha": 0.7,
            },
        )
        for patch, s in zip(boxes["boxes"], STRATEGIES, strict=True):
            patch.set_facecolor(STRATEGY_COLORS[s])
            patch.set_alpha(0.75)
        ax.set_title(short_model(model), loc="left", fontweight="bold", pad=6)
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.set_xlabel("Latência total (ms)")
        ax.set_ylabel("Estratégia")
    
    fig.suptitle(
        "Distribuição da latência total por modelo e estratégia",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=1.2)
    save_figure(fig, path)


def figure_03_latency_boxplot_compact(latency: pd.DataFrame, path: Path) -> None:
    """Gera a versão compacta dos boxplots de latência.

    Parameters
    ----------
    latency:
        Observações individuais contendo as colunas ``model``, ``strategy`` e
        ``total_ms``. Os boxplots seguem a definição de Tukey.
    path:
        Caminho de saída da figura em formato raster.

    Notes
    -----
    Esta versão mantém os modelos lado a lado para comparação compacta. A
    versão principal empilhada deve ser preferida quando a legibilidade dos
    rótulos no documento LaTeX for prioritária.
    """
    models = [str(m) for m in latency["model"].dropna().unique()]
    strategy_labels = {
        "naive": "Naive",
        "hybrid": "Hybrid",
        "reranking": "Reranking",
        "hyde": "HyDE",
        "adaptive": "Adaptive",
    }
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(14.5, 5.5),
        sharey=False,
        squeeze=False,
    )

    for ax, model in zip(axes[0], models, strict=True):
        subset = latency[latency["model"].astype(str) == model]
        series = [
            subset.loc[
                subset["strategy"].astype(str) == strategy, "total_ms"
            ].dropna().to_numpy()
            for strategy in STRATEGIES
        ]
        boxes = ax.boxplot(
            series,
            tick_labels=[strategy_labels[strategy] for strategy in STRATEGIES],
            patch_artist=True,
            widths=0.55,
            medianprops={"color": "#111827", "linewidth": 2.0},
            flierprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "#6B7280",
                "markersize": 3.5,
                "alpha": 0.7,
            },
        )
        for patch, strategy in zip(boxes["boxes"], STRATEGIES, strict=True):
            patch.set_facecolor(STRATEGY_COLORS[strategy])
            patch.set_alpha(0.75)
        ax.set_title(short_model(model), fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_xlabel("Estratégia")

    axes[0, 0].set_ylabel("Latência total (ms)")
    fig.suptitle(
        "Distribuição da latência total por modelo e estratégia",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, path)


def figure_04_tradeoff(consolidated: pd.DataFrame, path: Path) -> None:
    """Figura 4: Dispersão qualidade × latência (trade-off)."""
    models = sorted(consolidated["model_label"].unique(), key=lambda m: model_size(m.lower()))
    markers = dict(zip(models, MODEL_MARKERS, strict=True))
    
    fig, ax = plt.subplots(figsize=(9, 6.5))
    
    for _, row in consolidated.iterrows():
        strategy = str(row["strategy"])
        model_label = str(row["model_label"])
        ax.scatter(
            row["lat_median_ms"],
            row["ac_median"],
            color=STRATEGY_COLORS[strategy],
            marker=markers[model_label],
            s=90,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
    
    strategy_handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", markersize=7,
            color=STRATEGY_COLORS[s], label=s.title(),
        )
        for s in STRATEGIES
    ]
    model_handles = [
        plt.Line2D(
            [], [], marker=markers[m], linestyle="", markersize=7,
            color="#374151", label=m,
        )
        for m in models
    ]
    legend1 = ax.legend(
        handles=strategy_handles, title="Estratégia",
        frameon=False, loc="upper left",
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=model_handles, title="Modelo",
        frameon=False, loc="lower right",
    )
    
    ax.set_xlabel("Mediana da latência total (ms)")
    ax.set_ylabel("Mediana de Answer Correctness")
    ax.set_ylim(0, 1)
    ax.set_title(
        "Trade-off entre Answer Correctness e latência total",
        fontweight="bold",
    )
    ax.grid(linestyle="--", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, path)



def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_root = (
        args.output_root
        or run_root / "final_results"
    ).resolve()
    
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    
    print("=" * 60)
    print("  GERACAO DOS RESULTADOS FINAIS PARA O ARTIGO")
    print("=" * 60)
    
    print("\n-- 1. Carregando dados dos JSONs originais --")
    raw = order_frame(load_raw(run_root / "raw"))
    ragas = order_frame(load_ragas(run_root / "ragas"))
    retrieval = order_frame(load_retrieval(run_root / "metrics" / "retrieval"))
    latency = order_frame(load_latency(run_root / "metrics" / "latency"))
    print(f"  - raw: {len(raw)} registros")
    print(f"  - ragas: {len(ragas)} registros")
    print(f"  - retrieval: {len(retrieval)} registros")
    print(f"  - latency: {len(latency)} registros")
    
    print("\n-- 2. Validando denominadores originais --")
    validate_denominators(raw, ragas, retrieval, latency)
    print("  - Denominadores originais OK.")
    
    print("\n-- 3. Filtrando queries sem documento de referência --")
    no_ref_qids = raw[raw["no_reference_document"]]["query_id"].unique()
    print(f"  -> {len(no_ref_qids)} query_ids sem referência: {sorted(no_ref_qids)}")
    
    ragas_f = ragas[~ragas["query_id"].isin(no_ref_qids)]
    retrieval_f = retrieval[~retrieval["query_id"].isin(no_ref_qids)]
    latency_f = latency[~latency["query_id"].isin(no_ref_qids)]
    
    print(f"  - Apos filtragem: ragas={len(ragas_f)}, "
          f"retrieval={len(retrieval_f)}, latency={len(latency_f)}")
    
    print("\n-- 4. Validando denominadores filtrados --")
    validate_filtered_counts(ragas_f, retrieval_f, latency_f)
    
    cross_validate_with_raw_json(run_root, ragas_f, latency_f)
    
    print("\n-- 6. Construindo tabelas e resumo operacional --")
    consolidated = build_consolidated_table(ragas_f, retrieval_f, latency_f, raw)
    strategy_summary = build_strategy_summary(ragas_f, retrieval_f, latency_f)
    operational_summary = build_operational_summary(raw, latency)
    validate_final_outputs(consolidated, strategy_summary, operational_summary)

    consolidated.to_csv(output_root / "tabela_consolidada.csv", index=False)
    strategy_summary.to_csv(
        output_root / "tabela_mediana_por_estrategia.csv", index=False
    )
    operational_summary.to_csv(
        output_root / "tabela_resumo_operacional.csv", index=False
    )
    print(
        f"  - Tabela consolidada: {len(consolidated)} linhas "
        f"({EXPECTED_CONFIGURATIONS} configuracoes)"
    )
    print(f"  - Tabela por estrategia: {len(strategy_summary)} linhas")
    print("  - Resumo operacional: roteamento e abstencao")
    
    print("\n-- 7. Gerando as 4 figuras finais --")
    configure_style()
    
    figure_01_heatmap(consolidated, output_root / "figura_01_heatmap_correcao.png")
    print("  - Figura 1: Heatmap da mediana de correcao")
    
    figure_03_latency_boxplot(latency_f, output_root / "figura_03_latencia_boxplot.png")
    print("  - Figura 2: Boxplots de latencia em paineis empilhados")

    figure_03_latency_boxplot_compact(
        latency_f,
        output_root / "figura_03_latencia_boxplot_compacta.png",
    )
    print("  - Figura 2 (compacta): Boxplots de latencia lado a lado")
    
    figure_04_tradeoff(consolidated, output_root / "figura_04_tradeoff.png")
    print("  - Figura 3: Trade-off qualidade x latencia")
    
    print("\n-- 8. Validacao final de artefatos --")
    csvs = list(output_root.glob("*.csv"))
    pngs = list(output_root.glob("*.png"))
    print(f"  -> CSV: {len(csvs)}, PNG: {len(pngs)}")
    
    if len(pngs) != 4:
        raise ValueError(f"Esperadas 4 figuras, encontradas {len(pngs)}")
    if len(csvs) != 3:
        raise ValueError(f"Esperadas 3 tabelas CSV, encontradas {len(csvs)}")
    
    print("  - Todos os artefatos validados.")
    
    print("\n" + "=" * 60)
    print(f"  RESULTADOS FINAIS GERADOS EM: {output_root}")
    print("  4 figuras PNG + 3 tabelas CSV")
    print("=" * 60)


if __name__ == "__main__":
    main()
