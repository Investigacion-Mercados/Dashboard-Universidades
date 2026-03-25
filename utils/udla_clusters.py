from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from scipy.cluster.vq import kmeans2

from utils.propensity_helpers import build_student_feature_base
from utils.quintile_ranges import asignar_quintil_por_rangos, calcular_rangos_quintiles

CLUSTER_NUMERIC_COLUMNS = [
    "edad_estudiante",
    "estudiante_quito",
    "primera_generacion",
    "quintil_ingreso_num",
    "quintil_deuda_num",
    "hijos_hogar_promedio",
]

CLUSTER_CATEGORICAL_COLUMNS = ["sexo_estudiante"]

PROFILE_LABELS = {
    "edad_estudiante": "Edad prom.",
    "pct_mujeres": "% mujeres",
    "pct_quito": "% Quito",
    "pct_primera_generacion": "% primera gen.",
    "quintil_ingreso_promedio": "Quintil ingreso prom.",
    "quintil_deuda_promedio": "Quintil deuda prom.",
    "hijos_promedio": "Hijos prom.",
}


def _clean_text(series: pd.Series, default: str = "Sin dato") -> pd.Series:
    return (
        pd.Series(series)
        .fillna("")
        .astype(str)
        .str.strip()
        .replace({"": default, "nan": default, "None": default})
    )


def _mode_or_default(series: pd.Series, default: str = "Sin dato") -> str:
    clean = _clean_text(series, default="")
    clean = clean[clean != ""]
    if clean.empty:
        return default
    return str(clean.mode().iloc[0])


def _share_between(
    series: pd.Series, lower: int | None = None, upper: int | None = None
) -> float:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.notna()
    if int(valid.sum()) == 0:
        return 0.0

    selected = valid.copy()
    if lower is not None:
        selected &= values >= lower
    if upper is not None:
        selected &= values <= upper
    return float(selected.sum() / valid.sum() * 100.0)


def _stable_seed(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


@st.cache_data(show_spinner=False)
def build_udla_cluster_base() -> dict[str, Any]:
    student_df, income_ranges = build_student_feature_base()
    udla = student_df[
        (student_df["Universidad"] == "UDLA")
        & (student_df["fuente_archivo"] == "UDLA")
    ].copy()

    if udla.empty:
        return {
            "students": udla,
            "income_ranges": {},
            "debt_ranges": {},
        }

    udla["tipo_estudiante"] = _clean_text(udla["tipo_estudiante"])
    udla["facultad"] = _clean_text(udla["unidad_academica"])
    udla["carrera"] = _clean_text(udla["carrera"])
    udla["sexo_estudiante"] = _clean_text(udla["sexo_estudiante"], default="DESCONOCIDO")
    udla["estado_hogar"] = _clean_text(udla["estado_hogar"], default="Desconocido")
    udla["edad_estudiante"] = pd.to_numeric(udla["edad_estudiante"], errors="coerce")
    udla["hijos_hogar"] = (
        pd.to_numeric(udla["hijos_hogar"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .round()
        .astype(int)
    )
    udla["hijos_hogar_promedio"] = (
        udla.groupby("hogar_id")["hijos_hogar"].transform("mean").fillna(0.0)
    )
    udla["primera_generacion"] = (
        pd.to_numeric(udla["primera_generacion"], errors="coerce")
        .fillna(0)
        .clip(lower=0, upper=1)
        .astype(int)
    )
    udla["estudiante_quito"] = (
        udla["IDENTIFICACION"].astype(str).str.strip().str.startswith("17").astype(int)
    )
    udla["quintil_ingreso_hogar"] = _clean_text(
        udla["quintil_institucion"], default="Sin empleo"
    )
    udla["quintil_ingreso_num"] = (
        pd.to_numeric(
            udla["quintil_ingreso_hogar"].replace({"Sin empleo": "0"}),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    debt_ranges = calcular_rangos_quintiles(udla["deuda_hogar"])
    udla["quintil_deuda_hogar"] = udla["deuda_hogar"].apply(
        lambda value: asignar_quintil_por_rangos(value, debt_ranges, vacio="Sin deuda")
    )
    udla["quintil_deuda_num"] = (
        pd.to_numeric(
            udla["quintil_deuda_hogar"].replace({"Sin deuda": "0"}),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    return {
        "students": udla,
        "income_ranges": income_ranges.get("UDLA", {}),
        "debt_ranges": debt_ranges,
    }


def _build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df[CLUSTER_NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.median()).fillna(0.0)

    categoricals = (
        df[CLUSTER_CATEGORICAL_COLUMNS]
        .fillna("DESCONOCIDO")
        .astype(str)
        .apply(lambda col: col.str.strip().replace("", "DESCONOCIDO"))
    )
    dummies = pd.get_dummies(
        categoricals, prefix=CLUSTER_CATEGORICAL_COLUMNS, dtype=float
    )
    return pd.concat([numeric, dummies], axis=1)


def _scale_cluster_matrix(feature_df: pd.DataFrame) -> np.ndarray:
    if feature_df.empty:
        return np.empty((0, 0), dtype=float)

    matrix = feature_df.to_numpy(dtype=float)
    mean = np.nanmean(matrix, axis=0)
    matrix = np.where(np.isnan(matrix), mean, matrix)
    std = np.nanstd(matrix, axis=0)
    std = np.where(std == 0, 1.0, std)
    scaled = (matrix - mean) / std

    columns = feature_df.columns.tolist()
    for cat_col in CLUSTER_CATEGORICAL_COLUMNS:
        prefix = f"{cat_col}_"
        dummy_indices = [i for i, c in enumerate(columns) if c.startswith(prefix)]
        if len(dummy_indices) > 1:
            scaled[:, dummy_indices] /= np.sqrt(len(dummy_indices))

    return scaled


def _run_kmeans_labels(scaled: np.ndarray, requested_k: int) -> np.ndarray:
    n_rows = scaled.shape[0]
    k = max(1, min(int(requested_k), n_rows))
    if k == 1:
        return np.ones(n_rows, dtype=int)

    seed = _stable_seed(f"udla-students-{requested_k}")
    try:
        _centroids, labels = kmeans2(scaled, k, minit="points", iter=50, seed=seed)
        labels = np.asarray(labels, dtype=int)
    except Exception:
        order = np.argsort(np.nan_to_num(scaled[:, 0], nan=0.0))
        labels = np.zeros(n_rows, dtype=int)
        for idx, chunk in enumerate(np.array_split(order, k)):
            labels[chunk] = idx

    return labels + 1


def _minimum_cluster_size(n_rows: int) -> int:
    if n_rows <= 8:
        return 1
    if n_rows <= 20:
        return 2
    if n_rows <= 60:
        return 3
    return max(4, int(np.floor(n_rows * 0.04)))


def _candidate_cluster_counts(
    n_rows: int, min_clusters: int = 2, max_clusters: int = 6
) -> list[int]:
    if n_rows <= 3:
        return [1]

    upper_bound = min(int(max_clusters), n_rows)
    if n_rows < 12:
        upper_bound = min(upper_bound, max(2, n_rows // 2))
    elif n_rows < 30:
        upper_bound = min(upper_bound, 4)

    lower_bound = min(int(min_clusters), upper_bound)
    if upper_bound < 2:
        return [1]
    return list(range(lower_bound, upper_bound + 1))


def _calinski_harabasz_score(matrix: np.ndarray, labels: np.ndarray) -> float:
    if matrix.size == 0 or len(labels) <= 2:
        return float("-inf")

    unique_labels = np.unique(labels)
    k = len(unique_labels)
    n_rows = matrix.shape[0]
    if k <= 1 or n_rows <= k:
        return float("-inf")

    overall_mean = matrix.mean(axis=0)
    within_dispersion = 0.0
    between_dispersion = 0.0

    for label in unique_labels:
        points = matrix[labels == label]
        if len(points) == 0:
            return float("-inf")
        centroid = points.mean(axis=0)
        within_dispersion += float(((points - centroid) ** 2).sum())
        between_dispersion += float(len(points) * ((centroid - overall_mean) ** 2).sum())

    if within_dispersion <= 1e-12:
        return float("inf") if between_dispersion > 0 else float("-inf")

    numerator = between_dispersion / max(k - 1, 1)
    denominator = within_dispersion / max(n_rows - k, 1)
    return float(numerator / max(denominator, 1e-12))


def _assign_clusters(
    feature_df: pd.DataFrame, min_clusters: int = 2, max_clusters: int = 6
) -> tuple[np.ndarray, int]:
    if feature_df.empty:
        return np.array([], dtype=int), 0

    scaled = _scale_cluster_matrix(feature_df)
    n_rows = scaled.shape[0]
    candidates = _candidate_cluster_counts(
        n_rows, min_clusters=min_clusters, max_clusters=max_clusters
    )
    if candidates == [1]:
        return np.ones(n_rows, dtype=int), 1

    min_size = _minimum_cluster_size(n_rows)
    evaluations: list[dict[str, Any]] = []
    for k in candidates:
        labels = _run_kmeans_labels(scaled, k)
        unique_labels, counts = np.unique(labels, return_counts=True)
        if len(unique_labels) <= 1:
            continue

        score = _calinski_harabasz_score(scaled, labels)
        evaluations.append(
            {
                "k": int(k),
                "labels": labels,
                "score": score,
                "is_valid": len(unique_labels) == k and int(counts.min()) >= min_size,
            }
        )

    if not evaluations:
        labels = _run_kmeans_labels(scaled, 1)
        return labels, 1

    valid_evaluations = [
        item for item in evaluations if item["is_valid"] and np.isfinite(item["score"])
    ]
    candidate_pool = valid_evaluations or [
        item for item in evaluations if np.isfinite(item["score"])
    ]
    if not candidate_pool:
        fallback = min(
            max(2, int(round(np.sqrt(max(n_rows, 1) / 2.0)))),
            max(candidates),
        )
        labels = _run_kmeans_labels(scaled, fallback)
        return labels, int(np.unique(labels).size)

    best_score = max(float(item["score"]) for item in candidate_pool)
    close_candidates = [
        item for item in candidate_pool if float(item["score"]) >= best_score * 0.97
    ]
    chosen = min(
        close_candidates,
        key=lambda item: (int(item["k"]), -float(item["score"])),
    )
    return np.asarray(chosen["labels"], dtype=int), int(chosen["k"])


def _label_clusters(summary: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, str]]:
    ordered = summary.sort_values(
        ["estudiantes", "quintil_ingreso_promedio", "quintil_deuda_promedio"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ordered["cluster_id_original"] = ordered["cluster_id"].astype(int)
    ordered["cluster"] = [f"Cluster {idx}" for idx in range(1, len(ordered) + 1)]
    label_map = dict(zip(ordered["cluster_id_original"], ordered["cluster"]))
    return ordered, label_map


def _cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("cluster_id", as_index=False).agg(
        estudiantes=("IDENTIFICACION", "nunique"),
        hogares=("hogar_id", "nunique"),
        edad_estudiante=("edad_estudiante", "mean"),
        pct_edad_15_19=("edad_estudiante", lambda s: _share_between(s, 15, 19)),
        pct_edad_20_22=("edad_estudiante", lambda s: _share_between(s, 20, 22)),
        pct_edad_23_25=("edad_estudiante", lambda s: _share_between(s, 23, 25)),
        pct_edad_mas_25=("edad_estudiante", lambda s: _share_between(s, 26, None)),
        pct_mujeres=(
            "sexo_estudiante",
            lambda s: float(
                pd.Series(s).fillna("").astype(str).str.upper().eq("MUJER").mean() * 100.0
            ),
        ),
        pct_quito=("estudiante_quito", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).mean() * 100.0)),
        pct_primera_generacion=(
            "primera_generacion",
            lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).mean() * 100.0),
        ),
        quintil_ingreso_promedio=("quintil_ingreso_num", "mean"),
        quintil_deuda_promedio=("quintil_deuda_num", "mean"),
        hijos_promedio=("hijos_hogar_promedio", "mean"),
    )
    debt_counts = (
        df[pd.to_numeric(df["deuda_hogar"], errors="coerce").fillna(0) > 0]
        .groupby("cluster_id", as_index=False)["hogar_id"]
        .nunique()
        .rename(columns={"hogar_id": "hogares_con_deuda"})
    )

    modes = df.groupby("cluster_id", as_index=False).agg(
        sexo_modal=("sexo_estudiante", _mode_or_default),
        estado_hogar_modal=("estado_hogar", _mode_or_default),
        facultad_modal=("facultad", _mode_or_default),
        carrera_modal=("carrera", _mode_or_default),
        tipo_modal=("tipo_estudiante", _mode_or_default),
        quintil_ingreso_modal=("quintil_ingreso_hogar", _mode_or_default),
        quintil_deuda_modal=("quintil_deuda_hogar", _mode_or_default),
    )

    summary = grouped.merge(debt_counts, on="cluster_id", how="left").merge(
        modes, on="cluster_id", how="left"
    )
    summary["hogares_con_deuda"] = (
        pd.to_numeric(summary["hogares_con_deuda"], errors="coerce").fillna(0).astype(int)
    )
    summary["tipo_estudiantes_pg"] = np.where(
        summary["pct_primera_generacion"] >= 50.0,
        "Primera generacion",
        "No primera generacion",
    )
    summary, label_map = _label_clusters(summary)
    summary = summary.drop(columns=["cluster_id"]).copy()
    ordered_columns = [
        "cluster",
        "estudiantes",
        "hogares",
        "hogares_con_deuda",
        "edad_estudiante",
        "pct_edad_15_19",
        "pct_edad_20_22",
        "pct_edad_23_25",
        "pct_edad_mas_25",
        "pct_mujeres",
        "pct_quito",
        "pct_primera_generacion",
        "quintil_ingreso_promedio",
        "quintil_deuda_promedio",
        "hijos_promedio",
        "sexo_modal",
        "estado_hogar_modal",
        "tipo_modal",
        "tipo_estudiantes_pg",
        "facultad_modal",
        "carrera_modal",
        "quintil_ingreso_modal",
        "quintil_deuda_modal",
        "cluster_id_original",
    ]
    return summary[ordered_columns], label_map


def _profile_frame(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "edad_estudiante",
        "pct_mujeres",
        "pct_quito",
        "pct_primera_generacion",
        "quintil_ingreso_promedio",
        "quintil_deuda_promedio",
        "hijos_promedio",
    ]
    profile = summary[["cluster"] + cols].copy()
    profile = profile.rename(columns=PROFILE_LABELS)
    return profile


@st.cache_data(show_spinner=False)
def run_udla_cluster_analysis(
    filtered_df: pd.DataFrame, min_clusters: int = 2, max_clusters: int = 6
) -> dict[str, Any]:
    if filtered_df.empty:
        return {
            "students": filtered_df.copy(),
            "summary": pd.DataFrame(),
            "profile": pd.DataFrame(),
            "feature_columns": [],
            "k": 0,
        }

    feature_df = _build_feature_frame(filtered_df)
    labels, chosen_k = _assign_clusters(
        feature_df,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
    )

    clustered = filtered_df.copy()
    clustered["cluster_id"] = labels
    summary, label_map = _cluster_summary(clustered)
    clustered["cluster"] = clustered["cluster_id"].map(label_map)

    display_columns = [
        "IDENTIFICACION",
        "hogar_id",
        "tipo_estudiante",
        "facultad",
        "carrera",
        "sexo_estudiante",
        "edad_estudiante",
        "estudiante_quito",
        "primera_generacion",
        "quintil_ingreso_hogar",
        "quintil_deuda_hogar",
        "hijos_hogar",
        "hijos_hogar_promedio",
        "cluster",
    ]
    available_display = [col for col in display_columns if col in clustered.columns]

    return {
        "students": clustered,
        "students_display": clustered[available_display].copy(),
        "summary": summary,
        "profile": _profile_frame(summary),
        "feature_columns": feature_df.columns.tolist(),
        "k": chosen_k,
    }
