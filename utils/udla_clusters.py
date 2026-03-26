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


def _reference_debt_ranges(student_df: pd.DataFrame) -> dict[int, dict[str, float]]:
    udla_source = student_df[
        (student_df["Universidad"] == "UDLA")
        & (student_df["fuente_archivo"] == "UDLA")
    ].copy()
    return calcular_rangos_quintiles(udla_source["deuda_hogar"])


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


def _prepare_cluster_base(
    student_df: pd.DataFrame,
    income_ranges: dict[int, dict[str, float]],
    debt_ranges: dict[int, dict[str, float]],
) -> pd.DataFrame:
    base = student_df.copy()
    if base.empty:
        return base

    base["tipo_estudiante"] = _clean_text(base["tipo_estudiante"])
    base["facultad"] = _clean_text(base["unidad_academica"])
    base["carrera"] = _clean_text(base["carrera"])
    base["sexo_estudiante"] = _clean_text(
        base["sexo_estudiante"], default="DESCONOCIDO"
    )
    base["estado_hogar"] = _clean_text(base["estado_hogar"], default="Desconocido")
    base["edad_estudiante"] = pd.to_numeric(base["edad_estudiante"], errors="coerce")
    base["hijos_hogar"] = (
        pd.to_numeric(base["hijos_hogar"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .round()
        .astype(int)
    )
    base["hijos_hogar_promedio"] = (
        base.groupby("hogar_id")["hijos_hogar"].transform("mean").fillna(0.0)
    )
    base["primera_generacion"] = (
        pd.to_numeric(base["primera_generacion"], errors="coerce")
        .fillna(0)
        .clip(lower=0, upper=1)
        .astype(int)
    )
    base["estudiante_quito"] = (
        base["IDENTIFICACION"].astype(str).str.strip().str.startswith("17").astype(int)
    )
    base["quintil_ingreso_hogar"] = base["salario_hogar"].apply(
        lambda value: asignar_quintil_por_rangos(value, income_ranges, vacio="Sin empleo")
    )
    base["quintil_ingreso_num"] = (
        pd.to_numeric(
            base["quintil_ingreso_hogar"].replace({"Sin empleo": "0"}),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    base["quintil_deuda_hogar"] = base["deuda_hogar"].apply(
        lambda value: asignar_quintil_por_rangos(value, debt_ranges, vacio="Sin deuda")
    )
    base["quintil_deuda_num"] = (
        pd.to_numeric(
            base["quintil_deuda_hogar"].replace({"Sin deuda": "0"}),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    return base


@st.cache_data(show_spinner=False)
def build_udla_cluster_base() -> dict[str, Any]:
    student_df, income_ranges = build_student_feature_base()
    udla_raw = student_df[
        (student_df["Universidad"] == "UDLA")
        & (student_df["fuente_archivo"] == "UDLA")
    ].copy()
    debt_ranges = _reference_debt_ranges(student_df)
    udla = _prepare_cluster_base(udla_raw, income_ranges.get("UDLA", {}), debt_ranges)

    if udla.empty:
        return {
            "students": udla,
            "income_ranges": {},
            "debt_ranges": {},
        }

    return {
        "students": udla,
        "income_ranges": income_ranges.get("UDLA", {}),
        "debt_ranges": debt_ranges,
    }


@st.cache_data(show_spinner=False)
def build_university_cluster_base() -> dict[str, Any]:
    student_df, income_ranges = build_student_feature_base()
    debt_ranges = _reference_debt_ranges(student_df)
    all_students = _prepare_cluster_base(
        student_df, income_ranges.get("UDLA", {}), debt_ranges
    )
    return {
        "students": all_students,
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


def _fit_feature_template(df: pd.DataFrame) -> dict[str, Any]:
    numeric = df[CLUSTER_NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    numeric_fill = numeric.median().fillna(0.0).to_dict()
    categorical_levels = {
        col: sorted(
            (
                df[col]
                .fillna("DESCONOCIDO")
                .astype(str)
                .str.strip()
                .replace("", "DESCONOCIDO")
            )
            .unique()
            .tolist()
        )
        for col in CLUSTER_CATEGORICAL_COLUMNS
    }
    return {
        "numeric_fill": numeric_fill,
        "categorical_levels": categorical_levels,
    }


def _build_feature_frame_from_template(
    df: pd.DataFrame, template: dict[str, Any]
) -> pd.DataFrame:
    numeric = df[CLUSTER_NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    for col in CLUSTER_NUMERIC_COLUMNS:
        numeric[col] = numeric[col].fillna(float(template["numeric_fill"].get(col, 0.0)))
    numeric = numeric.fillna(0.0)

    categoricals = (
        df[CLUSTER_CATEGORICAL_COLUMNS]
        .fillna("DESCONOCIDO")
        .astype(str)
        .apply(lambda col: col.str.strip().replace("", "DESCONOCIDO"))
    )
    dummies = pd.get_dummies(
        categoricals, prefix=CLUSTER_CATEGORICAL_COLUMNS, dtype=float
    )
    expected_dummy_cols: list[str] = []
    for cat_col in CLUSTER_CATEGORICAL_COLUMNS:
        expected_dummy_cols.extend(
            [
                f"{cat_col}_{value}"
                for value in template["categorical_levels"].get(cat_col, [])
            ]
        )
    dummies = dummies.reindex(columns=expected_dummy_cols, fill_value=0.0)
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


def _fit_scaler(feature_df: pd.DataFrame) -> dict[str, Any]:
    if feature_df.empty:
        return {"columns": [], "mean": np.array([]), "std": np.array([]), "weights": np.array([])}

    matrix = feature_df.to_numpy(dtype=float)
    mean = np.nanmean(matrix, axis=0)
    matrix = np.where(np.isnan(matrix), mean, matrix)
    std = np.nanstd(matrix, axis=0)
    std = np.where(std == 0, 1.0, std)
    columns = feature_df.columns.tolist()
    weights = np.ones(len(columns), dtype=float)

    for cat_col in CLUSTER_CATEGORICAL_COLUMNS:
        prefix = f"{cat_col}_"
        dummy_indices = [i for i, c in enumerate(columns) if c.startswith(prefix)]
        if len(dummy_indices) > 1:
            weights[dummy_indices] = 1.0 / np.sqrt(len(dummy_indices))

    return {"columns": columns, "mean": mean, "std": std, "weights": weights}


def _transform_scaled(feature_df: pd.DataFrame, scaler: dict[str, Any]) -> np.ndarray:
    columns = scaler.get("columns", [])
    if not columns:
        return np.empty((len(feature_df), 0), dtype=float)

    matrix = feature_df[columns].to_numpy(dtype=float)
    mean = np.asarray(scaler["mean"], dtype=float)
    std = np.asarray(scaler["std"], dtype=float)
    weights = np.asarray(scaler["weights"], dtype=float)
    matrix = np.where(np.isnan(matrix), mean, matrix)
    scaled = (matrix - mean) / std
    scaled = scaled * weights
    return scaled


def _run_kmeans_labels(
    scaled: np.ndarray, requested_k: int, *, seed_token: str | None = None
) -> np.ndarray:
    n_rows = scaled.shape[0]
    k = max(1, min(int(requested_k), n_rows))
    if k == 1:
        return np.ones(n_rows, dtype=int)

    seed = _stable_seed(seed_token or f"udla-students-{requested_k}")
    try:
        _centroids, labels = kmeans2(scaled, k, minit="points", iter=50, seed=seed)
        labels = np.asarray(labels, dtype=int)
    except Exception:
        order = np.argsort(np.nan_to_num(scaled[:, 0], nan=0.0))
        labels = np.zeros(n_rows, dtype=int)
        for idx, chunk in enumerate(np.array_split(order, k)):
            labels[chunk] = idx

    return labels + 1


def _income_mode_diversity(labels: np.ndarray, income_series: pd.Series) -> int:
    if len(labels) == 0:
        return 0

    incomes = pd.to_numeric(income_series, errors="coerce").fillna(-1).astype(int)
    frame = pd.DataFrame({"cluster_id": labels, "income": incomes})
    if frame.empty:
        return 0

    modes = frame.groupby("cluster_id")["income"].agg(
        lambda s: int(pd.Series(s).mode().iloc[0]) if not pd.Series(s).mode().empty else -1
    )
    valid_modes = modes[modes >= 0]
    return int(valid_modes.nunique())


def _income_anchor_labels(
    income_series: pd.Series, requested_k: int, *, min_size: int
) -> np.ndarray | None:
    values = pd.to_numeric(income_series, errors="coerce").fillna(-1).astype(int)
    clean_values = values[values >= 0]
    if clean_values.empty:
        return None

    unique_income = sorted(clean_values.unique().tolist())
    k = int(requested_k)
    if k <= 1 or len(unique_income) < k:
        return None

    anchor_idx = np.linspace(0, len(unique_income) - 1, k).round().astype(int).tolist()
    anchors = [int(unique_income[idx]) for idx in anchor_idx]
    if len(set(anchors)) < k:
        return None

    income_np = values.to_numpy(dtype=float)
    dist = np.abs(income_np[:, None] - np.asarray(anchors, dtype=float)[None, :])
    labels = np.argmin(dist, axis=1).astype(int) + 1

    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) != k or int(counts.min()) < int(min_size):
        return None
    return labels


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
    feature_df: pd.DataFrame,
    min_clusters: int = 2,
    max_clusters: int = 6,
    *,
    income_series: pd.Series | None = None,
    prefer_distinct_income_modal: bool = False,
    cluster_trials: int = 1,
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
        n_trials = max(1, int(cluster_trials))
        for trial_idx in range(n_trials):
            seed_token = f"udla-students-{k}-trial-{trial_idx}"
            labels = _run_kmeans_labels(scaled, k, seed_token=seed_token)
            unique_labels, counts = np.unique(labels, return_counts=True)
            if len(unique_labels) <= 1:
                continue

            score = _calinski_harabasz_score(scaled, labels)
            income_mode_count = (
                _income_mode_diversity(labels, income_series)
                if income_series is not None
                else 0
            )
            evaluations.append(
                {
                    "k": int(k),
                    "labels": labels,
                    "score": score,
                    "is_valid": len(unique_labels) == k and int(counts.min()) >= min_size,
                    "min_count": int(counts.min()),
                    "income_mode_count": int(income_mode_count),
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
        labels = _run_kmeans_labels(scaled, fallback, seed_token=f"udla-students-{fallback}-fallback")
        return labels, int(np.unique(labels).size)

    if prefer_distinct_income_modal and income_series is not None:
        max_income_modes = max(
            int(item.get("income_mode_count", 0)) for item in candidate_pool
        )
        mode_candidates = [
            item
            for item in candidate_pool
            if int(item.get("income_mode_count", 0)) == max_income_modes
        ]
        if mode_candidates:
            best_mode_score = max(float(item["score"]) for item in mode_candidates)
            close_mode_candidates = [
                item
                for item in mode_candidates
                if float(item["score"]) >= best_mode_score * 0.97
            ]
            chosen_income = max(
                close_mode_candidates,
                key=lambda item: (int(item.get("min_count", 0)), float(item["score"]), -int(item["k"])),
            )
            return np.asarray(chosen_income["labels"], dtype=int), int(chosen_income["k"])

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


def _nearest_cluster_assignment(
    scaled_matrix: np.ndarray,
    centroids_df: pd.DataFrame,
    radius_by_cluster: dict[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    if scaled_matrix.size == 0 or centroids_df.empty:
        return np.array([], dtype=int), np.array([], dtype=float)

    centroid_ids = centroids_df.index.to_numpy(dtype=int)
    centroid_matrix = centroids_df.to_numpy(dtype=float)
    diffs = scaled_matrix[:, None, :] - centroid_matrix[None, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    nearest_idx = np.argmin(distances, axis=1)
    nearest_cluster_ids = centroid_ids[nearest_idx]
    nearest_distances = distances[np.arange(len(scaled_matrix)), nearest_idx]
    assigned = np.array(
        [
            cluster_id
            if float(distance) <= float(radius_by_cluster.get(int(cluster_id), np.inf))
            else 0
            for cluster_id, distance in zip(nearest_cluster_ids, nearest_distances)
        ],
        dtype=int,
    )
    return assigned, nearest_distances


@st.cache_data(show_spinner=False)
def run_udla_cluster_analysis(
    filtered_df: pd.DataFrame,
    min_clusters: int = 2,
    max_clusters: int = 6,
    *,
    prefer_distinct_income_modal: bool = False,
    cluster_trials: int = 1,
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
        income_series=filtered_df.get("quintil_ingreso_num", pd.Series(dtype="float")),
        prefer_distinct_income_modal=prefer_distinct_income_modal,
        cluster_trials=cluster_trials,
    )
    if (
        prefer_distinct_income_modal
        and int(min_clusters) == int(max_clusters)
        and int(chosen_k) == int(min_clusters)
        and int(chosen_k) > 1
    ):
        income_series = filtered_df.get("quintil_ingreso_num", pd.Series(dtype="float"))
        current_income_modes = _income_mode_diversity(labels, income_series)
        target_income_modes = min(
            int(chosen_k),
            int(pd.to_numeric(income_series, errors="coerce").dropna().astype(int).nunique()),
        )
        if current_income_modes < target_income_modes:
            fallback_labels = _income_anchor_labels(
                income_series,
                int(chosen_k),
                min_size=_minimum_cluster_size(len(filtered_df)),
            )
            if fallback_labels is not None:
                fallback_income_modes = _income_mode_diversity(fallback_labels, income_series)
                if fallback_income_modes > current_income_modes:
                    labels = np.asarray(fallback_labels, dtype=int)

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


@st.cache_data(show_spinner=False)
def run_university_cluster_projection(
    udla_reference_df: pd.DataFrame,
    candidate_df: pd.DataFrame | None = None,
    min_clusters: int = 2,
    max_clusters: int = 6,
    *,
    prefer_distinct_income_modal: bool = False,
    cluster_trials: int = 1,
) -> dict[str, Any]:
    if udla_reference_df.empty:
        return {
            "students": (candidate_df.copy() if candidate_df is not None else udla_reference_df.copy()),
            "udla_students": pd.DataFrame(),
            "udla_summary": pd.DataFrame(),
            "cluster_tables": {},
            "cluster_order": [],
            "k": 0,
            "other_count": 0,
        }

    candidates = candidate_df.copy() if candidate_df is not None else udla_reference_df.copy()
    if candidates.empty:
        return {
            "students": candidates,
            "udla_students": pd.DataFrame(),
            "udla_summary": pd.DataFrame(),
            "cluster_tables": {},
            "cluster_order": [],
            "k": 0,
            "other_count": 0,
        }

    udla_df = udla_reference_df[
        (udla_reference_df["Universidad"] == "UDLA")
        & (udla_reference_df["fuente_archivo"] == "UDLA")
    ].copy()
    if udla_df.empty:
        return {
            "students": candidates,
            "udla_students": udla_df,
            "udla_summary": pd.DataFrame(),
            "cluster_tables": {},
            "cluster_order": [],
            "k": 0,
            "other_count": 0,
        }
    template = _fit_feature_template(udla_df)
    udla_feature_df = _build_feature_frame_from_template(udla_df, template)
    udla_labels, chosen_k = _assign_clusters(
        udla_feature_df,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        income_series=udla_df.get("quintil_ingreso_num", pd.Series(dtype="float")),
        prefer_distinct_income_modal=prefer_distinct_income_modal,
        cluster_trials=cluster_trials,
    )
    if (
        prefer_distinct_income_modal
        and int(min_clusters) == int(max_clusters)
        and int(chosen_k) == int(min_clusters)
        and int(chosen_k) > 1
    ):
        income_series = udla_df.get("quintil_ingreso_num", pd.Series(dtype="float"))
        current_income_modes = _income_mode_diversity(udla_labels, income_series)
        target_income_modes = min(
            int(chosen_k),
            int(pd.to_numeric(income_series, errors="coerce").dropna().astype(int).nunique()),
        )
        if current_income_modes < target_income_modes:
            fallback_labels = _income_anchor_labels(
                income_series,
                int(chosen_k),
                min_size=_minimum_cluster_size(len(udla_df)),
            )
            if fallback_labels is not None:
                fallback_income_modes = _income_mode_diversity(fallback_labels, income_series)
                if fallback_income_modes > current_income_modes:
                    udla_labels = np.asarray(fallback_labels, dtype=int)

    scaler = _fit_scaler(udla_feature_df)
    udla_scaled = _transform_scaled(udla_feature_df, scaler)
    udla_clustered = udla_df.copy()
    udla_clustered["cluster_id"] = udla_labels
    udla_summary, label_map = _cluster_summary(udla_clustered)
    udla_clustered["cluster"] = udla_clustered["cluster_id"].map(label_map)

    udla_scaled_df = pd.DataFrame(
        udla_scaled, columns=udla_feature_df.columns.tolist(), index=udla_df.index
    )
    centroids_df = (
        udla_scaled_df.assign(cluster_id=udla_labels)
        .groupby("cluster_id", as_index=True)[udla_feature_df.columns.tolist()]
        .mean()
    )

    radius_by_cluster: dict[int, float] = {}
    for cluster_id in sorted(np.unique(udla_labels).tolist()):
        mask = udla_clustered["cluster_id"] == int(cluster_id)
        points = udla_scaled_df.loc[mask]
        centroid = centroids_df.loc[int(cluster_id)].to_numpy(dtype=float)
        distances = np.linalg.norm(points.to_numpy(dtype=float) - centroid, axis=1)
        radius_by_cluster[int(cluster_id)] = float(distances.max()) * 1.001 + 1e-9

    all_feature_df = _build_feature_frame_from_template(candidates, template)
    all_scaled = _transform_scaled(all_feature_df, scaler)
    assigned_ids, assigned_distances = _nearest_cluster_assignment(
        all_scaled, centroids_df, radius_by_cluster
    )

    projected = candidates.copy()
    projected["cluster_id_udla"] = assigned_ids
    projected["distancia_cluster_udla"] = assigned_distances
    projected["cluster_udla"] = projected["cluster_id_udla"].map(label_map)
    projected.loc[projected["cluster_id_udla"] == 0, "cluster_udla"] = "Otro cluster"

    cluster_order = udla_summary["cluster"].tolist()
    if (projected["cluster_udla"] == "Otro cluster").any():
        cluster_order.append("Otro cluster")

    counts = (
        projected.groupby(["cluster_udla", "Universidad"], as_index=False)["IDENTIFICACION"]
        .nunique()
        .rename(columns={"cluster_udla": "Cluster", "Universidad": "Universidad", "IDENTIFICACION": "Cantidad de personas"})
    )

    universities = sorted(projected["Universidad"].dropna().astype(str).str.strip().unique().tolist())
    cluster_tables: dict[str, pd.DataFrame] = {}
    for cluster_name in cluster_order:
        base_table = pd.DataFrame(
            {"Cluster": [cluster_name] * len(universities), "Universidad": universities}
        )
        table = base_table.merge(
            counts[counts["Cluster"] == cluster_name],
            on=["Cluster", "Universidad"],
            how="left",
        )
        table["Cantidad de personas"] = (
            pd.to_numeric(table["Cantidad de personas"], errors="coerce").fillna(0).astype(int)
        )
        cluster_tables[cluster_name] = table.sort_values(
            ["Cantidad de personas", "Universidad"],
            ascending=[False, True],
        ).reset_index(drop=True)

    return {
        "students": projected,
        "udla_students": udla_clustered,
        "udla_summary": udla_summary,
        "cluster_tables": cluster_tables,
        "cluster_order": cluster_order,
        "k": chosen_k,
        "other_count": int((projected["cluster_udla"] == "Otro cluster").sum()),
        "radius_by_cluster": radius_by_cluster,
    }
