"""
Funciones compartidas para las paginas de Comparacion UDLA.
Logica de calculo de perfiles, quintiles, vulnerabilidad, ubicacion, etc.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

import streamlit as st

# ─── Constantes ───────────────────────────────────────────────────────────────

QUINTILES = {
    1: {"min": 1.13, "max": 642.03},
    2: {"min": 642.04, "max": 909.07},
    3: {"min": 909.09, "max": 1415.89},
    4: {"min": 1415.92, "max": 2491.60},
    5: {"min": 2491.61, "max": 20009.99},
}

QUINTIL_LABELS = [
    "Sin informacion de empleo",
    "Quintil 1",
    "Quintil 2",
    "Quintil 3",
    "Quintil 4",
    "Quintil 5",
]

BASE_DIR = Path(__file__).resolve().parent.parent
GJSON_RURAL = BASE_DIR / "db" / "parroquiasRurales.geojson"
GJSON_URBANA = BASE_DIR / "db" / "parroquiasUrbanas.geojson"
GJSON_OTRAS = BASE_DIR / "db" / "otras.geojson"


# ─── Helpers de normalización ─────────────────────────────────────────────────


def norm_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace({"": "0", "nan": "0", "None": "0"})


def parse_valor_deuda(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    has_comma = s.str.contains(",", regex=False)
    s = s.where(
        ~has_comma,
        s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
    )
    return pd.to_numeric(s, errors="coerce").fillna(0)


def asignar_quintil(salario) -> int | None:
    if pd.isna(salario):
        return None
    try:
        salario_val = float(salario)
    except (TypeError, ValueError):
        return None
    for quintil, rango in QUINTILES.items():
        if rango["min"] <= salario_val <= rango["max"]:
            return quintil
    return None


# ─── Carga de datos geográficos ──────────────────────────────────────────────


def _obtener_nombre_parroquia(row: pd.Series) -> str:
    for campo in ["nombre", "DPA_DESPAR", "dpa_despar"]:
        valor = row.get(campo, None)
        if pd.notna(valor) and str(valor).strip() != "":
            return str(valor).strip()
    return "Sin nombre"


@st.cache_data(show_spinner=False)
def load_ubicacion_periodo() -> pd.DataFrame:
    path = Path("db/ubicacionEstudiantesPeriodo.csv")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.map(lambda c: str(c).strip().lower())
    return df


@st.cache_data(show_spinner=False)
def cargar_parroquias() -> gpd.GeoDataFrame:
    if not (GJSON_RURAL.exists() and GJSON_URBANA.exists() and GJSON_OTRAS.exists()):
        return gpd.GeoDataFrame(columns=["parroquia", "geometry"], geometry="geometry")

    gdf_rurales = gpd.read_file(GJSON_RURAL)
    gdf_urbanas = gpd.read_file(GJSON_URBANA)
    gdf_otras = gpd.read_file(GJSON_OTRAS)

    if gdf_rurales.crs != "EPSG:4326":
        gdf_rurales = gdf_rurales.to_crs("EPSG:4326")
    if gdf_urbanas.crs != "EPSG:4326":
        gdf_urbanas = gdf_urbanas.to_crs("EPSG:4326")
    if gdf_otras.crs != "EPSG:4326":
        gdf_otras = gdf_otras.to_crs("EPSG:4326")

    gdf = gpd.GeoDataFrame(
        pd.concat([gdf_rurales, gdf_urbanas, gdf_otras], ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )
    gdf["parroquia"] = gdf.apply(_obtener_nombre_parroquia, axis=1)
    return gdf[["parroquia", "geometry"]]


# ─── Construcción de familias / hogares ───────────────────────────────────────


def _build_hogar_id(df: pd.DataFrame, padre_col: str, madre_col: str) -> pd.Series:
    def _make_id(row):
        ids = []
        p = str(row[padre_col]).strip()
        m = str(row[madre_col]).strip()
        if p and p != "0":
            ids.append(p)
        if m and m != "0":
            ids.append(m)
        if not ids:
            return ""
        return "|".join(sorted(ids))

    return df.apply(_make_id, axis=1)


def build_familias(
    df_personas: pd.DataFrame,
    df_universo: pd.DataFrame,
    id_col: str,
    padre_col: str,
    madre_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    familias = df_personas.merge(df_universo, on=id_col, how="left")
    familias[padre_col] = norm_id(familias[padre_col])
    familias[madre_col] = norm_id(familias[madre_col])
    familias = familias[(familias[padre_col] != "0") | (familias[madre_col] != "0")]
    familias["hogar_id"] = _build_hogar_id(familias, padre_col, madre_col)
    familias = familias[familias["hogar_id"] != ""].copy()

    pares = []
    for _, r in familias.iterrows():
        if r[padre_col] != "0":
            pares.append((r["hogar_id"], r[padre_col]))
        if r[madre_col] != "0":
            pares.append((r["hogar_id"], r[madre_col]))
    df_mapa = pd.DataFrame(pares, columns=["hogar_id", "fam_id"]).drop_duplicates()
    return familias, df_mapa


def salario_por_id(df_ing: pd.DataFrame, id_col: str, salario_col: str) -> dict:
    if df_ing.empty or salario_col not in df_ing.columns:
        return {}
    tmp = df_ing.copy()
    tmp[salario_col] = pd.to_numeric(tmp[salario_col], errors="coerce").fillna(0)
    return tmp.groupby(id_col)[salario_col].sum().to_dict()


def deuda_por_id(df_deu: pd.DataFrame, id_col: str, valor_col: str) -> dict:
    if df_deu.empty or valor_col not in df_deu.columns:
        return {}
    tmp = df_deu.copy()
    tmp[valor_col] = pd.to_numeric(tmp[valor_col], errors="coerce").fillna(0)
    return tmp.groupby(id_col)[valor_col].sum().to_dict()


def hogares_salario_deuda(
    df_mapa: pd.DataFrame,
    salario_map: dict,
    deuda_map: dict,
) -> pd.DataFrame:
    hogares = pd.DataFrame({"hogar_id": df_mapa["hogar_id"].unique().tolist()})

    tmp_sal = df_mapa.copy()
    tmp_sal["salario"] = tmp_sal["fam_id"].map(salario_map).fillna(0)
    df_sal = tmp_sal.groupby("hogar_id", as_index=False)["salario"].sum()

    tmp_deu = df_mapa.copy()
    tmp_deu["deuda"] = tmp_deu["fam_id"].map(deuda_map).fillna(0)
    df_deu = tmp_deu.groupby("hogar_id", as_index=False)["deuda"].sum()

    hogares = hogares.merge(df_sal, on="hogar_id", how="left").merge(
        df_deu, on="hogar_id", how="left"
    )
    hogares["salario"] = hogares["salario"].fillna(0)
    hogares["deuda"] = hogares["deuda"].fillna(0)
    hogares["quintil"] = hogares["salario"].apply(asignar_quintil)
    hogares["grupo_quintil"] = hogares["quintil"].apply(
        lambda q: f"Quintil {int(q)}" if pd.notna(q) else "Sin informacion de empleo"
    )
    return hogares


def quintil_dist(hogares: pd.DataFrame) -> dict[str, float]:
    if hogares.empty or "grupo_quintil" not in hogares.columns:
        return {k: 0.0 for k in QUINTIL_LABELS}
    counts = hogares["grupo_quintil"].value_counts()
    total = float(counts.sum()) if counts.sum() > 0 else 1.0
    return {k: float(counts.get(k, 0)) / total for k in QUINTIL_LABELS}


def calcular_vulnerabilidad(
    familias: pd.DataFrame,
    id_col: str,
    padre_col: str,
    madre_col: str,
    ingresos_df: pd.DataFrame,
    ingresos_id_col: str,
    salario_col: str,
    deudas_df: pd.DataFrame,
    deudas_id_col: str,
    valor_col: str,
    calif_col: str,
) -> pd.DataFrame:
    if familias.empty:
        return pd.DataFrame()

    df = familias[[id_col, padre_col, madre_col]].copy()
    df[padre_col] = norm_id(df[padre_col])
    df[madre_col] = norm_id(df[madre_col])

    personas_con_ingresos: set[str] = set()
    ingreso_anual_por_id: dict[str, float] = {}
    if not ingresos_df.empty and salario_col in ingresos_df.columns:
        ing = ingresos_df.copy()
        ing[salario_col] = pd.to_numeric(ing[salario_col], errors="coerce").fillna(0)
        personas_con_ingresos = set(
            ing[ing[salario_col] > 0][ingresos_id_col].astype(str).unique().tolist()
        )
        ingreso_anual_por_id = (
            ing.groupby(ingresos_id_col)[salario_col].sum() * 14
        ).to_dict()

    deuda_total_por_id: dict[str, float] = {}
    deudores_criticos: set[str] = set()
    if not deudas_df.empty and valor_col in deudas_df.columns:
        deu = deudas_df.copy()
        deu[valor_col] = pd.to_numeric(deu[valor_col], errors="coerce").fillna(0)
        if calif_col in deu.columns:
            deu[calif_col] = deu[calif_col].astype(str).str.upper().str.strip()
            deu_crit = deu[deu[calif_col].isin(["D", "E"])].copy()
        else:
            deu_crit = deu.iloc[0:0].copy()
        if not deu_crit.empty:
            deuda_total_por_id = (
                deu_crit.groupby(deudas_id_col)[valor_col].sum().to_dict()
            )
            deudores_criticos = set(
                deu_crit[deudas_id_col].astype(str).unique().tolist()
            )

    df["vulnerable"] = False
    df["en_riesgo"] = False

    for idx, row in df.iterrows():
        contador = 0
        ced_padre = str(row[padre_col]).strip()
        ced_madre = str(row[madre_col]).strip()
        tiene_padre = ced_padre != "0"
        tiene_madre = ced_madre != "0"

        if not tiene_padre and not tiene_madre:
            df.loc[idx, ["vulnerable", "en_riesgo"]] = [True, False]
            continue

        padre_sin_empleo = tiene_padre and (ced_padre not in personas_con_ingresos)
        madre_sin_empleo = tiene_madre and (ced_madre not in personas_con_ingresos)

        if (
            (tiene_padre and tiene_madre and padre_sin_empleo and madre_sin_empleo)
            or (tiene_padre and not tiene_madre and padre_sin_empleo)
            or (not tiene_padre and tiene_madre and madre_sin_empleo)
        ):
            contador += 1

        if deuda_total_por_id:
            tiene_deuda_critica = False
            deuda_total = 0.0
            ingreso_anual = 0.0
            if tiene_padre:
                deuda_total += deuda_total_por_id.get(ced_padre, 0.0)
                ingreso_anual += ingreso_anual_por_id.get(ced_padre, 0.0)
                if ced_padre in deudores_criticos:
                    tiene_deuda_critica = True
            if tiene_madre:
                deuda_total += deuda_total_por_id.get(ced_madre, 0.0)
                ingreso_anual += ingreso_anual_por_id.get(ced_madre, 0.0)
                if ced_madre in deudores_criticos:
                    tiene_deuda_critica = True
            if tiene_deuda_critica and ingreso_anual > 0:
                if (deuda_total / ingreso_anual) >= 2.90:
                    contador += 1

        if contador >= 2:
            df.loc[idx, ["vulnerable", "en_riesgo"]] = [True, False]
        elif contador == 1:
            df.loc[idx, ["vulnerable", "en_riesgo"]] = [False, True]

    return df[[id_col, "vulnerable", "en_riesgo"]].copy()


def asignar_parroquia(
    df: pd.DataFrame, gdf_parroquias: gpd.GeoDataFrame, lat_col: str, lon_col: str
) -> pd.DataFrame:
    if df.empty or gdf_parroquias.empty:
        return pd.DataFrame(columns=list(df.columns) + ["parroquia"])

    tmp = df.copy()
    tmp[lat_col] = pd.to_numeric(tmp[lat_col], errors="coerce")
    tmp[lon_col] = pd.to_numeric(tmp[lon_col], errors="coerce")
    tmp = tmp.dropna(subset=[lat_col, lon_col])
    if tmp.empty:
        return pd.DataFrame(columns=list(df.columns) + ["parroquia"])

    gdf_points = gpd.GeoDataFrame(
        tmp,
        geometry=gpd.points_from_xy(tmp[lon_col], tmp[lat_col]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        gdf_points,
        gdf_parroquias[["parroquia", "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined.drop(columns=["index_right"], errors="ignore")
    joined = joined.drop(columns=["geometry"], errors="ignore")
    return pd.DataFrame(joined)


def parroquia_dist(series: pd.Series, categorias: list[str]) -> list[float]:
    if series is None or series.empty:
        return [0.0 for _ in categorias]
    counts = series.value_counts(normalize=True)
    dist = [float(counts.get(cat, 0.0)) for cat in categorias[:-1]]
    otros = max(0.0, 1.0 - sum(dist))
    dist.append(otros)
    return dist


def build_feature_vector(
    quintil_dist_vals: dict[str, float],
    deuda_avg: float,
    deuda_pct: float,
    vulnerable_pct: float,
    riesgo_pct: float,
    loc_dist: list[float] | None,
) -> dict[str, float]:
    features: dict[str, float] = {}
    for label in QUINTIL_LABELS:
        features[f"q_{label}"] = float(quintil_dist_vals.get(label, 0.0))
    features["deuda_avg"] = float(deuda_avg)
    features["deuda_pct"] = float(deuda_pct)
    features["vulnerable_pct"] = float(vulnerable_pct)
    features["riesgo_pct"] = float(riesgo_pct)
    if loc_dist is not None:
        for i, val in enumerate(loc_dist):
            features[f"loc_{i}"] = float(val)
    return features


# ─── Cálculo de similitud ────────────────────────────────────────────────────


def calcular_similitud(
    df_perfiles: pd.DataFrame,
    features_col: dict[str, float],
    feature_cols: list[str],
    weights: dict[str, float],
    categorias_sel: list[str],
    cat_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Calcula puntaje de similitud entre el colegio y cada grupo UDLA.
    El puntaje total es un promedio ponderado de las similitudes por categoria.
    """
    df_feat = df_perfiles.copy()
    for col in feature_cols:
        df_feat[col] = df_feat[col].astype(float)

    col_means = df_feat[feature_cols].mean()
    df_feat[feature_cols] = df_feat[feature_cols].fillna(col_means)

    x_school = np.array([features_col.get(c, np.nan) for c in feature_cols], dtype=float)
    x_school = np.where(np.isnan(x_school), col_means.to_numpy(dtype=float), x_school)

    X = df_feat[feature_cols].to_numpy(dtype=float)
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)

    Xz = (X - mu) / sigma
    xz = (x_school - mu) / sigma

    df_result = df_perfiles[["grupo", "total_estudiantes"]].copy()

    cat_cols = {
        "Quintiles": [c for c in feature_cols if c.startswith("q_")],
        "Deuda": [c for c in feature_cols if c in {"deuda_avg", "deuda_pct"}],
        "Vulnerabilidad": [
            c for c in feature_cols if c in {"vulnerable_pct", "riesgo_pct"}
        ],
        "Ubicacion (parroquia)": [c for c in feature_cols if c.startswith("loc_")],
    }

    def _similitud_por_columnas(cols: list[str]) -> np.ndarray:
        if not cols:
            return np.full((len(df_perfiles),), np.nan)
        idx = [feature_cols.index(c) for c in cols if c in feature_cols]
        if not idx:
            return np.full((len(df_perfiles),), np.nan)
        dist_c = np.linalg.norm(Xz[:, idx] - xz[idx], axis=1)
        return 100 / (1 + dist_c)

    if "Quintiles" in categorias_sel:
        df_result["sim_quintiles"] = np.round(
            _similitud_por_columnas(cat_cols["Quintiles"]), 4
        )
    if "Deuda" in categorias_sel:
        df_result["sim_deuda"] = np.round(_similitud_por_columnas(cat_cols["Deuda"]), 4)
    if "Vulnerabilidad" in categorias_sel:
        df_result["sim_vulnerabilidad"] = np.round(
            _similitud_por_columnas(cat_cols["Vulnerabilidad"]), 4
        )
    if "Ubicacion (parroquia)" in categorias_sel:
        df_result["sim_ubicacion"] = np.round(
            _similitud_por_columnas(cat_cols["Ubicacion (parroquia)"]), 4
        )

    # Pesos por categoria (si no se pasan, usar promedio simple)
    if not cat_weights:
        n = len(categorias_sel) if categorias_sel else 1
        cat_weights = {c: 1 / n for c in categorias_sel}

    # Reemplazar NaN por 0 en similitudes seleccionadas
    for col in ["sim_quintiles", "sim_deuda", "sim_vulnerabilidad", "sim_ubicacion"]:
        if col in df_result.columns:
            df_result[col] = df_result[col].fillna(0.0)

    # Total = suma ponderada de similitudes por categoria
    total = np.zeros(len(df_result), dtype=float)
    if "Quintiles" in categorias_sel and "sim_quintiles" in df_result.columns:
        total += df_result["sim_quintiles"].to_numpy() * float(
            cat_weights.get("Quintiles", 0.0)
        )
    if "Deuda" in categorias_sel and "sim_deuda" in df_result.columns:
        total += df_result["sim_deuda"].to_numpy() * float(
            cat_weights.get("Deuda", 0.0)
        )
    if (
        "Vulnerabilidad" in categorias_sel
        and "sim_vulnerabilidad" in df_result.columns
    ):
        total += df_result["sim_vulnerabilidad"].to_numpy() * float(
            cat_weights.get("Vulnerabilidad", 0.0)
        )
    if (
        "Ubicacion (parroquia)" in categorias_sel
        and "sim_ubicacion" in df_result.columns
    ):
        total += df_result["sim_ubicacion"].to_numpy() * float(
            cat_weights.get("Ubicacion (parroquia)", 0.0)
        )

    df_result["puntaje_similitud"] = np.round(total, 4)

    df_result = df_result.sort_values("puntaje_similitud", ascending=False).head(10)
    return df_result
