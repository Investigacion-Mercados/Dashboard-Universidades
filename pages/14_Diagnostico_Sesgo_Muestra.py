import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

from utils.excel_loader import load_excel_sheet

try:
    from scipy.stats import chi2 as chi2_dist
except Exception:  # pragma: no cover
    chi2_dist = None

try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:  # pragma: no cover
    gpd = None  # type: ignore[assignment]
    Point = None  # type: ignore[assignment]


st.set_page_config(page_title="Diagnostico de sesgo", layout="wide")
st.title("🔍 Diagnóstico de Sesgo de Selección")
st.markdown(
    """
<div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
    <p style='margin: 0; color: #262730;'>
    <strong>Análisis exploratorio</strong> con la data disponible: estudiantes, familiares, 
    información personal, empleos y deudas. Este módulo detecta indicios de sesgo en la muestra.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

UNIVERSO_TOTAL_ESTUDIANTES = 1221
MARGEN_ERROR_OBJETIVO = 0.05
UNIVERSO_POR_SEDE = {
    "Calderon": 563,
    "Quitumbe": 658,
}
QUINTILES_INGRESO = {
    1: {"min": 1.13, "max": 642.03},
    2: {"min": 642.04, "max": 909.07},
    3: {"min": 909.09, "max": 1415.89},
    4: {"min": 1415.92, "max": 2491.60},
    5: {"min": 2491.61, "max": 20009.99},
}
SIN_EMPLEO_FORMAL_LABEL = "Sin Empleo Formal"
QUINTIL_ORDER = [SIN_EMPLEO_FORMAL_LABEL] + [f"Quintil {i}" for i in range(1, 6)]
OUTLIER_RATE_ALERT = 0.15
BASE_DIR = Path(__file__).resolve().parent.parent
GJSON_RURAL = BASE_DIR / "db" / "parroquiasRurales.geojson"
GJSON_URBANA = BASE_DIR / "db" / "parroquiasUrbanas.geojson"
GJSON_OTRAS = BASE_DIR / "db" / "otras.geojson"


@st.cache_data(show_spinner=False)
def load_data() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    estudiantes = load_excel_sheet("Estudiantes")
    universo_familiares = load_excel_sheet("Universo Familiares")
    info_personal = load_excel_sheet("Informacion Personal")
    empleos = load_excel_sheet("Empleos")
    deudas = load_excel_sheet("Deudas")

    if "Cedula" in estudiantes.columns:
        estudiantes = estudiantes.rename(columns={"Cedula": "IDENTIFICACION"})
    elif "CEDULA" in estudiantes.columns:
        estudiantes = estudiantes.rename(columns={"CEDULA": "IDENTIFICACION"})

    return estudiantes, universo_familiares, info_personal, empleos, deudas


def _to_int_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna().astype(int)


def _max_consecutive_run(sorted_ids: list[int]) -> int:
    if not sorted_ids:
        return 0
    max_run = 1
    run = 1
    for i in range(1, len(sorted_ids)):
        if sorted_ids[i] == sorted_ids[i - 1] + 1:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    return max_run


def _hh_index(series: pd.Series) -> float:
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        return 0.0
    p = s.value_counts(normalize=True)
    return float((p**2).sum())


def _top_category_share(series: pd.Series) -> tuple[str, float]:
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        return ("Sin dato", 0.0)
    p = s.value_counts(normalize=True)
    return (str(p.index[0]), float(p.iloc[0]))


def _build_concentration_table(
    df: pd.DataFrame, variables: list[str], grupo: str
) -> pd.DataFrame:
    rows: list[dict[str, str | float]] = []
    for var in variables:
        if var not in df.columns:
            continue
        top_cat, top_share = _top_category_share(df[var])
        hhi = _hh_index(df[var])
        rows.append(
            {
                "Grupo": grupo,
                "Variable": var,
                "Categoria dominante": top_cat,
                "Porcentaje dominante (%)": round(top_share * 100, 2),
                "HHI": round(hhi, 4),
                "Bandera concentracion": (
                    "Si" if (top_share >= 0.7 or hhi >= 0.35) else "No"
                ),
            }
        )
    return pd.DataFrame(rows)


def _clean_series_for_coverage(series: pd.Series) -> pd.Series:
    s = series.copy()
    if s.dtype == "O":
        s = s.astype(str).str.strip()
        s_lower = s.str.lower()
        missing_tokens = {"", "nan", "none", "null", "na", "n/a", "sin dato"}
        s = s.mask(s_lower.isin(missing_tokens), np.nan)
    return s


def _parse_valor_deuda(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    has_comma = s.str.contains(",", regex=False)
    s = s.where(
        ~has_comma,
        s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
    )
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def _obtener_nombre_parroquia(row: pd.Series) -> str:
    for campo in ["nombre", "DPA_DESPAR", "dpa_despar"]:
        valor = row.get(campo, None)
        if pd.notna(valor) and str(valor).strip() != "":
            return str(valor).strip()
    return "Sin nombre"


@st.cache_data(show_spinner=False)
def _load_parroquias_geojson() -> pd.DataFrame:
    if gpd is None:
        return pd.DataFrame(columns=["PARROQUIA", "geometry"])

    frames: list[pd.DataFrame] = []
    for path in [GJSON_RURAL, GJSON_URBANA, GJSON_OTRAS]:
        if not path.exists():
            continue
        gdf = gpd.read_file(path)
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        gdf = gdf.copy()
        gdf["PARROQUIA"] = gdf.apply(_obtener_nombre_parroquia, axis=1)
        frames.append(gdf[["PARROQUIA", "geometry"]])

    if not frames:
        return pd.DataFrame(columns=["PARROQUIA", "geometry"])

    return pd.concat(frames, ignore_index=True)


def _build_income_latest(empleos: pd.DataFrame) -> pd.Series:
    if empleos.empty:
        return pd.Series(dtype=float)
    df = empleos.copy()
    df["IDENTIFICACION"] = pd.to_numeric(df["IDENTIFICACION"], errors="coerce")
    df["ANIO"] = pd.to_numeric(df["ANIO"], errors="coerce")
    df["MES"] = pd.to_numeric(df["MES"], errors="coerce")
    df["SALARIO"] = pd.to_numeric(df["SALARIO"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["IDENTIFICACION", "ANIO", "MES"]).copy()
    if df.empty:
        return pd.Series(dtype=float)
    df["IDENTIFICACION"] = df["IDENTIFICACION"].astype(int)
    df["PERIODO"] = df["ANIO"].astype(int) * 100 + df["MES"].astype(int)
    max_periodo = df.groupby("IDENTIFICACION")["PERIODO"].transform("max")
    ult = df[df["PERIODO"] == max_periodo].copy()
    return ult.groupby("IDENTIFICACION")["SALARIO"].sum()


def _build_debt_latest(deudas: pd.DataFrame) -> pd.Series:
    if deudas.empty:
        return pd.Series(dtype=float)
    df = deudas.copy()
    df["IDENTIFICACION"] = pd.to_numeric(df["IDENTIFICACION"], errors="coerce")
    df["ANIO"] = pd.to_numeric(df["ANIO"], errors="coerce")
    df["MES"] = pd.to_numeric(df["MES"], errors="coerce")
    df["VALOR"] = _parse_valor_deuda(df["VALOR"])
    df = df.dropna(subset=["IDENTIFICACION", "ANIO", "MES"]).copy()
    if df.empty:
        return pd.Series(dtype=float)
    df["IDENTIFICACION"] = df["IDENTIFICACION"].astype(int)
    df["PERIODO"] = df["ANIO"].astype(int) * 100 + df["MES"].astype(int)
    max_periodo = df.groupby("IDENTIFICACION")["PERIODO"].transform("max")
    ult = df[df["PERIODO"] == max_periodo].copy()
    return ult.groupby("IDENTIFICACION")["VALOR"].sum()


def _sample_size_consistency(
    n_muestra: int,
    n_universo: int,
    confianza_z: float = 1.96,
    p: float = 0.5,
    error_objetivo: float = 0.05,
) -> tuple[float, float]:
    if n_muestra <= 0 or n_universo <= 1:
        return (np.nan, np.nan)
    q = 1.0 - p
    moe_inf = confianza_z * np.sqrt((p * q) / n_muestra)
    fpc = (
        np.sqrt((n_universo - n_muestra) / (n_universo - 1))
        if n_universo > n_muestra
        else 0.0
    )
    margen_error = float(moe_inf * fpc)
    n_requerida = _required_sample_size(
        n_universo=n_universo,
        confianza_z=confianza_z,
        p=p,
        error_objetivo=error_objetivo,
    )
    return margen_error, n_requerida


def _required_sample_size(
    n_universo: int,
    confianza_z: float = 1.96,
    p: float = 0.5,
    error_objetivo: float = 0.05,
) -> float:
    if n_universo <= 1:
        return np.nan
    q = 1.0 - p
    n0 = (confianza_z**2 * p * q) / (error_objetivo**2)
    return float((n_universo * n0) / (n_universo + n0 - 1))


def _assign_quintil_ingreso(valor: float) -> int | None:
    if pd.isna(valor):
        return None
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return None
    for quintil, rango in QUINTILES_INGRESO.items():
        if rango["min"] <= v <= rango["max"]:
            return quintil
    return None


def _build_hogares_quintil(
    links: pd.DataFrame,
    income_by_id: pd.Series,
    debt_by_id: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for _, row in links.iterrows():
        padre = int(row.get("CED_PADRE", 0))
        madre = int(row.get("CED_MADRE", 0))
        fam_ids = sorted({fid for fid in [padre, madre] if fid > 0})
        if not fam_ids:
            continue
        hogar_id = "|".join(str(fid) for fid in fam_ids)
        ingreso_hogar = float(sum(float(income_by_id.get(fid, 0.0)) for fid in fam_ids))
        deuda_hogar = float(sum(float(debt_by_id.get(fid, 0.0)) for fid in fam_ids))
        rows.append(
            {
                "hogar_id": hogar_id,
                "INGRESO_HOGAR": ingreso_hogar,
                "DEUDA_HOGAR": deuda_hogar,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["hogar_id", "INGRESO_HOGAR", "DEUDA_HOGAR", "QUINTIL", "GRUPO_QUINTIL"])

    df = pd.DataFrame(rows).drop_duplicates(subset=["hogar_id"]).copy()
    df["QUINTIL"] = df["INGRESO_HOGAR"].apply(_assign_quintil_ingreso)
    df["GRUPO_QUINTIL"] = df["QUINTIL"].apply(
        lambda q: f"Quintil {int(q)}" if pd.notna(q) else SIN_EMPLEO_FORMAL_LABEL
    )
    return df


def _build_outlier_table(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Quintil",
                "Observaciones",
                "Outliers",
                "Tasa outliers (%)",
                "Limite inferior",
                "Limite superior",
            ]
        )

    for quintil_label in QUINTIL_ORDER:
        g = df[df["GRUPO_QUINTIL"] == quintil_label].copy()
        vals = pd.to_numeric(g[value_col], errors="coerce").dropna()
        n_obs = int(len(vals))

        if n_obs == 0:
            rows.append(
                {
                    "Quintil": quintil_label,
                    "Observaciones": 0,
                    "Outliers": 0,
                    "Tasa outliers (%)": 0.0,
                    "Limite inferior": np.nan,
                    "Limite superior": np.nan,
                }
            )
            continue

        q1 = float(vals.quantile(0.25))
        q3 = float(vals.quantile(0.75))
        iqr = float(q3 - q1)
        lim_inf = max(0.0, q1 - 1.5 * iqr)
        lim_sup = q3 + 1.5 * iqr
        n_out = int(((vals < lim_inf) | (vals > lim_sup)).sum())
        tasa = float((n_out / n_obs) * 100.0) if n_obs > 0 else 0.0

        rows.append(
            {
                "Quintil": quintil_label,
                "Observaciones": n_obs,
                "Outliers": n_out,
                "Tasa outliers (%)": round(tasa, 2),
                "Limite inferior": round(lim_inf, 2),
                "Limite superior": round(lim_sup, 2),
            }
        )

    return pd.DataFrame(rows)


def _build_hogares_por_parroquia(
    links: pd.DataFrame,
    info_personal: pd.DataFrame,
    parroquias: pd.DataFrame,
) -> pd.DataFrame:
    if links.empty:
        return pd.DataFrame(columns=["Parroquia", "Hogares"])

    df_links = links[["IDENTIFICACION", "CED_PADRE", "CED_MADRE"]].copy()
    df_links["hogar_id"] = df_links.apply(
        lambda r: (
            "|".join(
                str(v)
                for v in sorted(
                    {
                        int(v)
                        for v in [r["CED_PADRE"], r["CED_MADRE"]]
                        if pd.notna(v) and int(v) > 0
                    }
                )
            )
            if any(
                (pd.notna(v) and int(v) > 0) for v in [r["CED_PADRE"], r["CED_MADRE"]]
            )
            else None
        ),
        axis=1,
    )
    df_links = df_links.dropna(subset=["hogar_id"]).copy()
    if df_links.empty:
        return pd.DataFrame(columns=["Parroquia", "Hogares"])

    # Miembros evaluables del hogar: hijo(s) + padre + madre.
    miembros_rows: list[dict[str, int | str]] = []
    for _, row in df_links.iterrows():
        hogar_id = str(row["hogar_id"])
        for pid in [row["IDENTIFICACION"], row["CED_PADRE"], row["CED_MADRE"]]:
            if pd.notna(pid) and int(pid) > 0:
                miembros_rows.append(
                    {"hogar_id": hogar_id, "IDENTIFICACION": int(pid)}
                )

    if not miembros_rows:
        return pd.DataFrame(columns=["Parroquia", "Hogares"])

    df_miembros = pd.DataFrame(miembros_rows).drop_duplicates()
    miembros_por_hogar = (
        df_miembros.groupby("hogar_id", as_index=False)["IDENTIFICACION"]
        .agg(list)
        .rename(columns={"IDENTIFICACION": "miembros"})
    )

    fecha_col = (
        "FECHA EXPEDICION"
        if "FECHA EXPEDICION" in info_personal.columns
        else ("FECHA_EXPEDICION" if "FECHA_EXPEDICION" in info_personal.columns else None)
    )

    if info_personal.empty:
        df_info = pd.DataFrame(
            columns=["IDENTIFICACION", "LATITUD", "LONGITUD", "FECHA_EXPEDICION_DT"]
        )
    else:
        cols = ["IDENTIFICACION", "LATITUD", "LONGITUD"]
        if fecha_col is not None:
            cols.append(fecha_col)
        df_info = info_personal[cols].copy()
        df_info["IDENTIFICACION"] = pd.to_numeric(
            df_info["IDENTIFICACION"], errors="coerce"
        )
        df_info = df_info.dropna(subset=["IDENTIFICACION"]).copy()
        df_info["IDENTIFICACION"] = df_info["IDENTIFICACION"].astype(int)
        df_info["LATITUD"] = pd.to_numeric(df_info["LATITUD"], errors="coerce")
        df_info["LONGITUD"] = pd.to_numeric(df_info["LONGITUD"], errors="coerce")
        if fecha_col is not None:
            df_info["FECHA_EXPEDICION_DT"] = pd.to_datetime(
                df_info[fecha_col], errors="coerce", dayfirst=True
            )
        else:
            df_info["FECHA_EXPEDICION_DT"] = pd.NaT

    if not df_info.empty:
        df_info = (
            df_info.sort_values(
                ["IDENTIFICACION", "FECHA_EXPEDICION_DT"],
                ascending=[True, False],
                na_position="last",
            )
            .drop_duplicates(subset=["IDENTIFICACION"], keep="first")
            .set_index("IDENTIFICACION")
        )

    parish_geoms: list[tuple[str, object]] = []
    if gpd is not None and Point is not None and not parroquias.empty:
        for _, parr_row in parroquias.iterrows():
            geom = parr_row.get("geometry", None)
            if geom is None:
                continue
            parish_geoms.append((str(parr_row.get("PARROQUIA", "Sin nombre")), geom))

    parroquia_rows: list[dict[str, str]] = []
    for _, row in miembros_por_hogar.iterrows():
        hogar_id = str(row["hogar_id"])
        miembros = row["miembros"]
        candidates: list[dict[str, object]] = []

        for pid in miembros:
            if df_info.empty or int(pid) not in df_info.index:
                continue
            p = df_info.loc[int(pid)]
            lat = pd.to_numeric(p.get("LATITUD"), errors="coerce")
            lon = pd.to_numeric(p.get("LONGITUD"), errors="coerce")
            fecha = p.get("FECHA_EXPEDICION_DT", pd.NaT)
            if pd.isna(lat) or pd.isna(lon):
                continue
            candidates.append(
                {
                    "pid": int(pid),
                    "lat": float(lat),
                    "lon": float(lon),
                    "fecha": fecha if pd.notna(fecha) else pd.NaT,
                }
            )

        # Prioriza fecha de expedición más reciente; sin fecha quedan al final.
        candidates = sorted(
            candidates,
            key=lambda c: (
                1 if pd.notna(c["fecha"]) else 0,
                c["fecha"] if pd.notna(c["fecha"]) else pd.Timestamp.min,
            ),
            reverse=True,
        )

        parroquia = "Sin parroquia"
        if Point is not None and parish_geoms:
            for c in candidates:
                punto = Point(float(c["lon"]), float(c["lat"]))
                found = False
                for nombre_parr, geom in parish_geoms:
                    if geom.contains(punto):
                        parroquia = nombre_parr
                        found = True
                        break
                if found:
                    break

        parroquia_rows.append({"hogar_id": hogar_id, "Parroquia": parroquia})

    df_parr = pd.DataFrame(parroquia_rows)
    return (
        df_parr.groupby("Parroquia", as_index=False)["hogar_id"]
        .nunique()
        .rename(columns={"hogar_id": "Hogares"})
        .sort_values("Hogares", ascending=False)
        .reset_index(drop=True)
    )


with st.spinner("Cargando datos..."):
    estudiantes, universo_familiares, info_personal, empleos, deudas = load_data()

if "IDENTIFICACION" not in estudiantes.columns:
    st.error("No existe la columna IDENTIFICACION en la hoja Estudiantes.")
    st.stop()

ids_est = set(_to_int_id(estudiantes["IDENTIFICACION"]).unique().tolist())
if not ids_est:
    st.error("No se pudieron extraer IDs de estudiantes.")
    st.stop()

info = info_personal.copy()
info["IDENTIFICACION"] = pd.to_numeric(info["IDENTIFICACION"], errors="coerce")
info = info.dropna(subset=["IDENTIFICACION"]).copy()
info["IDENTIFICACION"] = info["IDENTIFICACION"].astype(int)

est_info = info[info["IDENTIFICACION"].isin(ids_est)].copy()
if est_info.empty:
    st.error("No hay informacion personal para los estudiantes.")
    st.stop()

est_info["FECHA_NACIMIENTO"] = pd.to_datetime(
    est_info.get("FECHA_NACIMIENTO"), errors="coerce", dayfirst=True
)
today = pd.Timestamp.today().normalize()
est_info["EDAD"] = np.floor((today - est_info["FECHA_NACIMIENTO"]).dt.days / 365.25)
est_info.loc[est_info["EDAD"] < 0, "EDAD"] = np.nan

links = universo_familiares.copy()
for c in ["IDENTIFICACION", "CED_PADRE", "CED_MADRE"]:
    links[c] = pd.to_numeric(links[c], errors="coerce")
links = links.dropna(subset=["IDENTIFICACION"]).copy()
links["IDENTIFICACION"] = links["IDENTIFICACION"].astype(int)
links = links[links["IDENTIFICACION"].isin(ids_est)].copy()
for c in ["CED_PADRE", "CED_MADRE"]:
    links[c] = links[c].fillna(0).astype(int)

income_by_id = _build_income_latest(empleos)
debt_by_id = _build_debt_latest(deudas)

ids_hogar = set()
for c in ["CED_PADRE", "CED_MADRE"]:
    if c in links.columns:
        ids_hogar.update([int(v) for v in links[c].tolist() if int(v) > 0])
hogar_info = info[info["IDENTIFICACION"].isin(ids_hogar)].copy()
hogar_info["FECHA_NACIMIENTO"] = pd.to_datetime(
    hogar_info.get("FECHA_NACIMIENTO"), errors="coerce", dayfirst=True
)
hogar_info["EDAD"] = np.floor((today - hogar_info["FECHA_NACIMIENTO"]).dt.days / 365.25)
hogar_info.loc[hogar_info["EDAD"] < 0, "EDAD"] = np.nan
hogar_info["INGRESO_FAMILIAR"] = hogar_info["IDENTIFICACION"].map(income_by_id)
hogar_info["DEUDA_FAMILIAR"] = hogar_info["IDENTIFICACION"].map(debt_by_id)

hogar_rows: list[dict[str, float | int]] = []
for _, row in links.iterrows():
    padre = int(row["CED_PADRE"])
    madre = int(row["CED_MADRE"])
    fam_ids = [fid for fid in [padre, madre] if fid > 0]
    if not fam_ids:
        continue
    ingreso_hogar = float(sum(float(income_by_id.get(fid, 0.0)) for fid in fam_ids))
    deuda_hogar = float(sum(float(debt_by_id.get(fid, 0.0)) for fid in fam_ids))
    hogar_rows.append(
        {
            "IDENTIFICACION": int(row["IDENTIFICACION"]),
            "INGRESO_HOGAR": ingreso_hogar,
            "DEUDA_HOGAR": deuda_hogar,
        }
    )

df_hogar = pd.DataFrame(hogar_rows).drop_duplicates(subset=["IDENTIFICACION"])
base = est_info.merge(df_hogar, on="IDENTIFICACION", how="left")

ids_sorted = sorted(ids_est)
n_ids = len(ids_sorted)
last_digits = pd.Series([i % 10 for i in ids_sorted], name="DIGITO")
digit_counts = (
    last_digits.value_counts()
    .reindex(list(range(10)), fill_value=0)
    .rename_axis("DIGITO")
    .reset_index(name="FRECUENCIA")
)
esperado = n_ids / 10.0
chi2_stat = float((((digit_counts["FRECUENCIA"] - esperado) ** 2) / esperado).sum())
p_value = (
    float(1.0 - chi2_dist.cdf(chi2_stat, df=9)) if chi2_dist is not None else np.nan
)
max_run = _max_consecutive_run(ids_sorted)

conc_variables = ["SEXO", "UBICACION", "NIVEL_ESTUDIO", "ESTADO_CIVIL"]
df_conc_est = _build_concentration_table(base, conc_variables, "Estudiantes")
df_conc_hogar = _build_concentration_table(
    hogar_info, conc_variables, "Hogar estudiantes"
)
df_conc = pd.concat([df_conc_est, df_conc_hogar], ignore_index=True)

df_hogares_quintil = _build_hogares_quintil(links, income_by_id, debt_by_id)
if not df_hogares_quintil.empty:
    df_hogares_quintil["GRUPO_QUINTIL"] = pd.Categorical(
        df_hogares_quintil["GRUPO_QUINTIL"],
        categories=QUINTIL_ORDER,
        ordered=True,
    )
    df_hogares_quintil = df_hogares_quintil.sort_values("GRUPO_QUINTIL")

hogares_por_quintil = (
    df_hogares_quintil.groupby("GRUPO_QUINTIL", observed=False)["hogar_id"]
    .nunique()
    .reindex(QUINTIL_ORDER, fill_value=0)
)
hogares_deuda_por_quintil = (
    df_hogares_quintil[df_hogares_quintil["DEUDA_HOGAR"] > 0]
    .groupby("GRUPO_QUINTIL", observed=False)["hogar_id"]
    .nunique()
    .reindex(QUINTIL_ORDER, fill_value=0)
)

total_hogares_quintil = int(hogares_por_quintil.sum())
total_hogares_con_deuda_quintil = int(hogares_deuda_por_quintil.sum())

df_quintil_ingreso = pd.DataFrame(
    {
        "Quintil": QUINTIL_ORDER,
        "Hogares": [int(hogares_por_quintil.get(q, 0)) for q in QUINTIL_ORDER],
    }
)
if total_hogares_quintil > 0:
    df_quintil_ingreso["Participacion (%)"] = (
        (df_quintil_ingreso["Hogares"] / total_hogares_quintil) * 100.0
    ).round(2)
else:
    df_quintil_ingreso["Participacion (%)"] = 0.0

df_quintil_deuda = pd.DataFrame(
    {
        "Quintil": QUINTIL_ORDER,
        "Hogares con deuda": [
            int(hogares_deuda_por_quintil.get(q, 0)) for q in QUINTIL_ORDER
        ],
    }
)
if total_hogares_con_deuda_quintil > 0:
    df_quintil_deuda["Participacion deuda (%)"] = (
        (df_quintil_deuda["Hogares con deuda"] / total_hogares_con_deuda_quintil)
        * 100.0
    ).round(2)
else:
    df_quintil_deuda["Participacion deuda (%)"] = 0.0
df_quintil_deuda["Cobertura dentro quintil (%)"] = np.where(
    df_quintil_ingreso["Hogares"] > 0,
    (df_quintil_deuda["Hogares con deuda"] / df_quintil_ingreso["Hogares"]) * 100.0,
    0.0,
).round(2)

top_quintil_ingreso, top_share_ingreso = _top_category_share(
    df_hogares_quintil["GRUPO_QUINTIL"]
)
hhi_ingreso_quintil = _hh_index(df_hogares_quintil["GRUPO_QUINTIL"])
flag_conc_quintil_ingreso = bool(
    total_hogares_quintil > 0
    and (top_share_ingreso >= 0.7 or hhi_ingreso_quintil >= 0.35)
)

df_hogares_quintil_deuda = df_hogares_quintil[df_hogares_quintil["DEUDA_HOGAR"] > 0].copy()
top_quintil_deuda, top_share_deuda = _top_category_share(
    df_hogares_quintil_deuda["GRUPO_QUINTIL"]
)
hhi_deuda_quintil = _hh_index(df_hogares_quintil_deuda["GRUPO_QUINTIL"])
flag_conc_quintil_deuda = bool(
    total_hogares_con_deuda_quintil > 0
    and (top_share_deuda >= 0.7 or hhi_deuda_quintil >= 0.35)
)

df_outliers_ingreso = _build_outlier_table(df_hogares_quintil, "INGRESO_HOGAR")
df_outliers_deuda = _build_outlier_table(df_hogares_quintil, "DEUDA_HOGAR")

df_parroquias = _load_parroquias_geojson()
df_hogares_parroquia = _build_hogares_por_parroquia(links, info, df_parroquias)
total_hogares_parroquia = (
    int(df_hogares_parroquia["Hogares"].sum()) if not df_hogares_parroquia.empty else 0
)

out_ingreso_total = int(df_outliers_ingreso["Outliers"].sum()) if not df_outliers_ingreso.empty else 0
out_deuda_total = int(df_outliers_deuda["Outliers"].sum()) if not df_outliers_deuda.empty else 0
max_outlier_ingreso = (
    float(df_outliers_ingreso["Tasa outliers (%)"].max())
    if not df_outliers_ingreso.empty
    else 0.0
)
max_outlier_deuda = (
    float(df_outliers_deuda["Tasa outliers (%)"].max())
    if not df_outliers_deuda.empty
    else 0.0
)
flag_outlier_ingreso = bool(max_outlier_ingreso >= (OUTLIER_RATE_ALERT * 100))
flag_outlier_deuda = bool(max_outlier_deuda >= (OUTLIER_RATE_ALERT * 100))


def _coverage(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    s = _clean_series_for_coverage(df[col])
    return float(s.notna().mean())


coverage_rows_est = [
    {
        "Grupo": "Estudiantes",
        "Variable": "SEXO",
        "Cobertura (%)": round(_coverage(base, "SEXO") * 100, 2),
    },
    {
        "Grupo": "Estudiantes",
        "Variable": "EDAD",
        "Cobertura (%)": round(_coverage(base, "EDAD") * 100, 2),
    },
    {
        "Grupo": "Estudiantes",
        "Variable": "UBICACION",
        "Cobertura (%)": round(_coverage(base, "UBICACION") * 100, 2),
    },
    {
        "Grupo": "Estudiantes",
        "Variable": "NIVEL_ESTUDIO",
        "Cobertura (%)": round(_coverage(base, "NIVEL_ESTUDIO") * 100, 2),
    },
    {
        "Grupo": "Estudiantes",
        "Variable": "INGRESO_HOGAR",
        "Cobertura (%)": round(_coverage(base, "INGRESO_HOGAR") * 100, 2),
    },
    {
        "Grupo": "Estudiantes",
        "Variable": "DEUDA_HOGAR",
        "Cobertura (%)": round(_coverage(base, "DEUDA_HOGAR") * 100, 2),
    },
]
coverage_rows_hogar = [
    {
        "Grupo": "Hogar estudiantes",
        "Variable": "SEXO",
        "Cobertura (%)": round(_coverage(hogar_info, "SEXO") * 100, 2),
    },
    {
        "Grupo": "Hogar estudiantes",
        "Variable": "EDAD",
        "Cobertura (%)": round(_coverage(hogar_info, "EDAD") * 100, 2),
    },
    {
        "Grupo": "Hogar estudiantes",
        "Variable": "UBICACION",
        "Cobertura (%)": round(_coverage(hogar_info, "UBICACION") * 100, 2),
    },
    {
        "Grupo": "Hogar estudiantes",
        "Variable": "NIVEL_ESTUDIO",
        "Cobertura (%)": round(_coverage(hogar_info, "NIVEL_ESTUDIO") * 100, 2),
    },
    {
        "Grupo": "Hogar estudiantes",
        "Variable": "INGRESO_FAMILIAR",
        "Cobertura (%)": round(_coverage(hogar_info, "INGRESO_FAMILIAR") * 100, 2),
    },
    {
        "Grupo": "Hogar estudiantes",
        "Variable": "DEUDA_FAMILIAR",
        "Cobertura (%)": round(_coverage(hogar_info, "DEUDA_FAMILIAR") * 100, 2),
    },
]
df_cov_est = pd.DataFrame(coverage_rows_est)
df_cov_hogar = pd.DataFrame(coverage_rows_hogar)
df_cov = pd.concat([df_cov_est, df_cov_hogar], ignore_index=True)

n_muestra = len(ids_est)
n_universo = UNIVERSO_TOTAL_ESTUDIANTES
margen_error, n_requerida_5 = _sample_size_consistency(
    n_muestra=n_muestra,
    n_universo=n_universo,
    error_objetivo=MARGEN_ERROR_OBJETIVO,
)
cumple_5 = bool(np.isfinite(margen_error) and margen_error <= MARGEN_ERROR_OBJETIVO)

coherencia_sede: list[dict[str, float | str]] = []
for sede, n_universo_sede in UNIVERSO_POR_SEDE.items():
    n_requerida_5_sede = _required_sample_size(
        n_universo=n_universo_sede,
        error_objetivo=MARGEN_ERROR_OBJETIVO,
    )
    coherencia_sede.append(
        {
            "Sede": sede,
            "Universo total (N)": float(n_universo_sede),
            "Margen de error (95%)": float(MARGEN_ERROR_OBJETIVO * 100),
            "n requerida para 5%": float(n_requerida_5_sede),
        }
    )

signals = 0
if not cumple_5:
    signals += 1
if not np.isnan(p_value) and p_value < 0.05:
    signals += 1
if max_run >= 5:
    signals += 1
if flag_conc_quintil_ingreso:
    signals += 1
if flag_conc_quintil_deuda:
    signals += 1
if flag_outlier_ingreso:
    signals += 1
if flag_outlier_deuda:
    signals += 1

if signals <= 1:
    nivel = "Bajo"
    nivel_color = "🟢"
elif signals <= 3:
    nivel = "Medio"
    nivel_color = "🟡"
else:
    nivel = "Alto"
    nivel_color = "🔴"

# Resumen ejecutivo
st.markdown("## 📊 Resumen Ejecutivo")
with st.container():
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown(
            f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; color: white; text-align: center;'>
            <h3 style='margin: 0; color: white;'>{len(ids_est):,}</h3>
            <p style='margin: 5px 0 0 0; color: white;'>Estudiantes en muestra</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 20px; border-radius: 10px; color: white; text-align: center;'>
            <h3 style='margin: 0; color: white;'>{signals}</h3>
            <p style='margin: 5px 0 0 0; color: white;'>Señales detectadas</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 20px; border-radius: 10px; color: white; text-align: center;'>
            <h3 style='margin: 0; color: white;'>{nivel_color} {nivel}</h3>
            <p style='margin: 5px 0 0 0; color: white;'>Nivel de indicios</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

st.markdown("---")

# Métricas adicionales
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.metric(
        "📈 Con ingreso de hogar",
        f"{int((base['INGRESO_HOGAR'].fillna(0) > 0).sum()):,}",
    )
with col_b:
    st.metric(
        "💳 Con deuda de hogar", f"{int((base['DEUDA_HOGAR'].fillna(0) > 0).sum()):,}"
    )
with col_c:
    pct_ing = (base["INGRESO_HOGAR"].fillna(0) > 0).mean() * 100
    st.metric("% con ingreso", f"{pct_ing:.1f}%")
with col_d:
    pct_deu = (base["DEUDA_HOGAR"].fillna(0) > 0).mean() * 100
    st.metric("% con deuda", f"{pct_deu:.1f}%")

st.markdown("## 📐 Coherencia de Tamaño Muestral")
with st.container():
    st.markdown(
        """
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #667eea; margin-bottom: 15px;'>
        <p style='margin: 0; color: #262730;'>
        <strong>Evaluación estadística:</strong> Verifica si el tamaño de la muestra es coherente 
        con el universo total para lograr precisión estadística (95% confianza, p=0.5).
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("🎯 Universo total (N)", f"{n_universo:,}")
    mc2.metric("📋 Muestra (n)", f"{n_muestra:,}")
    mc3.metric("📊 Margen de error (95%)", f"{margen_error*100:.2f}%")
    mc4.metric("✅ n requerida para 5%", f"{n_requerida_5:.0f}")

    if cumple_5:
        st.success("✅ Dictamen: Muestra coherente para margen de error del 5%")
    else:
        st.error("❌ Dictamen: Muestra NO coherente para margen de error del 5%")

st.markdown("### Coherencia muestral por sede")
sede_cols = st.columns(len(coherencia_sede))
for col, metrica in zip(sede_cols, coherencia_sede):
    with col:
        st.markdown(f"#### {metrica['Sede']}")
        s1, s2, s3 = st.columns(3)
        s1.metric("🎯 Universo total (N)", f"{int(metrica['Universo total (N)']):,}")
        s2.metric("📊 Margen de error (95%)", f"{metrica['Margen de error (95%)']:.2f}%")
        s3.metric("✅ n requerida para 5%", f"{metrica['n requerida para 5%']:.0f}")

st.markdown("## 📋 Cobertura de Variables")
with st.container():
    st.markdown(
        """
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #f5576c; margin-bottom: 15px;'>
        <p style='margin: 0; color: #262730;'>
        <strong>Análisis de completitud:</strong> Porcentaje de datos disponibles por variable. 
        Coberturas inferiores al 70% incrementan el riesgo de sesgo por datos faltantes.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    cov_col_1, cov_col_2 = st.columns(2)
    with cov_col_1:
        st.markdown("### 👨‍🎓 Estudiantes")
        st.dataframe(
            df_cov_est.drop(columns=["Grupo"]).style.background_gradient(
                subset=["Cobertura (%)"], cmap="RdYlGn", vmin=0, vmax=100
            ),
            use_container_width=True,
            hide_index=True,
        )
    with cov_col_2:
        st.markdown("### 🏠 Hogar de Estudiantes")
        st.dataframe(
            df_cov_hogar.drop(columns=["Grupo"]).style.background_gradient(
                subset=["Cobertura (%)"], cmap="RdYlGn", vmin=0, vmax=100
            ),
            use_container_width=True,
            hide_index=True,
        )

st.markdown("## 🎯 Concentración Demográfica")
with st.container():
    st.markdown(
        """
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #00f2fe; margin-bottom: 15px;'>
        <p style='margin: 0; color: #262730;'>
        <strong>Diversidad de la muestra:</strong> Las tablas demográficas se mantienen como referencia descriptiva.
        En esta sección también se agregan métricas de quintiles y outliers del hogar.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Nota: La concentración demográfica (sexo, ubicación, nivel de estudio, estado civil)."
    )

    col_conc_1, col_conc_2 = st.columns(2)
    with col_conc_1:
        st.markdown("### 👨‍🎓 Estudiantes")
        if df_conc_est.empty:
            st.info("No hay variables categóricas suficientes para este grupo.")
        else:
            st.dataframe(
                df_conc_est.drop(columns=["Grupo"]),
                use_container_width=True,
                hide_index=True,
            )
    with col_conc_2:
        st.markdown("### 🏠 Hogar de Estudiantes")
        if df_conc_hogar.empty:
            st.info("No hay variables categóricas suficientes para este grupo.")
        else:
            st.dataframe(
                df_conc_hogar.drop(columns=["Grupo"]),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("### 🏠 Cantidad de hogares por quintil de ingreso")
    q_ing_1, q_ing_2, q_ing_3 = st.columns(3)
    q_ing_1.metric("Hogares en grupos", f"{total_hogares_quintil:,}")
    q_ing_2.metric(
        "Quintil dominante (ingreso)",
        top_quintil_ingreso,
        f"{top_share_ingreso*100:.2f}%",
    )
    q_ing_3.metric(
        "HHI quintil ingreso",
        f"{hhi_ingreso_quintil:.4f}",
        delta="Concentrado" if flag_conc_quintil_ingreso else "Diverso",
        delta_color="inverse" if flag_conc_quintil_ingreso else "off",
    )
    st.dataframe(
        df_quintil_ingreso.style.background_gradient(
            subset=["Hogares", "Participacion (%)"], cmap="Blues"
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 💳 Cantidad de hogares con deuda por quintil")
    q_deu_1, q_deu_2, q_deu_3 = st.columns(3)
    q_deu_1.metric("Hogares con deuda", f"{total_hogares_con_deuda_quintil:,}")
    q_deu_2.metric(
        "Quintil dominante (deuda)",
        top_quintil_deuda,
        f"{top_share_deuda*100:.2f}%",
    )
    q_deu_3.metric(
        "HHI quintil deuda",
        f"{hhi_deuda_quintil:.4f}",
        delta="Concentrado" if flag_conc_quintil_deuda else "Diverso",
        delta_color="inverse" if flag_conc_quintil_deuda else "off",
    )
    st.dataframe(
        df_quintil_deuda.style.background_gradient(
            subset=["Hogares con deuda", "Participacion deuda (%)"], cmap="Reds"
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 📦 Outliers por quintil en ingresos y deudas")
    out_1, out_2 = st.columns(2)
    with out_1:
        st.metric(
            "Outliers ingreso (total)",
            f"{out_ingreso_total:,}",
            delta=f"Max quintil {max_outlier_ingreso:.2f}%",
            delta_color="inverse" if flag_outlier_ingreso else "off",
        )
        st.dataframe(
            df_outliers_ingreso.style.background_gradient(
                subset=["Outliers", "Tasa outliers (%)"], cmap="Oranges"
            ),
            use_container_width=True,
            hide_index=True,
        )
    with out_2:
        st.metric(
            "Outliers deuda (total)",
            f"{out_deuda_total:,}",
            delta=f"Max quintil {max_outlier_deuda:.2f}%",
            delta_color="inverse" if flag_outlier_deuda else "off",
        )
        st.dataframe(
            df_outliers_deuda.style.background_gradient(
                subset=["Outliers", "Tasa outliers (%)"], cmap="Purples"
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### 🗺️ Cantidad de hogares por parroquia")
    if gpd is None or Point is None:
        st.info(
            "No se pudo calcular parroquia por ausencia de dependencias geoespaciales."
        )
    elif df_hogares_parroquia.empty:
        st.info("No hay datos suficientes para calcular hogares por parroquia.")
    else:
        pcol1, pcol2 = st.columns(2)
        pcol1.metric("Parroquias con hogares", f"{len(df_hogares_parroquia):,}")
        pcol2.metric("Hogares georreferenciados", f"{total_hogares_parroquia:,}")
        st.dataframe(
            df_hogares_parroquia.style.background_gradient(
                subset=["Hogares"], cmap="YlGnBu"
            ),
            use_container_width=True,
            hide_index=True,
        )

st.markdown("## 🆔 Patrones en Cédulas de Estudiantes")
with st.container():
    st.markdown(
        """
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #764ba2; margin-bottom: 15px;'>
        <p style='margin: 0; color: #262730;'>
        <strong>Aleatoriedad en IDs:</strong> Verifica patrones inusuales en las cédulas. 
        En selección aleatoria, los últimos dígitos se distribuyen uniformemente y no hay rachas largas consecutivas.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    id_cols = st.columns(3)
    id_cols[0].metric("📊 Chi-cuadrado último dígito", f"{chi2_stat:.2f}")
    id_cols[1].metric(
        "📈 p-valor último dígito",
        f"{p_value:.4f}" if not np.isnan(p_value) else "N/D",
        delta=(
            "Significativo"
            if (not np.isnan(p_value) and p_value < 0.05)
            else "No significativo"
        ),
        delta_color="inverse" if (not np.isnan(p_value) and p_value < 0.05) else "off",
    )
    id_cols[2].metric(
        "🔢 Racha consecutiva máxima",
        str(max_run),
        delta="Alto" if max_run >= 5 else "Normal",
        delta_color="inverse" if max_run >= 5 else "normal",
    )

    fig_digits = px.bar(
        digit_counts,
        x="DIGITO",
        y="FRECUENCIA",
        title="Frecuencia de último dígito en cédulas",
        text="FRECUENCIA",
        color="FRECUENCIA",
        color_continuous_scale="Blues",
    )
    fig_digits.update_layout(
        height=400,
        xaxis_title="Último dígito",
        yaxis_title="Frecuencia",
        showlegend=False,
    )
    st.plotly_chart(fig_digits, use_container_width=True)

st.markdown("## 💰 Diversidad Socioeconómica Familiar")
with st.container():
    st.markdown(
        """
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #f093fb; margin-bottom: 15px;'>
        <p style='margin: 0; color: #262730;'>
        <strong>Perfil económico:</strong> Estadísticas descriptivas de ingresos y deudas de los hogares estudiantes.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    ing = base["INGRESO_HOGAR"].fillna(0.0)
    deu = base["DEUDA_HOGAR"].fillna(0.0)

    soc_cols = st.columns(2)

    with soc_cols[0]:
        st.markdown("### 💵 Ingresos del Hogar")
        col1, col2, col3 = st.columns(3)
        col1.metric("Promedio", f"${ing.mean():,.2f}")
        col2.metric("Mediana", f"${ing.median():,.2f}")
        col3.metric("Sin ingreso", f"{(ing <= 0).mean() * 100:.1f}%")

        # Gráfico de distribución de ingresos
        ing_no_zero = ing[ing > 0]
        if len(ing_no_zero) > 0:
            fig_ing = px.histogram(
                ing_no_zero,
                nbins=30,
                title="Distribución de ingresos (excluye $0)",
                labels={"value": "Ingreso", "count": "Frecuencia"},
            )
            fig_ing.update_layout(showlegend=False, height=250)
            st.plotly_chart(fig_ing, use_container_width=True)

    with soc_cols[1]:
        st.markdown("### 💳 Deudas del Hogar")
        col1, col2, col3 = st.columns(3)
        col1.metric("Promedio", f"${deu.mean():,.2f}")
        col2.metric("Mediana", f"${deu.median():,.2f}")
        col3.metric("Sin deuda", f"{(deu <= 0).mean() * 100:.1f}%")

        # Gráfico de distribución de deudas
        deu_no_zero = deu[deu > 0]
        if len(deu_no_zero) > 0:
            fig_deu = px.histogram(
                deu_no_zero,
                nbins=30,
                title="Distribución de deudas (excluye $0)",
                labels={"value": "Deuda", "count": "Frecuencia"},
                color_discrete_sequence=["#f5576c"],
            )
            fig_deu.update_layout(showlegend=False, height=250)
            st.plotly_chart(fig_deu, use_container_width=True)

# Calculo de señales detectadas para la conclusión
detected_signals: list[dict[str, str]] = []

if not cumple_5:
    detected_signals.append(
        {
            "senal": "Tamaño muestral no coherente para 5%",
            "justificacion": (
                "Se usa para verificar precisión estadística mínima esperada "
                "(95% de confianza, p=0.5)."
            ),
            "interpretacion": (
                f"Con N={n_universo} y n={n_muestra}, el margen de error estimado es "
                f"{margen_error*100:.2f}%. Para un margen objetivo del 5% se requiere "
                f"n≈{n_requerida_5:.0f}."
            ),
        }
    )

if not np.isnan(p_value) and p_value < 0.05:
    detected_signals.append(
        {
            "senal": "Patrón no uniforme en último dígito de cédula",
            "justificacion": (
                "En una selección más aleatoria, los últimos dígitos tienden a distribuirse "
                "sin desviaciones fuertes."
            ),
            "interpretacion": (
                f"El test chi-cuadrado reporta p-valor={p_value:.4f} (<0.05), "
                "sugiriendo una distribución no uniforme."
            ),
        }
    )

if max_run >= 5:
    detected_signals.append(
        {
            "senal": "Racha alta de cédulas consecutivas",
            "justificacion": (
                "Rachas largas consecutivas pueden indicar captura por bloques "
                "o selección administrativa."
            ),
            "interpretacion": (
                f"Se observó racha máxima de {max_run} cédulas consecutivas (umbral: >=5)."
            ),
        }
    )

if flag_conc_quintil_ingreso:
    detected_signals.append(
        {
            "senal": "Concentración de hogares por quintil de ingreso",
            "justificacion": (
                "Una distribución muy concentrada por quintil puede indicar "
                "sobrerrepresentación socioeconómica."
            ),
            "interpretacion": (
                f"Quintil dominante: {top_quintil_ingreso} ({top_share_ingreso*100:.2f}%), "
                f"HHI={hhi_ingreso_quintil:.4f}."
            ),
        }
    )

if flag_conc_quintil_deuda:
    detected_signals.append(
        {
            "senal": "Concentración de hogares con deuda por quintil",
            "justificacion": (
                "Si la deuda se acumula en pocos quintiles, puede sesgar "
                "las conclusiones financieras."
            ),
            "interpretacion": (
                f"Quintil dominante con deuda: {top_quintil_deuda} ({top_share_deuda*100:.2f}%), "
                f"HHI={hhi_deuda_quintil:.4f}."
            ),
        }
    )

if flag_outlier_ingreso:
    fila_peak_ing = (
        df_outliers_ingreso.loc[df_outliers_ingreso["Tasa outliers (%)"].idxmax()]
        if not df_outliers_ingreso.empty
        else None
    )
    if fila_peak_ing is not None:
        detected_signals.append(
            {
                "senal": "Outliers altos de ingreso en quintiles",
                "justificacion": (
                    "Valores extremos altos por quintil afectan media y dispersión, "
                    "distorsionando lectura de representatividad."
                ),
                "interpretacion": (
                    f"Total outliers ingreso: {out_ingreso_total}. "
                    f"Máxima tasa en {fila_peak_ing['Quintil']}: "
                    f"{float(fila_peak_ing['Tasa outliers (%)']):.2f}%."
                ),
            }
        )

if flag_outlier_deuda:
    fila_peak_deu = (
        df_outliers_deuda.loc[df_outliers_deuda["Tasa outliers (%)"].idxmax()]
        if not df_outliers_deuda.empty
        else None
    )
    if fila_peak_deu is not None:
        detected_signals.append(
            {
                "senal": "Outliers altos de deuda en quintiles",
                "justificacion": (
                    "Valores extremos de deuda pueden inflar indicadores agregados "
                    "y sesgar comparaciones por quintil."
                ),
                "interpretacion": (
                    f"Total outliers deuda: {out_deuda_total}. "
                    f"Máxima tasa en {fila_peak_deu['Quintil']}: "
                    f"{float(fila_peak_deu['Tasa outliers (%)']):.2f}%."
                ),
            }
        )

st.markdown("## 📝 Conclusión Automática")
with st.container():
    if nivel == "Bajo":
        color_fondo = "#d4edda"
        color_borde = "#28a745"
        icono = "✅"
    elif nivel == "Medio":
        color_fondo = "#fff3cd"
        color_borde = "#ffc107"
        icono = "⚠️"
    else:
        color_fondo = "#f8d7da"
        color_borde = "#dc3545"
        icono = "❌"

    st.markdown(
        f"""
    <div style='background-color: {color_fondo}; padding: 20px; border-radius: 10px; border-left: 5px solid {color_borde};'>
        <h3 style='margin-top: 0; color: #262730;'>{icono} Nivel de Indicios: {nivel}</h3>
        <p style='font-size: 16px; margin: 10px 0; color: #262730;'>
        <strong>Señales detectadas:</strong> {signals}
        </p>
        <p style='margin-bottom: 0; color: #262730;'>
        Este resultado es <strong>exploratorio</strong> y debe interpretarse como alerta de posible sesgo, 
        no como prueba estadística formal de aleatoriedad. Se requiere análisis adicional para conclusiones definitivas.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("## 🔎 Detalle de Señales Detectadas")
if not detected_signals:
    st.success("✅ No se detectaron señales de sesgo con los umbrales actuales.")
else:
    st.markdown(
        f"Se detectaron **{len(detected_signals)}** señal(es) que requieren atención:"
    )

    for i, s in enumerate(detected_signals, start=1):
        with st.expander(f"🔹 Señal {i}: {s['senal']}", expanded=(i <= 2)):
            st.markdown("##### 📌 Justificación")
            st.info(s["justificacion"])

            st.markdown("##### 📊 Interpretación")
            st.warning(s["interpretacion"])

            st.markdown("---")

st.markdown("---")
st.caption(
    "⚠️ Nota: Este módulo no prueba aleatoriedad formal porque no se dispone del universo completo elegible."
)
