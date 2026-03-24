"""
Pagina 13 – Comparacion de Quintiles: Colegio vs UDLA
Compara quintiles de ingreso entre el Colegio (Innova) y UDLA usando
rangos especificos de cada institucion. Muestra que quintil UDLA
se parece mas a cada quintil del Colegio segun categorias como
Deuda, Vulnerabilidad y Ubicacion.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import streamlit as st

from utils.excel_loader import load_excel_sheet
from utils.udla_sql import cargar_datos_udla
from utils.comparacion_helpers import (
    load_ubicacion_periodo,
    cargar_parroquias,
    norm_id,
    parse_valor_deuda,
    build_familias,
    salario_por_id,
    deuda_por_id,
    hogares_salario_deuda,
    calcular_vulnerabilidad,
    asignar_parroquia,
    parroquia_dist,
    calcular_similitud,
)

# ─── Rangos de quintiles por institucion (mismos de pagina 11) ────────────────

QUINTILES_INNOVA = {
    1: {"min": 470.00, "max": 482.00},
    2: {"min": 482.01, "max": 707.07},
    3: {"min": 707.08, "max": 1104.36},
    4: {"min": 1104.37, "max": 1856.00},
    5: {"min": 1856.01, "max": 5011.00},
}

QUINTILES_UDLA = {
    1: {"min": 105.75, "max": 482.00},
    2: {"min": 482.01, "max": 850.92},
    3: {"min": 850.93, "max": 1542.92},
    4: {"min": 1542.93, "max": 2525.00},
    5: {"min": 2525.01, "max": 15392.53},
}

QUINTIL_ORDER = ["Sin empleo", "1", "2", "3", "4", "5"]


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _asignar_quintil_custom(salario: float, rangos: dict) -> str:
    """Asigna quintil usando rangos personalizados por institucion."""
    if pd.isna(salario) or float(salario) <= 0:
        return "Sin empleo"
    val = float(salario)
    q_min = rangos[1]["min"]
    q_max = rangos[5]["max"]
    if val < q_min:
        return "1"
    if val > q_max:
        return "5"
    for q in [1, 2, 3, 4, 5]:
        r = rangos[q]
        if r["min"] <= val <= r["max"]:
            return str(q)
    return "Sin empleo"


def _fmt_money(v: float) -> str:
    return f"${v:,.2f}"


def _rango_text(quintil: str, rangos: dict) -> str:
    if quintil == "Sin empleo":
        return "Sin ingreso laboral"
    q = int(quintil)
    r = rangos[q]
    return f"{_fmt_money(r['min'])} – {_fmt_money(r['max'])}"


def _quintil_display(q: str, institucion: str = "") -> str:
    """Nombre legible de un quintil."""
    if q == "Sin empleo":
        return f"Sin empleo{' ' + institucion if institucion else ''}"
    return f"Quintil {q}{' ' + institucion if institucion else ''}"


def _build_qcomp_feature_vector(
    deuda_avg: float,
    deuda_pct: float,
    vulnerable_pct: float,
    riesgo_pct: float,
    loc_dist: list[float] | None,
) -> dict[str, float]:
    """Vector de caracteristicas sin dimension de quintiles (el quintil ES la agrupacion)."""
    features: dict[str, float] = {}
    features["deuda_avg"] = float(deuda_avg)
    features["deuda_pct"] = float(deuda_pct)
    features["vulnerable_pct"] = float(vulnerable_pct)
    features["riesgo_pct"] = float(riesgo_pct)
    if loc_dist is not None:
        for i, val in enumerate(loc_dist):
            features[f"loc_{i}"] = float(val)
    return features


# ─── Configuración de página ──────────────────────────────────────────────────

st.set_page_config(page_title="Comparacion Quintiles", page_icon="⚖️", layout="wide")

# ─── CSS global ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Tarjetas del ranking ── */
    .card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fb 100%);
        border: 1px solid #e0e4ea;
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s, transform 0.15s;
    }
    .card:hover {
        box-shadow: 0 6px 20px rgba(0,0,0,0.09);
        transform: translateY(-2px);
    }
    .card-rank {
        font-size: 13px; font-weight: 700; color: #6c63ff;
        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px;
    }
    .card-title {
        font-size: 18px; font-weight: 700; color: #1a1a2e;
        margin-bottom: 6px; line-height: 1.25;
    }
    .card-students { font-size: 13px; color: #888; margin-bottom: 14px; }
    .score-bar-bg {
        background: #eef0f4; border-radius: 8px;
        height: 10px; width: 100%; overflow: hidden; margin-top: 4px;
    }
    .score-bar-fill {
        height: 100%; border-radius: 8px; transition: width 0.4s ease;
    }
    .score-label { font-size: 26px; font-weight: 800; margin-right: 6px; }
    .score-max   { font-size: 13px; color: #aaa; }
    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 11px; font-weight: 600; margin-right: 6px; margin-top: 8px;
    }
    .badge-deuda   { background: #fef3c7; color: #b45309; }
    .badge-vuln    { background: #fee2e2; color: #dc2626; }
    .badge-ubic    { background: #d1fae5; color: #047857; }

    /* ── Header principal ── */
    .header-card {
        background: linear-gradient(135deg, #6c63ff 0%, #48b1ff 100%);
        border-radius: 16px; padding: 28px 32px;
        color: white; margin-bottom: 24px;
    }
    .header-card h1 { font-size: 28px; font-weight: 800; margin: 0 0 4px 0; }
    .header-card p  { font-size: 14px; opacity: 0.9; margin: 0; }

    /* ── Tarjetas de categoría (detalle) ── */
    .cat-card {
        background: #ffffff; border: 1px solid #e0e4ea;
        border-radius: 14px; padding: 24px 28px;
        margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .cat-card-header {
        display: flex; align-items: center; margin-bottom: 16px;
    }
    .cat-icon {
        width: 44px; height: 44px; border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 22px; margin-right: 14px; flex-shrink: 0;
    }
    .cat-title    { font-size: 18px; font-weight: 700; color: #1a1a2e; }
    .cat-subtitle { font-size: 12px; color: #888; }

    .metric-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 12px; }
    .metric-box {
        flex: 1; min-width: 160px; background: #f8f9fb;
        border-radius: 12px; padding: 16px 20px; text-align: center;
    }
    .metric-label {
        font-size: 11px; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.5px; color: #888; margin-bottom: 4px;
    }
    .metric-value { font-size: 24px; font-weight: 800; line-height: 1.2; }
    .metric-sub   { font-size: 11px; color: #aaa; margin-top: 2px; }

    .bar-row { display: flex; align-items: center; margin-bottom: 10px; }
    .bar-label {
        width: 180px; font-size: 13px; font-weight: 600;
        color: #4a5568; flex-shrink: 0; text-align: right; padding-right: 12px;
    }
    .bar-track {
        flex: 1; height: 18px; background: #eef0f4;
        border-radius: 9px; overflow: hidden; position: relative;
    }
    .bar-fill { height: 100%; border-radius: 9px; transition: width 0.4s ease; }
    .bar-pct {
        width: 60px; font-size: 12px; font-weight: 700;
        text-align: right; padding-left: 8px; flex-shrink: 0;
    }

    .legend-row { display: flex; gap: 20px; margin-bottom: 16px; }
    .legend-item { display: flex; align-items: center; font-size: 13px; color: #4a5568; }
    .legend-dot {
        width: 12px; height: 12px; border-radius: 50%;
        margin-right: 6px; flex-shrink: 0;
    }

    .grupo-header {
        background: #f8f9fb; border: 1px solid #e0e4ea;
        border-radius: 14px; padding: 18px 24px; margin-bottom: 20px;
        display: flex; align-items: center; gap: 16px;
    }
    .grupo-name  { font-size: 22px; font-weight: 800; color: #1a1a2e; }
    .grupo-badge {
        background: #6c63ff; color: white; border-radius: 20px;
        padding: 4px 14px; font-size: 13px; font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Header ──────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="header-card">
        <h1>⚖️ Comparacion de Quintiles: Colegio vs UDLA</h1>
        <p>Selecciona un quintil del Colegio como referencia y descubre qué quintil
        UDLA tiene el perfil socioeconómico más similar según deuda, vulnerabilidad
        y ubicación.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─── Selector de año/mes reutilizable ────────────────────────────────────────


def _select_anio_mes(
    df: pd.DataFrame,
    anio_col: str,
    mes_col: str,
    label: str,
    default_anio: int | None = None,
    default_mes: int | None = None,
) -> tuple[int | None, int | None]:
    if df.empty or anio_col not in df.columns or mes_col not in df.columns:
        return default_anio, default_mes
    anios = sorted(
        pd.to_numeric(df[anio_col], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if not anios:
        return default_anio, default_mes
    anio_default = default_anio if default_anio in anios else anios[-1]
    anio = st.selectbox(
        f"Año – {label}",
        options=anios,
        index=anios.index(anio_default),
        key=f"q13_anio_{label}",
    )
    df_anio = df[pd.to_numeric(df[anio_col], errors="coerce").astype("Int64") == anio]
    meses = sorted(
        pd.to_numeric(df_anio[mes_col], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if not meses:
        return anio, default_mes
    mes_default = default_mes if default_mes in meses else meses[-1]
    mes = st.selectbox(
        f"Mes – {label}",
        options=meses,
        index=meses.index(mes_default),
        key=f"q13_mes_{label}",
    )
    return anio, mes


def _norm_period(value) -> str:
    """Normaliza valores de periodo/semestre a formato digitos (ej: 202520)."""
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if s == "":
        return ""
    try:
        f = float(s)
        if np.isfinite(f):
            return str(int(f))
    except Exception:
        pass
    return re.sub(r"\D", "", s)


# ─── Carga de datos ──────────────────────────────────────────────────────────

with st.spinner("Cargando datos del colegio …"):
    estudiantes = load_excel_sheet("Estudiantes")
    universo_familiares = load_excel_sheet("Universo Familiares")
    empleo = load_excel_sheet("Empleos")
    deudas = load_excel_sheet("Deudas")
    info_personal = load_excel_sheet("Informacion Personal")
    if "Cedula" in estudiantes.columns:
        estudiantes = estudiantes.rename(columns={"Cedula": "IDENTIFICACION"})
    elif "CEDULA" in estudiantes.columns:
        estudiantes = estudiantes.rename(columns={"CEDULA": "IDENTIFICACION"})

with st.spinner("Conectando con SQL UDLA …"):
    udla = cargar_datos_udla()

personas_udla = udla.get("Personas", pd.DataFrame())
familiares_udla = udla.get("Familiares", pd.DataFrame())
ingresos_udla = udla.get("Ingresos", pd.DataFrame())
deudas_udla = udla.get("Deudas", pd.DataFrame())
ubicacion_udla = load_ubicacion_periodo()

if personas_udla.empty or familiares_udla.empty:
    st.info("No hay datos suficientes de UDLA para comparar.")
    st.stop()

# ─── Filtros principales ─────────────────────────────────────────────────────

st.markdown("#### ⚙️ Filtros principales")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    quintil_ref_opts = ["Todos"] + QUINTIL_ORDER
    quintil_ref_sel = st.selectbox(
        "Quintil Colegio (referencia)",
        options=quintil_ref_opts,
        format_func=lambda x: (
            "Todos los hogares" if x == "Todos" else _quintil_display(x, "Colegio")
        ),
        index=0,
    )

with col_f2:
    grupo_udla = st.selectbox(
        "Grupo UDLA",
        options=["E", "A", "G"],
        format_func=lambda x: {
            "E": "Enrollment",
            "A": "Afluentes",
            "G": "Graduados",
        }[x],
        index=0,
    )

with col_f3:
    if "periodo" in personas_udla.columns:
        personas_periodo = personas_udla.copy()
        if "tipo" in personas_periodo.columns:
            personas_periodo = personas_periodo[personas_periodo["tipo"] == grupo_udla]
        periodos = (
            personas_periodo["periodo"]
            .map(_norm_period)
            .replace("", pd.NA)
            .dropna()
            .unique()
            .tolist()
        )
        periodos = sorted(periodos)
        opts_periodo = ["Todos"] + periodos if periodos else ["Todos"]
        periodo_sel = st.selectbox(
            "Periodo UDLA", options=opts_periodo, index=0, key="q13_periodo"
        )
    else:
        periodo_sel = "Todos"

with st.expander("📅 Ajustes de periodo", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        anio_emp, mes_emp = _select_anio_mes(empleo, "ANIO", "MES", "Empleos (Colegio)")
    with c2:
        anio_deu, mes_deu = _select_anio_mes(deudas, "ANIO", "MES", "Deudas (Colegio)")
    c3, c4 = st.columns(2)
    with c3:
        anio_ing_udla, mes_ing_udla = _select_anio_mes(
            ingresos_udla, "anio", "mes", "Ingresos (UDLA)"
        )
    with c4:
        anio_deu_udla, mes_deu_udla = _select_anio_mes(
            deudas_udla, "anio", "mes", "Deudas (UDLA)"
        )

for label_check, val in [
    ("empleos del colegio", (anio_emp, mes_emp)),
    ("deudas del colegio", (anio_deu, mes_deu)),
    ("ingresos UDLA", (anio_ing_udla, mes_ing_udla)),
    ("deudas UDLA", (anio_deu_udla, mes_deu_udla)),
]:
    if val[0] is None or val[1] is None:
        st.info(f"No hay periodo de {label_check}.")
        st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# Perfil Colegio – hogares con quintiles personalizados (Innova)
# ═══════════════════════════════════════════════════════════════════════════════

estudiantes["IDENTIFICACION"] = norm_id(estudiantes["IDENTIFICACION"])
ids_estudiantes = set(estudiantes["IDENTIFICACION"].unique().tolist())
universo_familiares["IDENTIFICACION"] = norm_id(universo_familiares["IDENTIFICACION"])

familias_col, mapa_col = build_familias(
    estudiantes[["IDENTIFICACION"]].copy(),
    universo_familiares,
    id_col="IDENTIFICACION",
    padre_col="CED_PADRE",
    madre_col="CED_MADRE",
)

emp_col = empleo[(empleo["ANIO"] == anio_emp) & (empleo["MES"] == mes_emp)].copy()
emp_col["IDENTIFICACION"] = norm_id(emp_col["IDENTIFICACION"])
emp_col["SALARIO"] = pd.to_numeric(emp_col["SALARIO"], errors="coerce").fillna(0)

deu_col = deudas[(deudas["ANIO"] == anio_deu) & (deudas["MES"] == mes_deu)].copy()
deu_col["IDENTIFICACION"] = norm_id(deu_col["IDENTIFICACION"])
deu_col["VALOR"] = parse_valor_deuda(deu_col["VALOR"])

salario_map_col = salario_por_id(emp_col, "IDENTIFICACION", "SALARIO")
deuda_map_col = deuda_por_id(deu_col, "IDENTIFICACION", "VALOR")
hogares_col = hogares_salario_deuda(mapa_col, salario_map_col, deuda_map_col)

if hogares_col.empty:
    st.info("No hay hogares del colegio con datos suficientes.")
    st.stop()

# Asignar quintil personalizado Innova
hogares_col["quintil_custom"] = hogares_col["salario"].apply(
    lambda x: _asignar_quintil_custom(x, QUINTILES_INNOVA)
)

# Mapa estudiante → quintil (para ubicacion)
est_quintil_map_col: dict[str, str] = {}
if not familias_col.empty and "hogar_id" in familias_col.columns:
    _est_hog_col = familias_col[["IDENTIFICACION", "hogar_id"]].drop_duplicates(
        "IDENTIFICACION"
    )
    _est_quintil_col = _est_hog_col.merge(
        hogares_col[["hogar_id", "quintil_custom"]].drop_duplicates("hogar_id"),
        on="hogar_id",
        how="left",
    )
    est_quintil_map_col = dict(
        zip(
            _est_quintil_col["IDENTIFICACION"],
            _est_quintil_col["quintil_custom"],
        )
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Perfil UDLA – hogares con quintiles personalizados (UDLA)
# ═══════════════════════════════════════════════════════════════════════════════

personas_udla = personas_udla.copy()
personas_udla = personas_udla[personas_udla["tipo"] == grupo_udla]
periodo_norm_sel = _norm_period(periodo_sel) if periodo_sel != "Todos" else ""
if periodo_sel != "Todos" and "periodo" in personas_udla.columns:
    personas_udla = personas_udla[
        personas_udla["periodo"].map(_norm_period) == periodo_norm_sel
    ]

if personas_udla.empty:
    st.info(
        f"No hay registros UDLA para grupo `{grupo_udla}` en el periodo `{periodo_sel}`."
    )
    st.stop()

familias_udla, mapa_udla = build_familias(
    personas_udla[["identificacion"]].copy(),
    familiares_udla,
    id_col="identificacion",
    padre_col="ced_padre",
    madre_col="ced_madre",
)

if familias_udla.empty:
    st.warning(
        "No hay datos familiares para este periodo/grupo UDLA. "
        "El ranking se calculará con las categorías disponibles."
    )

ing_udla = ingresos_udla[
    (ingresos_udla["anio"] == anio_ing_udla) & (ingresos_udla["mes"] == mes_ing_udla)
].copy()
ing_udla["identificacion"] = norm_id(ing_udla["identificacion"])

deu_udla = deudas_udla[
    (deudas_udla["anio"] == anio_deu_udla) & (deudas_udla["mes"] == mes_deu_udla)
].copy()
deu_udla["identificacion"] = norm_id(deu_udla["identificacion"])

salario_map_udla = salario_por_id(ing_udla, "identificacion", "salario")
deuda_map_udla = deuda_por_id(deu_udla, "identificacion", "valor")
hogares_udla = hogares_salario_deuda(mapa_udla, salario_map_udla, deuda_map_udla)

if hogares_udla.empty:
    st.info("No hay hogares UDLA con datos suficientes.")
    st.stop()

# Asignar quintil personalizado UDLA
hogares_udla["quintil_custom"] = hogares_udla["salario"].apply(
    lambda x: _asignar_quintil_custom(x, QUINTILES_UDLA)
)

# Mapa estudiante → quintil UDLA (para ubicacion)
est_quintil_map_udla: dict[str, str] = {}
if not familias_udla.empty and "hogar_id" in familias_udla.columns:
    _est_hog_udla = familias_udla[["identificacion", "hogar_id"]].drop_duplicates(
        "identificacion"
    )
    _est_quintil_udla = _est_hog_udla.merge(
        hogares_udla[["hogar_id", "quintil_custom"]].drop_duplicates("hogar_id"),
        on="hogar_id",
        how="left",
    )
    est_quintil_map_udla = dict(
        zip(
            _est_quintil_udla["identificacion"],
            _est_quintil_udla["quintil_custom"],
        )
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Ubicacion – parroquias
# ═══════════════════════════════════════════════════════════════════════════════

gdf_parroquias = cargar_parroquias()

# Innova: parroquias por estudiante
info_personal["IDENTIFICACION"] = norm_id(info_personal["IDENTIFICACION"])
loc_df_col = info_personal[info_personal["IDENTIFICACION"].isin(ids_estudiantes)].copy()
df_parro_col = pd.DataFrame()
if not gdf_parroquias.empty:
    df_parro_col = asignar_parroquia(loc_df_col, gdf_parroquias, "LATITUD", "LONGITUD")
    if "parroquia" in df_parro_col.columns:
        df_parro_col["quintil_custom"] = df_parro_col["IDENTIFICACION"].map(
            est_quintil_map_col
        )

# UDLA: parroquias desde CSV
# El CSV usa codigos de estudiante (A00000456) mientras que personas_udla usa
# cedulas, por lo que no se pueden cruzar por ID directamente.
# Se usa la columna 'carrera' como puente: para cada carrera conocemos la
# proporcion de estudiantes en cada quintil (via datos familiares/ingresos),
# y esa proporcion se aplica a los estudiantes del CSV.
ub_parro = None
carrera_q_peso: dict[tuple[str, str], float] = {}  # (carrera, quintil) → proporcion

if (
    not ubicacion_udla.empty
    and not gdf_parroquias.empty
    and {"id", "semestre", "latitud", "longitud"}.issubset(set(ubicacion_udla.columns))
):
    ub = ubicacion_udla.copy()
    ub["id"] = ub["id"].astype(str).str.strip()
    ub["semestre_norm"] = ub["semestre"].map(_norm_period)
    # Siempre usar semestre 202520 para ubicaciones UDLA
    ub = ub[ub["semestre_norm"] == "202520"]
    # Normalizar carrera en el CSV
    if "carrera" in ub.columns:
        ub["carrera_norm"] = ub["carrera"].astype(str).str.strip().str.upper()
    if not ub.empty:
        ub_parro = asignar_parroquia(ub, gdf_parroquias, "latitud", "longitud")

    # Construir puente carrera → quintil usando personas_udla con datos familiares
    if est_quintil_map_udla and "carrera" in personas_udla.columns:
        _pq = personas_udla[["identificacion", "carrera"]].copy()
        _pq["carrera"] = _pq["carrera"].astype(str).str.strip().str.upper()
        _pq["quintil_custom"] = _pq["identificacion"].map(est_quintil_map_udla)
        _pq = _pq.dropna(subset=["quintil_custom"])
        if not _pq.empty:
            _cq = (
                _pq.groupby(["carrera", "quintil_custom"]).size().reset_index(name="n")
            )
            _ct = _pq.groupby("carrera").size().reset_index(name="total")
            _cq = _cq.merge(_ct, on="carrera")
            _cq["peso"] = _cq["n"] / _cq["total"]
            carrera_q_peso = dict(
                zip(
                    zip(_cq["carrera"], _cq["quintil_custom"]),
                    _cq["peso"],
                )
            )

# Categorias de parroquia (top 12 + Otros)
loc_categorias = None
all_parro = []
if "parroquia" in df_parro_col.columns:
    all_parro.append(df_parro_col["parroquia"].dropna())
if ub_parro is not None and "parroquia" in ub_parro.columns:
    all_parro.append(ub_parro["parroquia"].dropna())
if all_parro:
    combined = pd.concat(all_parro)
    if not combined.empty:
        top = combined.value_counts().head(12).index.tolist()
        loc_categorias = list(top) + ["Otros"]

# ═══════════════════════════════════════════════════════════════════════════════
# Construir perfiles por quintil – Colegio (Innova)
# ═══════════════════════════════════════════════════════════════════════════════

detalles_col_q: dict[str, dict] = {}

for q_label in QUINTIL_ORDER:
    hog_q = hogares_col[hogares_col["quintil_custom"] == q_label]
    n_hogares = int(hog_q["hogar_id"].nunique()) if not hog_q.empty else 0
    deuda_avg_q = float(hog_q["deuda"].mean()) if not hog_q.empty else 0.0
    deuda_pct_q = float((hog_q["deuda"] > 0).mean()) if not hog_q.empty else 0.0

    # Vulnerabilidad
    hog_ids_q = set(hog_q["hogar_id"].unique()) if not hog_q.empty else set()
    fam_q = (
        familias_col[familias_col["hogar_id"].isin(hog_ids_q)]
        if hog_ids_q
        else familias_col.iloc[0:0]
    )
    vuln_q = (
        calcular_vulnerabilidad(
            fam_q,
            id_col="IDENTIFICACION",
            padre_col="CED_PADRE",
            madre_col="CED_MADRE",
            ingresos_df=emp_col,
            ingresos_id_col="IDENTIFICACION",
            salario_col="SALARIO",
            deudas_df=deu_col,
            deudas_id_col="IDENTIFICACION",
            valor_col="VALOR",
            calif_col="COD_CALIFICACION",
        )
        if not fam_q.empty
        else pd.DataFrame()
    )
    vuln_pct_q = float(vuln_q["vulnerable"].mean()) if not vuln_q.empty else 0.0
    riesgo_pct_q = float(vuln_q["en_riesgo"].mean()) if not vuln_q.empty else 0.0
    vuln_total_q = int(len(vuln_q)) if not vuln_q.empty else 0
    vuln_count_q = int(vuln_q["vulnerable"].sum()) if not vuln_q.empty else 0
    riesgo_count_q = int(vuln_q["en_riesgo"].sum()) if not vuln_q.empty else 0

    # Ubicacion
    loc_dist_q = None
    loc_count_q = 0
    if (
        loc_categorias
        and "parroquia" in df_parro_col.columns
        and "quintil_custom" in df_parro_col.columns
    ):
        parro_q = df_parro_col.loc[
            df_parro_col["quintil_custom"] == q_label, "parroquia"
        ].dropna()
        if not parro_q.empty:
            loc_dist_q = parroquia_dist(parro_q, loc_categorias)
            loc_count_q = int(parro_q.shape[0])

    salario_avg_q = float(hog_q["salario"].mean()) if not hog_q.empty else 0.0

    detalles_col_q[q_label] = {
        "n_hogares": n_hogares,
        "deuda_avg": deuda_avg_q,
        "deuda_pct": deuda_pct_q,
        "vulnerable_pct": vuln_pct_q,
        "riesgo_pct": riesgo_pct_q,
        "vuln_total": vuln_total_q,
        "vuln_count": vuln_count_q,
        "riesgo_count": riesgo_count_q,
        "loc_dist": loc_dist_q,
        "loc_count": loc_count_q,
        "salario_avg": salario_avg_q,
    }

# ─── Perfil de referencia (quintil seleccionado o Todos) ─────────────────────

if quintil_ref_sel == "Todos":
    _ref: dict = {
        "n_hogares": int(hogares_col["hogar_id"].nunique()),
        "deuda_avg": float(hogares_col["deuda"].mean()),
        "deuda_pct": float((hogares_col["deuda"] > 0).mean()),
    }
    _vuln_all = (
        calcular_vulnerabilidad(
            familias_col,
            id_col="IDENTIFICACION",
            padre_col="CED_PADRE",
            madre_col="CED_MADRE",
            ingresos_df=emp_col,
            ingresos_id_col="IDENTIFICACION",
            salario_col="SALARIO",
            deudas_df=deu_col,
            deudas_id_col="IDENTIFICACION",
            valor_col="VALOR",
            calif_col="COD_CALIFICACION",
        )
        if not familias_col.empty
        else pd.DataFrame()
    )
    _ref["vulnerable_pct"] = (
        float(_vuln_all["vulnerable"].mean()) if not _vuln_all.empty else 0.0
    )
    _ref["riesgo_pct"] = (
        float(_vuln_all["en_riesgo"].mean()) if not _vuln_all.empty else 0.0
    )
    _ref["vuln_total"] = int(len(_vuln_all)) if not _vuln_all.empty else 0
    _ref["vuln_count"] = (
        int(_vuln_all["vulnerable"].sum()) if not _vuln_all.empty else 0
    )
    _ref["riesgo_count"] = (
        int(_vuln_all["en_riesgo"].sum()) if not _vuln_all.empty else 0
    )
    _loc_all = None
    _loc_count_all = 0
    if loc_categorias and "parroquia" in df_parro_col.columns:
        _ps = df_parro_col["parroquia"].dropna()
        if not _ps.empty:
            _loc_all = parroquia_dist(_ps, loc_categorias)
            _loc_count_all = int(_ps.shape[0])
    _ref["loc_dist"] = _loc_all
    _ref["loc_count"] = _loc_count_all
    _ref["salario_avg"] = float(hogares_col["salario"].mean())
    ref_profile = _ref
else:
    ref_profile = detalles_col_q.get(quintil_ref_sel, {})

ref_features = _build_qcomp_feature_vector(
    ref_profile.get("deuda_avg", 0.0),
    ref_profile.get("deuda_pct", 0.0),
    ref_profile.get("vulnerable_pct", 0.0),
    ref_profile.get("riesgo_pct", 0.0),
    ref_profile.get("loc_dist"),
)

# ═══════════════════════════════════════════════════════════════════════════════
# Construir perfiles por quintil – UDLA
# ═══════════════════════════════════════════════════════════════════════════════

perfiles_udla: list[dict] = []
detalles_udla_q: dict[str, dict] = {}

for q_label in QUINTIL_ORDER:
    hog_q = hogares_udla[hogares_udla["quintil_custom"] == q_label]
    n_hogares = int(hog_q["hogar_id"].nunique()) if not hog_q.empty else 0
    if n_hogares == 0:
        continue

    deuda_avg_q = float(hog_q["deuda"].mean())
    deuda_pct_q = float((hog_q["deuda"] > 0).mean())

    # Vulnerabilidad
    hog_ids_q = set(hog_q["hogar_id"].unique())
    fam_q = (
        familias_udla[familias_udla["hogar_id"].isin(hog_ids_q)]
        if not familias_udla.empty
        else familias_udla.iloc[0:0]
    )
    vuln_q = (
        calcular_vulnerabilidad(
            fam_q,
            id_col="identificacion",
            padre_col="ced_padre",
            madre_col="ced_madre",
            ingresos_df=ing_udla,
            ingresos_id_col="identificacion",
            salario_col="salario",
            deudas_df=deu_udla,
            deudas_id_col="identificacion",
            valor_col="valor",
            calif_col="cod_calificacion",
        )
        if not fam_q.empty
        else pd.DataFrame()
    )
    vuln_pct_q = float(vuln_q["vulnerable"].mean()) if not vuln_q.empty else 0.0
    riesgo_pct_q = float(vuln_q["en_riesgo"].mean()) if not vuln_q.empty else 0.0
    vuln_total_q = int(len(vuln_q)) if not vuln_q.empty else 0
    vuln_count_q = int(vuln_q["vulnerable"].sum()) if not vuln_q.empty else 0
    riesgo_count_q = int(vuln_q["en_riesgo"].sum()) if not vuln_q.empty else 0

    # Ubicacion (ponderada por carrera → quintil)
    loc_dist_q = None
    loc_count_q = 0
    if (
        loc_categorias
        and ub_parro is not None
        and "parroquia" in ub_parro.columns
        and "carrera_norm" in ub_parro.columns
        and carrera_q_peso
    ):
        _ub = (
            ub_parro[["parroquia", "carrera_norm"]].dropna(subset=["parroquia"]).copy()
        )
        _ub["peso"] = _ub["carrera_norm"].apply(
            lambda c: carrera_q_peso.get((c, q_label), 0.0)
        )
        _ub = _ub[_ub["peso"] > 0]
        if not _ub.empty:
            parro_w = _ub.groupby("parroquia")["peso"].sum()
            total_w = parro_w.sum()
            if total_w > 0:
                dist = [
                    float(parro_w.get(cat, 0.0) / total_w)
                    for cat in loc_categorias[:-1]
                ]
                otros = max(0.0, 1.0 - sum(dist))
                dist.append(otros)
                loc_dist_q = dist
                loc_count_q = int(round(total_w))

    salario_avg_q = float(hog_q["salario"].mean()) if not hog_q.empty else 0.0

    features = _build_qcomp_feature_vector(
        deuda_avg_q, deuda_pct_q, vuln_pct_q, riesgo_pct_q, loc_dist_q
    )
    grupo_name = _quintil_display(q_label, "UDLA")
    perfiles_udla.append(
        {"grupo": grupo_name, "total_estudiantes": n_hogares, **features}
    )
    detalles_udla_q[grupo_name] = {
        "q_label": q_label,
        "n_hogares": n_hogares,
        "deuda_avg": deuda_avg_q,
        "deuda_pct": deuda_pct_q,
        "vulnerable_pct": vuln_pct_q,
        "riesgo_pct": riesgo_pct_q,
        "vuln_total": vuln_total_q,
        "vuln_count": vuln_count_q,
        "riesgo_count": riesgo_count_q,
        "loc_dist": loc_dist_q,
        "loc_count": loc_count_q,
        "salario_avg": salario_avg_q,
    }

if not perfiles_udla:
    st.info("No hay quintiles UDLA con datos suficientes.")
    st.stop()

df_perfiles = pd.DataFrame(perfiles_udla)
feature_cols = [
    c for c in df_perfiles.columns if c not in ["grupo", "total_estudiantes"]
]

# ─── Pesos por categoría ─────────────────────────────────────────────────────

with st.expander("⚖️ Categorías y pesos (total 100)", expanded=False):
    categorias_disp = ["Deuda", "Vulnerabilidad"]
    has_loc = loc_categorias is not None and ref_profile.get("loc_dist") is not None
    if has_loc:
        categorias_disp.append("Ubicacion (parroquia)")
    else:
        st.caption("ℹ️ Ubicacion (parroquia) no disponible por falta de datos.")

    categorias_sel = st.multiselect(
        "Categorias a usar",
        options=categorias_disp,
        default=categorias_disp,
        key="q13_cats",
    )
    categorias_sel = [c for c in categorias_sel if c in categorias_disp]
    if not categorias_sel:
        categorias_sel = categorias_disp.copy()
        st.warning(
            "No había categorías seleccionadas; se aplicaron automáticamente las disponibles."
        )

    n_sel = len(categorias_sel)
    base = 100 // n_sel if n_sel > 0 else 0
    resto = 100 - base * n_sel

    pesos: dict[str, int] = {}
    cols_peso = st.columns(n_sel)
    for i, cat in enumerate(categorias_sel):
        with cols_peso[i]:
            pesos[cat] = st.slider(
                f"Peso: {cat}",
                0,
                100,
                base + (1 if i < resto else 0),
                1,
                key=f"q13_peso_{cat}",
            )

    total_pesos = sum(pesos.values())
    if total_pesos != 100:
        st.warning(f"Los pesos suman **{total_pesos}** — deben sumar **100**.")
        st.stop()

    w_deuda = pesos.get("Deuda", 0) / 100.0 if "Deuda" in categorias_sel else 0.0
    w_vuln = (
        pesos.get("Vulnerabilidad", 0) / 100.0
        if "Vulnerabilidad" in categorias_sel
        else 0.0
    )
    w_loc = (
        pesos.get("Ubicacion (parroquia)", 0) / 100.0
        if "Ubicacion (parroquia)" in categorias_sel
        else 0.0
    )

weights: dict[str, float] = {}
for col in feature_cols:
    if col.startswith("loc_"):
        weights[col] = w_loc
    elif col in {"deuda_avg", "deuda_pct"}:
        weights[col] = w_deuda
    else:
        weights[col] = w_vuln

# ═══════════════════════════════════════════════════════════════════════════════
# Cálculo de similitud
# ═══════════════════════════════════════════════════════════════════════════════

df_result = calcular_similitud(
    df_perfiles,
    ref_features,
    feature_cols,
    weights,
    categorias_sel,
    {
        "Deuda": w_deuda,
        "Vulnerabilidad": w_vuln,
        "Ubicacion (parroquia)": w_loc,
    },
)

if "sim_ubicacion" in df_result.columns:
    sin_ubicacion = df_result["grupo"].apply(
        lambda g: detalles_udla_q.get(g, {}).get("loc_dist") is None
    )
    df_result.loc[sin_ubicacion, "sim_ubicacion"] = np.nan

# ═══════════════════════════════════════════════════════════════════════════════
# ██  DOS PESTAÑAS  ██
# ═══════════════════════════════════════════════════════════════════════════════

ref_label = (
    "Todos los hogares del Colegio"
    if quintil_ref_sel == "Todos"
    else _quintil_display(quintil_ref_sel, "Colegio")
)

tab_ranking, tab_detalle = st.tabs(
    ["🏆 Ranking de Quintiles Similares", "📊 Detalle por Categoría"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: RANKING
# ─────────────────────────────────────────────────────────────────────────────

with tab_ranking:
    st.markdown("---")
    st.markdown(
        f'<div style="text-align:center; margin-bottom:8px;">'
        f'<span style="font-size:22px; font-weight:800; color:#1a1a2e;">'
        f"🏆 Quintiles UDLA más similares a {ref_label}</span></div>",
        unsafe_allow_html=True,
    )

    def _score_color(score: float) -> str:
        if score >= 70:
            return "#22c55e"
        if score >= 50:
            return "#84cc16"
        if score >= 35:
            return "#eab308"
        if score >= 20:
            return "#f97316"
        return "#ef4444"

    def _render_card(row: pd.Series, rank: int) -> str:
        score = float(row["puntaje_similitud"])
        color = _score_color(score)
        pct = min(score, 100)
        grupo_name = row["grupo"]
        det = detalles_udla_q.get(grupo_name, {})
        q_label = det.get("q_label", "")
        rango = _rango_text(q_label, QUINTILES_UDLA) if q_label else ""

        badges = ""
        if "sim_deuda" in row.index and pd.notna(row.get("sim_deuda")):
            badges += (
                f'<span class="badge badge-deuda">Deuda: {row["sim_deuda"]:.1f}</span>'
            )
        if "sim_vulnerabilidad" in row.index and pd.notna(
            row.get("sim_vulnerabilidad")
        ):
            badges += f'<span class="badge badge-vuln">Vulnerabilidad: {row["sim_vulnerabilidad"]:.1f}</span>'
        if "sim_ubicacion" in row.index:
            sim_u = row.get("sim_ubicacion")
            sim_u_val = 0.0 if pd.isna(sim_u) else float(sim_u)
            badges += (
                f'<span class="badge badge-ubic">Ubicacion: {sim_u_val:.1f}</span>'
            )

        rango_html = (
            f'<div style="font-size:12px;color:#6c63ff;margin-bottom:4px;">Rango: {rango}</div>'
            if rango
            else ""
        )
        return f"""
        <div class="card">
            <div class="card-rank">#{rank}</div>
            <div class="card-title">{grupo_name}</div>
            {rango_html}
            <div class="card-students">🏠 {int(row['total_estudiantes']):,} hogares</div>
            <div style="display:flex; align-items:baseline;">
                <span class="score-label" style="color:{color};">{score:.1f}</span>
                <span class="score-max">/ 100</span>
            </div>
            <div class="score-bar-bg">
                <div class="score-bar-fill" style="width:{pct}%; background:{color};"></div>
            </div>
            <div>{badges}</div>
        </div>
        """

    rows = df_result.reset_index(drop=True)
    col_left, col_right = st.columns(2)
    for i, (_, row) in enumerate(rows.iterrows()):
        html = _render_card(row, rank=i + 1)
        if i % 2 == 0:
            with col_left:
                st.markdown(html, unsafe_allow_html=True)
        else:
            with col_right:
                st.markdown(html, unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
        "💡 **¿Cómo se calcula?** Se compara el perfil del quintil de referencia del Colegio "
        "contra cada quintil UDLA usando distancia estandarizada en las categorías seleccionadas. "
        "Los pesos permiten dar mayor importancia a cada categoría."
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: DETALLE POR CATEGORÍA
# ─────────────────────────────────────────────────────────────────────────────

COLOR_COL = "#6c63ff"
COLOR_UDLA = "#48b1ff"

with tab_detalle:
    grupos_disponibles = df_result["grupo"].tolist()
    grupo_detalle = st.selectbox(
        "Selecciona Quintil UDLA para ver detalle",
        options=grupos_disponibles,
        index=0,
        key="q13_sel_quintil_detalle",
    )
    detalle = detalles_udla_q.get(grupo_detalle, {})
    q_label_det = detalle.get("q_label", "")

    row_sel = df_result[df_result["grupo"] == grupo_detalle]
    score_sel = (
        float(row_sel["puntaje_similitud"].iloc[0]) if not row_sel.empty else 0.0
    )
    n_hog_sel = int(row_sel["total_estudiantes"].iloc[0]) if not row_sel.empty else 0
    rank_sel = (
        grupos_disponibles.index(grupo_detalle) + 1
        if grupo_detalle in grupos_disponibles
        else 0
    )

    rango_udla_det = _rango_text(q_label_det, QUINTILES_UDLA) if q_label_det else ""
    rango_col_det = (
        _rango_text(quintil_ref_sel, QUINTILES_INNOVA)
        if quintil_ref_sel != "Todos"
        else "Todos los rangos"
    )

    html_header = (
        '<div class="grupo-header">'
        "<div>"
        f'<div class="grupo-name">{grupo_detalle}</div>'
        f'<span style="font-size:13px; color:#888;">'
        f"🏠 {n_hog_sel:,} hogares &nbsp;·&nbsp; Rango: {rango_udla_det}"
        f" &nbsp;·&nbsp; Posición #{rank_sel}"
        f"</span>"
        "</div>"
        '<div style="margin-left:auto;">'
        f'<span class="grupo-badge">Similitud: {score_sel:.1f} / 100</span>'
        "</div>"
        "</div>"
    )
    st.markdown(html_header, unsafe_allow_html=True)

    # -- helpers de visualización --

    def _dual_bar(
        label: str,
        val_col: float,
        val_udla: float,
        max_val: float | None = None,
        tooltip_col: str = "",
        tooltip_udla: str = "",
    ) -> str:
        if max_val is None or max_val == 0:
            max_val = max(val_col, val_udla, 1)
        pct_c = min(val_col / max_val * 100, 100) if max_val else 0
        pct_u = min(val_udla / max_val * 100, 100) if max_val else 0
        return (
            f'<div class="bar-row">'
            f'<div class="bar-label">{label}</div>'
            f'<div style="flex:1;">'
            f'<div style="display:flex;align-items:center;margin-bottom:3px;">'
            f'<div class="bar-track" title="{tooltip_col}"><div class="bar-fill" style="width:{pct_c:.1f}%;background:{COLOR_COL};"></div></div>'
            f'<div class="bar-pct" style="color:{COLOR_COL};">{val_col:.1f}%</div>'
            f"</div>"
            f'<div style="display:flex;align-items:center;">'
            f'<div class="bar-track" title="{tooltip_udla}"><div class="bar-fill" style="width:{pct_u:.1f}%;background:{COLOR_UDLA};"></div></div>'
            f'<div class="bar-pct" style="color:{COLOR_UDLA};">{val_udla:.1f}%</div>'
            f"</div>"
            f"</div>"
            f"</div>"
        )

    def _legend() -> str:
        return (
            f'<div class="legend-row">'
            f'<div class="legend-item"><div class="legend-dot" style="background:{COLOR_COL};"></div>{ref_label}</div>'
            f'<div class="legend-item"><div class="legend-dot" style="background:{COLOR_UDLA};"></div>{grupo_detalle}</div>'
            f"</div>"
        )

    def _metric_html(
        label: str, value: str, sub: str = "", color: str = "#1a1a2e"
    ) -> str:
        return (
            f'<div class="metric-box">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value" style="color:{color};">{value}</div>'
            f'<div class="metric-sub">{sub}</div>'
            f"</div>"
        )

    # ── TARJETA: Ingresos (informativa) ──

    sal_avg_ref = ref_profile.get("salario_avg", 0.0)
    sal_avg_udla = detalle.get("salario_avg", 0.0)
    n_hog_ref = ref_profile.get("n_hogares", 0)

    html_ingresos = (
        '<div class="cat-card">'
        '<div class="cat-card-header">'
        '<div class="cat-icon" style="background:#ede9fe;">💵</div>'
        "<div>"
        '<div class="cat-title">Ingresos del Hogar</div>'
        '<div class="cat-subtitle">Contexto salarial de los hogares en cada quintil</div>'
        "</div></div>"
        '<div class="metric-row">'
        + _metric_html("Rango Colegio", rango_col_det, ref_label, COLOR_COL)
        + _metric_html("Rango UDLA", rango_udla_det, grupo_detalle, COLOR_UDLA)
        + "</div>"
        '<div class="metric-row">'
        + _metric_html(
            "Salario promedio", _fmt_money(sal_avg_ref), ref_label, COLOR_COL
        )
        + _metric_html(
            "Salario promedio", _fmt_money(sal_avg_udla), grupo_detalle, COLOR_UDLA
        )
        + "</div>"
        '<div class="metric-row">'
        + _metric_html("Hogares", f"{n_hog_ref:,}", ref_label, COLOR_COL)
        + _metric_html("Hogares", f"{n_hog_sel:,}", grupo_detalle, COLOR_UDLA)
        + "</div>"
        "</div>"
    )
    st.markdown(html_ingresos, unsafe_allow_html=True)

    # ── TARJETA: Deuda ──

    if "Deuda" in categorias_sel:
        d_avg_ref = ref_profile.get("deuda_avg", 0.0)
        d_pct_ref = ref_profile.get("deuda_pct", 0.0)
        d_avg_g = detalle.get("deuda_avg", 0.0)
        d_pct_g = detalle.get("deuda_pct", 0.0)
        sim_d = ""
        if not row_sel.empty and "sim_deuda" in row_sel.columns:
            sim_d = f"{float(row_sel['sim_deuda'].iloc[0]):.1f}"

        sim_d_html = (
            (
                '<div style="margin-left:auto;">'
                + _metric_html("Similitud", sim_d, "de 100", "#b45309")
                + "</div>"
            )
            if sim_d
            else ""
        )
        html_deuda = (
            '<div class="cat-card">'
            '<div class="cat-card-header">'
            '<div class="cat-icon" style="background:#fef3c7;">💰</div>'
            "<div>"
            '<div class="cat-title">Endeudamiento</div>'
            '<div class="cat-subtitle">Nivel de deuda promedio y porcentaje de hogares endeudados</div>'
            "</div>" + sim_d_html + "</div>"
            '<div class="metric-row">'
            + _metric_html("Deuda promedio", f"${d_avg_ref:,.0f}", ref_label, COLOR_COL)
            + _metric_html(
                "Deuda promedio", f"${d_avg_g:,.0f}", grupo_detalle, COLOR_UDLA
            )
            + "</div>"
            '<div class="metric-row">'
            + _metric_html(
                "Hogares con deuda",
                f"{d_pct_ref*100:.1f}%",
                ref_label,
                COLOR_COL,
            )
            + _metric_html(
                "Hogares con deuda",
                f"{d_pct_g*100:.1f}%",
                grupo_detalle,
                COLOR_UDLA,
            )
            + "</div>"
            "</div>"
        )
        st.markdown(html_deuda, unsafe_allow_html=True)

    # ── TARJETA: Vulnerabilidad ──

    if "Vulnerabilidad" in categorias_sel:
        v_pct_ref = ref_profile.get("vulnerable_pct", 0.0)
        r_pct_ref = ref_profile.get("riesgo_pct", 0.0)
        v_pct_g = detalle.get("vulnerable_pct", 0.0)
        r_pct_g = detalle.get("riesgo_pct", 0.0)
        sim_v = ""
        if not row_sel.empty and "sim_vulnerabilidad" in row_sel.columns:
            sim_v = f"{float(row_sel['sim_vulnerabilidad'].iloc[0]):.1f}"

        v_col_pct = v_pct_ref * 100
        v_udla_pct = v_pct_g * 100
        r_col_pct = r_pct_ref * 100
        r_udla_pct = r_pct_g * 100

        # Conteos de personas para tooltips
        vuln_total_ref = ref_profile.get("vuln_total", 0)
        vuln_count_ref = ref_profile.get("vuln_count", 0)
        riesgo_count_ref = ref_profile.get("riesgo_count", 0)
        vuln_total_g = detalle.get("vuln_total", 0)
        vuln_count_g = detalle.get("vuln_count", 0)
        riesgo_count_g = detalle.get("riesgo_count", 0)

        bar_vuln = _dual_bar(
            "Vulnerables",
            v_col_pct,
            v_udla_pct,
            max_val=100,
            tooltip_col=f"{ref_label}: {v_col_pct:.1f}% ({vuln_count_ref}/{vuln_total_ref} personas)",
            tooltip_udla=f"{grupo_detalle}: {v_udla_pct:.1f}% ({vuln_count_g}/{vuln_total_g} personas)",
        )
        bar_riesgo = _dual_bar(
            "En riesgo",
            r_col_pct,
            r_udla_pct,
            max_val=100,
            tooltip_col=f"{ref_label}: {r_col_pct:.1f}% ({riesgo_count_ref}/{vuln_total_ref} personas)",
            tooltip_udla=f"{grupo_detalle}: {r_udla_pct:.1f}% ({riesgo_count_g}/{vuln_total_g} personas)",
        )

        sim_v_html = (
            (
                '<div style="margin-left:auto;">'
                + _metric_html("Similitud", sim_v, "de 100", "#dc2626")
                + "</div>"
            )
            if sim_v
            else ""
        )
        html_vuln = (
            '<div class="cat-card">'
            '<div class="cat-card-header">'
            '<div class="cat-icon" style="background:#fee2e2;">⚠️</div>'
            "<div>"
            '<div class="cat-title">Vulnerabilidad</div>'
            '<div class="cat-subtitle">Proporción de hogares vulnerables y en riesgo</div>'
            "</div>"
            + sim_v_html
            + "</div>"
            + _legend()
            + '<div class="metric-row">'
            + _metric_html(
                "% Vulnerables",
                f"{v_pct_ref*100:.1f}%",
                ref_label,
                COLOR_COL,
            )
            + _metric_html(
                "% Vulnerables",
                f"{v_pct_g*100:.1f}%",
                grupo_detalle,
                COLOR_UDLA,
            )
            + _metric_html(
                "% En riesgo",
                f"{r_pct_ref*100:.1f}%",
                ref_label,
                COLOR_COL,
            )
            + _metric_html(
                "% En riesgo",
                f"{r_pct_g*100:.1f}%",
                grupo_detalle,
                COLOR_UDLA,
            )
            + "</div>"
            + bar_vuln
            + bar_riesgo
            + "</div>"
        )
        st.markdown(html_vuln, unsafe_allow_html=True)

    # ── TARJETA: Ubicación ──

    if "Ubicacion (parroquia)" in categorias_sel and loc_categorias is not None:
        loc_ref = ref_profile.get("loc_dist")
        loc_g = detalle.get("loc_dist")
        sim_u = ""
        if not row_sel.empty and "sim_ubicacion" in row_sel.columns:
            val = row_sel["sim_ubicacion"].iloc[0]
            sim_u = f"{0.0:.1f}" if pd.isna(val) else f"{float(val):.1f}"

        if loc_ref is not None and loc_g is not None:
            bars_loc = ""
            loc_count_ref = int(ref_profile.get("loc_count", 0))
            loc_count_g = int(detalle.get("loc_count", 0))
            for i, parr in enumerate(loc_categorias):
                v_c = loc_ref[i] * 100 if i < len(loc_ref) else 0.0
                v_u = loc_g[i] * 100 if i < len(loc_g) else 0.0
                c_c = (
                    int(round(loc_count_ref * (v_c / 100))) if loc_count_ref > 0 else 0
                )
                c_u = int(round(loc_count_g * (v_u / 100))) if loc_count_g > 0 else 0
                t_c = f"{ref_label}: {v_c:.1f}% ({c_c}/{loc_count_ref})"
                t_u = f"{grupo_detalle}: {v_u:.1f}% ({c_u}/{loc_count_g})"
                bars_loc += _dual_bar(
                    parr, v_c, v_u, max_val=100, tooltip_col=t_c, tooltip_udla=t_u
                )

            sim_u_html = (
                (
                    '<div style="margin-left:auto;">'
                    + _metric_html("Similitud", sim_u, "de 100", "#047857")
                    + "</div>"
                )
                if sim_u
                else ""
            )
            html_ubic = (
                '<div class="cat-card">'
                '<div class="cat-card-header">'
                '<div class="cat-icon" style="background:#d1fae5;">📍</div>'
                "<div>"
                '<div class="cat-title">Ubicación por Parroquia</div>'
                '<div class="cat-subtitle">Distribución geográfica de los estudiantes (top 12 parroquias)</div>'
                "</div>" + sim_u_html + "</div>" + _legend() + bars_loc + "</div>"
            )
            st.markdown(html_ubic, unsafe_allow_html=True)
        else:
            st.info("No hay datos suficientes de ubicación para este quintil.")

    st.markdown("---")
    st.caption(
        "💡 Las barras comparan las proporciones entre el quintil del Colegio y el quintil UDLA. "
        "El puntaje de similitud indica cuánto se parecen en cada categoría."
    )
