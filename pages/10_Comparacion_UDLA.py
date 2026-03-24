"""
Pagina 10 – Comparacion UDLA
Una sola pagina con dos pestañas internas (st.tabs):
  Tab 1 → Filtros y Ranking: configuracion + Top 10 en tarjetas
  Tab 2 → Detalle por Categoria: comparacion visual con tarjetas
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import streamlit as st

from utils.excel_loader import load_excel_sheet
from utils.udla_sql import cargar_datos_udla
from utils.comparacion_helpers import (
    QUINTIL_LABELS,
    load_ubicacion_periodo,
    cargar_parroquias,
    norm_id,
    parse_valor_deuda,
    build_familias,
    salario_por_id,
    deuda_por_id,
    hogares_salario_deuda,
    quintil_dist,
    calcular_vulnerabilidad,
    asignar_parroquia,
    parroquia_dist,
    build_feature_vector,
    calcular_similitud,
)

# ─── Configuración de página ──────────────────────────────────────────────────

st.set_page_config(page_title="Comparacion UDLA", page_icon="🔍", layout="wide")

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
    .badge-quintil { background: #ede9fe; color: #6c63ff; }
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
        <h1>🔍 Comparacion de Perfiles vs UDLA</h1>
        <p>Configura filtros y pesos para encontrar las carreras o facultades UDLA
        con perfil socioeconómico más similar al del colegio.</p>
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
        key=f"anio_{label}",
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
        key=f"mes_{label}",
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

# ─── Filtros principales (sidebar-style, arriba de los tabs) ──────────────────

st.markdown("#### ⚙️ Filtros principales")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    comparar_por = st.selectbox(
        "Comparar por", options=["Carrera", "Facultad"], index=0
    )
with col_f2:
    grupo_udla = st.selectbox(
        "Grupo UDLA",
        options=["E", "A", "G"],
        format_func=lambda x: {"E": "Enrollment", "A": "Afluentes", "G": "Graduados"}[
            x
        ],
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
        periodo_sel = st.selectbox("Periodo UDLA", options=opts_periodo, index=0)
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
# Perfil colegio
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
total_hogares_col = (
    int(hogares_col["hogar_id"].nunique()) if not hogares_col.empty else 0
)

quintil_dist_col = quintil_dist(hogares_col)
deuda_avg_col = float(hogares_col["deuda"].mean()) if not hogares_col.empty else 0.0
deuda_pct_col = (
    float((hogares_col["deuda"] > 0).mean()) if not hogares_col.empty else 0.0
)

vuln_col = calcular_vulnerabilidad(
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
vulnerable_pct_col = float(vuln_col["vulnerable"].mean()) if not vuln_col.empty else 0.0
riesgo_pct_col = float(vuln_col["en_riesgo"].mean()) if not vuln_col.empty else 0.0
total_vuln_col = int(len(vuln_col)) if not vuln_col.empty else 0

info_personal["IDENTIFICACION"] = norm_id(info_personal["IDENTIFICACION"])
loc_df_col = info_personal[info_personal["IDENTIFICACION"].isin(ids_estudiantes)].copy()

gdf_parroquias = cargar_parroquias()
loc_series_col = None
if not gdf_parroquias.empty:
    df_parro_col = asignar_parroquia(loc_df_col, gdf_parroquias, "LATITUD", "LONGITUD")
    if "parroquia" in df_parro_col.columns:
        loc_series_col = df_parro_col["parroquia"].dropna().astype(str).str.strip()
        if loc_series_col.empty:
            loc_series_col = None
total_loc_col = int(loc_series_col.shape[0]) if loc_series_col is not None else 0

# ═══════════════════════════════════════════════════════════════════════════════
# Perfil UDLA
# ═══════════════════════════════════════════════════════════════════════════════

personas_udla = personas_udla.copy()
personas_udla = personas_udla[personas_udla["tipo"] == grupo_udla]
periodo_norm_sel = _norm_period(periodo_sel) if periodo_sel != "Todos" else ""
if periodo_sel != "Todos" and "periodo" in personas_udla.columns:
    personas_udla = personas_udla[
        personas_udla["periodo"].map(_norm_period) == periodo_norm_sel
    ]

group_col = "carrera" if comparar_por == "Carrera" else "facultad"
if group_col not in personas_udla.columns:
    st.info("No existe la columna de agrupacion en UDLA.")
    st.stop()

personas_udla[group_col] = personas_udla[group_col].astype(str).str.strip().str.upper()
personas_udla = personas_udla[personas_udla[group_col] != ""]

if personas_udla.empty:
    st.info(
        f"No hay registros UDLA para grupo `{grupo_udla}` en el periodo `{periodo_sel}`."
    )
    st.stop()

familias_udla, mapa_udla = build_familias(
    personas_udla[["identificacion", group_col]].copy(),
    familiares_udla,
    id_col="identificacion",
    padre_col="ced_padre",
    madre_col="ced_madre",
)

if familias_udla.empty:
    st.warning(
        "No hay datos familiares para este periodo/grupo UDLA. "
        "El ranking se calculara con categorias disponibles (ej. ubicacion)."
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

# ubicación UDLA
ub_parro = None
if (
    not ubicacion_udla.empty
    and not gdf_parroquias.empty
    and {"id", "semestre", "latitud", "longitud"}.issubset(set(ubicacion_udla.columns))
):
    ub = ubicacion_udla.copy()
    ub["id"] = ub["id"].astype(str).str.strip()
    ub["semestre_norm"] = ub["semestre"].map(_norm_period)
    if periodo_sel != "Todos":
        # Regla de negocio: para 202520 y 202610 usar ubicacion del semestre 202520.
        semestre_csv = (
            "202520" if periodo_norm_sel in {"202520", "202610"} else periodo_norm_sel
        )
        ub = ub[ub["semestre_norm"] == semestre_csv]
    # El CSV ya tiene la columna 'carrera': usarla directamente
    # (el CSV usa codigos de estudiante en 'id', no cedula, asi que no
    #  se puede cruzar con personas_udla.identificacion)
    if group_col in ub.columns:
        ub[group_col] = ub[group_col].astype(str).str.strip().str.upper()
        valid_groups = set(personas_udla[group_col].unique())
        ub = ub[ub[group_col].isin(valid_groups)]
    else:
        # Fallback: si el CSV no tiene la columna de agrupacion,
        # intentar cruzar por ID (puede no funcionar)
        ub = ub[ub["id"].isin(personas_udla["identificacion"].astype(str))]
        ub = ub.merge(
            personas_udla[["identificacion", group_col]].drop_duplicates(),
            left_on="id",
            right_on="identificacion",
            how="inner",
        )
    if not ub.empty:
        ub_parro = asignar_parroquia(ub, gdf_parroquias, "latitud", "longitud")

loc_categorias = None
if (
    loc_series_col is not None
    and ub_parro is not None
    and "parroquia" in ub_parro.columns
):
    top = (
        pd.concat([loc_series_col, ub_parro["parroquia"]])
        .value_counts()
        .head(12)
        .index.tolist()
    )
    loc_categorias = list(top) + ["Otros"]

# ═══════════════════════════════════════════════════════════════════════════════
# Construir perfiles por grupo
# ═══════════════════════════════════════════════════════════════════════════════

perfiles = []
detalles_grupo: dict[str, dict] = {}
for grupo, df_personas_g in personas_udla.groupby(group_col):
    if df_personas_g.empty:
        continue
    df_g = (
        familias_udla[familias_udla[group_col] == grupo].copy()
        if not familias_udla.empty
        else familias_udla.copy()
    )
    hogares_ids = df_g["hogar_id"].unique().tolist()
    df_mapa_g = mapa_udla[mapa_udla["hogar_id"].isin(hogares_ids)].copy()
    hogares_g = hogares_salario_deuda(df_mapa_g, salario_map_udla, deuda_map_udla)
    total_hogares_g = int(hogares_g["hogar_id"].nunique()) if not hogares_g.empty else 0

    quintil_dist_g = quintil_dist(hogares_g)
    deuda_avg_g = float(hogares_g["deuda"].mean()) if not hogares_g.empty else 0.0
    deuda_pct_g = float((hogares_g["deuda"] > 0).mean()) if not hogares_g.empty else 0.0

    vuln_g = (
        calcular_vulnerabilidad(
            df_g,
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
        if not df_g.empty
        else pd.DataFrame()
    )
    vulnerable_pct_g = float(vuln_g["vulnerable"].mean()) if not vuln_g.empty else 0.0
    riesgo_pct_g = float(vuln_g["en_riesgo"].mean()) if not vuln_g.empty else 0.0

    loc_dist_g = None
    loc_count_g = 0
    if (
        loc_categorias is not None
        and ub_parro is not None
        and "parroquia" in ub_parro.columns
    ):
        loc_series_g = ub_parro.loc[ub_parro[group_col] == grupo, "parroquia"]
        if loc_series_g.empty:
            loc_dist_g = None
        else:
            loc_dist_g = parroquia_dist(loc_series_g, loc_categorias)
            loc_count_g = int(loc_series_g.dropna().shape[0])

    features = build_feature_vector(
        quintil_dist_g,
        deuda_avg_g,
        deuda_pct_g,
        vulnerable_pct_g,
        riesgo_pct_g,
        loc_dist_g,
    )
    total_est = int(df_personas_g["identificacion"].nunique())
    perfiles.append({"grupo": grupo, "total_estudiantes": total_est, **features})
    detalles_grupo[grupo] = {
        "quintil_dist": quintil_dist_g,
        "total_hogares": total_hogares_g,
        "deuda_avg": deuda_avg_g,
        "deuda_pct": deuda_pct_g,
        "vulnerable_pct": vulnerable_pct_g,
        "riesgo_pct": riesgo_pct_g,
        "loc_dist": loc_dist_g,
        "loc_count": loc_count_g,
    }

if not perfiles:
    st.info("No hay grupos UDLA con datos suficientes.")
    st.stop()

df_perfiles = pd.DataFrame(perfiles)

loc_dist_col = None
if loc_categorias is not None and loc_series_col is not None:
    loc_dist_col = parroquia_dist(loc_series_col, loc_categorias)

features_col = build_feature_vector(
    quintil_dist_col,
    deuda_avg_col,
    deuda_pct_col,
    vulnerable_pct_col,
    riesgo_pct_col,
    loc_dist_col,
)
feature_cols = [
    c for c in df_perfiles.columns if c not in ["grupo", "total_estudiantes"]
]

# ─── Pesos por categoría ─────────────────────────────────────────────────────

with st.expander("⚖️ Categorias y pesos (total 100)", expanded=False):
    categorias_disp = ["Quintiles", "Deuda", "Vulnerabilidad"]
    if loc_categorias is not None:
        categorias_disp.append("Ubicacion (parroquia)")
    else:
        st.caption("ℹ️ Ubicacion (parroquia) no disponible por falta de datos.")

    categorias_sel = st.multiselect(
        "Categorias a usar", options=categorias_disp, default=categorias_disp
    )
    categorias_sel = [c for c in categorias_sel if c in categorias_disp]
    if not categorias_sel:
        categorias_sel = categorias_disp.copy()
        st.warning(
            "No habia categorias seleccionadas; se aplicaron automaticamente las disponibles."
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
                key=f"peso_{cat}",
            )

    total_pesos = sum(pesos.values())
    if total_pesos != 100:
        st.warning(f"Los pesos suman **{total_pesos}** — deben sumar **100**.")
        st.stop()

    w_quintil = (
        pesos.get("Quintiles", 0) / 100.0 if "Quintiles" in categorias_sel else 0.0
    )
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
    if col.startswith("q_"):
        weights[col] = w_quintil
    elif col.startswith("loc_"):
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
    features_col,
    feature_cols,
    weights,
    categorias_sel,
    {
        "Quintiles": w_quintil,
        "Deuda": w_deuda,
        "Vulnerabilidad": w_vuln,
        "Ubicacion (parroquia)": w_loc,
    },
)

if "sim_ubicacion" in df_result.columns:
    sin_ubicacion = df_result["grupo"].apply(
        lambda g: detalles_grupo.get(g, {}).get("loc_dist") is None
    )
    df_result.loc[sin_ubicacion, "sim_ubicacion"] = np.nan

# ═══════════════════════════════════════════════════════════════════════════════
# ██  DOS PESTAÑAS  ██
# ═══════════════════════════════════════════════════════════════════════════════

tab_ranking, tab_detalle = st.tabs(
    ["🏆 Top Perfiles Similares", "📊 Detalle por Categoria"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: RANKING
# ─────────────────────────────────────────────────────────────────────────────

with tab_ranking:
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; margin-bottom:8px;">'
        '<span style="font-size:22px; font-weight:800; color:#1a1a2e;">'
        "🏆 Top 10 Perfiles UDLA mas Similares</span></div>",
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
        badges = ""
        if "sim_quintiles" in row.index and pd.notna(row.get("sim_quintiles")):
            badges += f'<span class="badge badge-quintil">Quintiles: {row["sim_quintiles"]:.1f}</span>'
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
        return f"""
        <div class="card">
            <div class="card-rank">#{rank}</div>
            <div class="card-title">{row['grupo']}</div>
            <div class="card-students">👤 {int(row['total_estudiantes']):,} estudiantes</div>
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
        "💡 **¿Como se calcula?** El puntaje usa distancia estandarizada entre perfiles. "
        "Los pesos permiten dar mayor importancia a cada categoria."
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: DETALLE POR CATEGORÍA
# ─────────────────────────────────────────────────────────────────────────────

COLOR_COL = "#6c63ff"
COLOR_UDLA = "#48b1ff"

with tab_detalle:
    grupos_disponibles = df_result["grupo"].tolist()
    grupo_detalle = st.selectbox(
        f"Selecciona {comparar_por} para ver detalle",
        options=grupos_disponibles,
        index=0,
        key="sel_grupo_detalle",
    )
    detalle = detalles_grupo.get(grupo_detalle, {})

    row_sel = df_result[df_result["grupo"] == grupo_detalle]
    score_sel = (
        float(row_sel["puntaje_similitud"].iloc[0]) if not row_sel.empty else 0.0
    )
    total_est_sel = (
        int(row_sel["total_estudiantes"].iloc[0]) if not row_sel.empty else 0
    )
    rank_sel = (
        grupos_disponibles.index(grupo_detalle) + 1
        if grupo_detalle in grupos_disponibles
        else 0
    )

    html_header = (
        '<div class="grupo-header">'
        "<div>"
        f'<div class="grupo-name">{grupo_detalle}</div>'
        f'<span style="font-size:13px; color:#888;">'
        f"👤 {total_est_sel:,} estudiantes &nbsp;·&nbsp; Posicion #{rank_sel} del ranking"
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
            f'<div class="legend-item"><div class="legend-dot" style="background:{COLOR_COL};"></div>Colegio</div>'
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

    # ── TARJETA: Quintiles ──

    if "Quintiles" in categorias_sel:
        bars_html = ""
        q_dist_g = detalle.get("quintil_dist", {})
        total_hog_g = int(detalle.get("total_hogares", 0))
        for q_label in QUINTIL_LABELS:
            v_col = quintil_dist_col.get(q_label, 0.0) * 100
            v_udla = q_dist_g.get(q_label, 0.0) * 100
            c_col = (
                int(round(total_hogares_col * (v_col / 100)))
                if total_hogares_col > 0
                else 0
            )
            c_udla = int(round(total_hog_g * (v_udla / 100))) if total_hog_g > 0 else 0
            t_col = f"Colegio: {v_col:.1f}% ({c_col}/{total_hogares_col})"
            t_udla = f"{grupo_detalle}: {v_udla:.1f}% ({c_udla}/{total_hog_g})"
            bars_html += _dual_bar(
                q_label,
                v_col,
                v_udla,
                max_val=100,
                tooltip_col=t_col,
                tooltip_udla=t_udla,
            )
        sim_q = ""
        if not row_sel.empty and "sim_quintiles" in row_sel.columns:
            sim_q = f"{float(row_sel['sim_quintiles'].iloc[0]):.1f}"

        sim_q_html = (
            (
                '<div style="margin-left:auto;">'
                + _metric_html("Similitud", sim_q, "de 100", "#6c63ff")
                + "</div>"
            )
            if sim_q
            else ""
        )
        html_quintiles = (
            '<div class="cat-card">'
            '<div class="cat-card-header">'
            '<div class="cat-icon" style="background:#ede9fe;">📊</div>'
            "<div>"
            '<div class="cat-title">Distribucion por Quintiles</div>'
            '<div class="cat-subtitle">Comparacion de la distribucion de ingresos de los hogares</div>'
            "</div>" + sim_q_html + "</div>" + _legend() + bars_html + "</div>"
        )
        st.markdown(html_quintiles, unsafe_allow_html=True)

    # ── TARJETA: Deuda ──

    if "Deuda" in categorias_sel:
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
            + _metric_html(
                "Deuda promedio", f"${deuda_avg_col:,.0f}", "Colegio", COLOR_COL
            )
            + _metric_html(
                "Deuda promedio", f"${d_avg_g:,.0f}", grupo_detalle, COLOR_UDLA
            )
            + "</div>"
            '<div class="metric-row">'
            + _metric_html(
                "Hogares con deuda", f"{deuda_pct_col*100:.1f}%", "Colegio", COLOR_COL
            )
            + _metric_html(
                "Hogares con deuda", f"{d_pct_g*100:.1f}%", grupo_detalle, COLOR_UDLA
            )
            + "</div>"
            "</div>"
        )
        st.markdown(html_deuda, unsafe_allow_html=True)

    # ── TARJETA: Vulnerabilidad ──

    if "Vulnerabilidad" in categorias_sel:
        v_pct_g = detalle.get("vulnerable_pct", 0.0)
        r_pct_g = detalle.get("riesgo_pct", 0.0)
        sim_v = ""
        if not row_sel.empty and "sim_vulnerabilidad" in row_sel.columns:
            sim_v = f"{float(row_sel['sim_vulnerabilidad'].iloc[0]):.1f}"

        v_col = vulnerable_pct_col * 100
        v_udla = v_pct_g * 100
        c_col = (
            int(round(total_vuln_col * vulnerable_pct_col)) if total_vuln_col > 0 else 0
        )
        c_udla = int(round(total_est_sel * v_pct_g)) if total_est_sel > 0 else 0
        bar_vuln = _dual_bar(
            "Vulnerables",
            v_col,
            v_udla,
            max_val=100,
            tooltip_col=f"Colegio: {v_col:.1f}% ({c_col}/{total_vuln_col})",
            tooltip_udla=f"{grupo_detalle}: {v_udla:.1f}% ({c_udla}/{total_est_sel})",
        )

        r_col = riesgo_pct_col * 100
        r_udla = r_pct_g * 100
        c_col_r = (
            int(round(total_vuln_col * riesgo_pct_col)) if total_vuln_col > 0 else 0
        )
        c_udla_r = int(round(total_est_sel * r_pct_g)) if total_est_sel > 0 else 0
        bar_riesgo = _dual_bar(
            "En riesgo",
            r_col,
            r_udla,
            max_val=100,
            tooltip_col=f"Colegio: {r_col:.1f}% ({c_col_r}/{total_vuln_col})",
            tooltip_udla=f"{grupo_detalle}: {r_udla:.1f}% ({c_udla_r}/{total_est_sel})",
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
            '<div class="cat-subtitle">Proporcion de hogares vulnerables y en riesgo</div>'
            "</div>"
            + sim_v_html
            + "</div>"
            + _legend()
            + '<div class="metric-row">'
            + _metric_html(
                "% Vulnerables", f"{vulnerable_pct_col*100:.1f}%", "Colegio", COLOR_COL
            )
            + _metric_html(
                "% Vulnerables", f"{v_pct_g*100:.1f}%", grupo_detalle, COLOR_UDLA
            )
            + _metric_html(
                "% En riesgo", f"{riesgo_pct_col*100:.1f}%", "Colegio", COLOR_COL
            )
            + _metric_html(
                "% En riesgo", f"{r_pct_g*100:.1f}%", grupo_detalle, COLOR_UDLA
            )
            + "</div>"
            + bar_vuln
            + bar_riesgo
            + "</div>"
        )
        st.markdown(html_vuln, unsafe_allow_html=True)

    # ── TARJETA: Ubicación ──

    if "Ubicacion (parroquia)" in categorias_sel and loc_categorias is not None:
        loc_dist_g = detalle.get("loc_dist", None)
        sim_u = ""
        if not row_sel.empty and "sim_ubicacion" in row_sel.columns:
            val = row_sel["sim_ubicacion"].iloc[0]
            sim_u = f"{0.0:.1f}" if pd.isna(val) else f"{float(val):.1f}"

        if loc_dist_g is not None and loc_dist_col is not None:
            bars_loc = ""
            loc_total_g = int(detalle.get("loc_count", 0))
            for i, parr in enumerate(loc_categorias):
                v_c = loc_dist_col[i] * 100 if i < len(loc_dist_col) else 0.0
                v_u = loc_dist_g[i] * 100 if i < len(loc_dist_g) else 0.0
                c_c = (
                    int(round(total_loc_col * (v_c / 100))) if total_loc_col > 0 else 0
                )
                c_u = int(round(loc_total_g * (v_u / 100))) if loc_total_g > 0 else 0
                t_c = f"Colegio: {v_c:.1f}% ({c_c}/{total_loc_col})"
                t_u = f"{grupo_detalle}: {v_u:.1f}% ({c_u}/{loc_total_g})"
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
                '<div class="cat-title">Ubicacion por Parroquia</div>'
                '<div class="cat-subtitle">Distribucion geografica de los estudiantes (top 12 parroquias)</div>'
                "</div>" + sim_u_html + "</div>" + _legend() + bars_loc + "</div>"
            )
            st.markdown(html_ubic, unsafe_allow_html=True)
        else:
            st.info("No hay datos suficientes de ubicacion para este grupo.")

    st.markdown("---")
    st.caption(
        "💡 Las barras muestran la proporcion en cada dimension. "
        "El puntaje de similitud indica cuanto se parece cada categoria entre el colegio y el grupo UDLA."
    )
