from __future__ import annotations

import re

import altair as alt
import pandas as pd
import streamlit as st

from utils.comparacion_helpers import build_familias, norm_id, salario_por_id
from utils.excel_loader import load_excel_sheet
from utils.quintile_ranges import asignar_quintil_por_rangos, calcular_rangos_quintiles
from utils.student_columns import normalize_university_column
from utils.student_filters import render_student_academic_filters
from utils.udla_sql import cargar_datos_udla

QUINTIL_ORDER = ["Sin empleo", "1", "2", "3", "4", "5"]

QUINTILES_UDLA = {
    1: {"min": 105.75, "max": 482.00},
    2: {"min": 482.01, "max": 850.92},
    3: {"min": 850.93, "max": 1542.92},
    4: {"min": 1542.93, "max": 2525.00},
    5: {"min": 2525.01, "max": 15392.53},
}


def _normalizar_universidad(df: pd.DataFrame) -> pd.DataFrame:
    return normalize_university_column(df)


def _norm_period(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if s == "":
        return ""
    try:
        return str(int(float(s)))
    except Exception:
        return re.sub(r"\D", "", s)


def _select_anio_mes(
    df: pd.DataFrame, anio_col: str, mes_col: str, label: str, key_prefix: str
) -> tuple[int | None, int | None]:
    if df.empty or anio_col not in df.columns or mes_col not in df.columns:
        return None, None
    anios = sorted(
        pd.to_numeric(df[anio_col], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if not anios:
        return None, None
    anio = st.selectbox(
        f"Anio - {label}", options=anios, index=len(anios) - 1, key=f"{key_prefix}_anio"
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
        return anio, None
    mes = st.selectbox(
        f"Mes - {label}", options=meses, index=len(meses) - 1, key=f"{key_prefix}_mes"
    )
    return anio, mes


def _asignar_quintil_custom(salario: float, rangos: dict[int, dict[str, float]]) -> str:
    return asignar_quintil_por_rangos(salario, rangos, vacio="Sin empleo")


def _hogares_salario(df_mapa: pd.DataFrame, salario_map: dict) -> pd.DataFrame:
    if df_mapa.empty:
        return pd.DataFrame(columns=["hogar_id", "salario"])
    tmp = df_mapa.copy()
    tmp["salario"] = tmp["fam_id"].map(salario_map).fillna(0.0)
    return tmp.groupby("hogar_id", as_index=False)["salario"].sum()


def _resumen_quintiles(df_hogares: pd.DataFrame, label_col: str) -> pd.DataFrame:
    total = int(df_hogares["hogar_id"].nunique()) if not df_hogares.empty else 0
    if total == 0:
        return pd.DataFrame(
            {
                "quintil": QUINTIL_ORDER,
                f"hogares_{label_col}": [0] * len(QUINTIL_ORDER),
                f"pct_{label_col}": [0.0] * len(QUINTIL_ORDER),
            }
        )
    counts = (
        df_hogares["quintil"]
        .value_counts()
        .reindex(QUINTIL_ORDER, fill_value=0)
        .astype(int)
    )
    pct = (counts / total * 100.0).round(2)
    return pd.DataFrame(
        {
            "quintil": QUINTIL_ORDER,
            f"hogares_{label_col}": counts.values.tolist(),
            f"pct_{label_col}": pct.values.tolist(),
        }
    )


def _fmt_money(v: float) -> str:
    return f"${v:,.2f}"


def _rango_text(quintil: str, rangos: dict[int, dict[str, float]]) -> str:
    if quintil == "Sin empleo":
        return "Sin ingreso laboral"
    q = int(quintil)
    r = rangos[q]
    return f"{_fmt_money(r['min'])} - {_fmt_money(r['max'])}"


def _card_html(
    quintil: str,
    hogares_i: int,
    pct_i: float,
    hogares_u: int,
    pct_u: float,
    rango_i: str,
    rango_u: str,
    label_local: str,
) -> str:
    return f"""
    <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;padding:14px 16px;margin-bottom:12px;">
      <div style="font-size:18px;font-weight:700;color:#111827;margin-bottom:8px;">Quintil {quintil if quintil != 'Sin empleo' else 'Sin empleo'}</div>
      <div style="font-size:12px;color:#4b5563;margin-bottom:10px;">Rango referencia: {rango_i}</div>
      <div style="font-size:12px;color:#4b5563;margin-bottom:10px;">Rango UDLA: {rango_u}</div>
      <div style="display:flex;gap:10px;">
        <div style="flex:1;background:#eff6ff;border-radius:8px;padding:10px;">
          <div style="font-size:11px;color:#1d4ed8;font-weight:600;">{label_local}</div>
          <div style="font-size:22px;font-weight:700;color:#1e3a8a;line-height:1.1;">{hogares_i:,}</div>
          <div style="font-size:12px;color:#1e40af;">{pct_i:.2f}% del total</div>
        </div>
        <div style="flex:1;background:#f0fdf4;border-radius:8px;padding:10px;">
          <div style="font-size:11px;color:#15803d;font-weight:600;">UDLA</div>
          <div style="font-size:22px;font-weight:700;color:#14532d;line-height:1.1;">{hogares_u:,}</div>
          <div style="font-size:12px;color:#166534;">{pct_u:.2f}% del total</div>
        </div>
      </div>
    </div>
    """


st.set_page_config(page_title="Quintiles Similares vs UDLA", page_icon="Q", layout="wide")
title_placeholder = st.empty()
caption_placeholder = st.empty()

with st.spinner("Cargando datos..."):
    estudiantes = load_excel_sheet("Estudiantes")
    universo_familiares = load_excel_sheet("Universo Familiares")
    empleo = load_excel_sheet("Empleos")
    udla = cargar_datos_udla()

personas_udla = udla.get("Personas", pd.DataFrame())
familiares_udla = udla.get("Familiares", pd.DataFrame())
ingresos_udla = udla.get("Ingresos", pd.DataFrame())

if estudiantes.empty or universo_familiares.empty or empleo.empty:
    st.info("No hay datos suficientes de la universidad para calcular quintiles.")
    st.stop()
if personas_udla.empty or familiares_udla.empty or ingresos_udla.empty:
    st.info("No hay datos suficientes de UDLA para calcular quintiles.")
    st.stop()

if "Cedula" in estudiantes.columns:
    estudiantes = estudiantes.rename(columns={"Cedula": "IDENTIFICACION"})
elif "CEDULA" in estudiantes.columns:
    estudiantes = estudiantes.rename(columns={"CEDULA": "IDENTIFICACION"})

estudiantes = _normalizar_universidad(estudiantes)
estudiantes["IDENTIFICACION"] = norm_id(estudiantes["IDENTIFICACION"])
universo_familiares["IDENTIFICACION"] = norm_id(universo_familiares["IDENTIFICACION"])

personas_udla = personas_udla.copy()
personas_udla["identificacion"] = norm_id(personas_udla["identificacion"])
familiares_udla = familiares_udla.copy()
familiares_udla["identificacion"] = norm_id(familiares_udla["identificacion"])
ingresos_udla = ingresos_udla.copy()
ingresos_udla["identificacion"] = norm_id(ingresos_udla["identificacion"])
ingresos_udla["salario"] = pd.to_numeric(ingresos_udla["salario"], errors="coerce").fillna(
    0.0
)

st.markdown("### Filtros")
estudiantes_filtrados, filtros_estudiantes = render_student_academic_filters(
    estudiantes, key_prefix="quintiles_similares"
)
universidad_sel = filtros_estudiantes["universidad"] or "Todas las universidades"
f1, f2 = st.columns(2)
with f1:
    tipo_udla = st.selectbox(
        "Grupo UDLA",
        options=["E", "A", "G"],
        format_func=lambda x: {"E": "Enrollment", "A": "Afluentes", "G": "Graduados"}[
            x
        ],
        index=0,
    )
with f2:
    periodos_udla = []
    if "periodo" in personas_udla.columns:
        periodos_udla = (
            personas_udla.loc[personas_udla["tipo"] == tipo_udla, "periodo"]
            .map(_norm_period)
            .replace("", pd.NA)
            .dropna()
            .unique()
            .tolist()
        )
        periodos_udla = sorted(periodos_udla)
    periodo_udla_sel = st.selectbox(
        "Periodo UDLA",
        options=["Todos"] + periodos_udla if periodos_udla else ["Todos"],
        index=0,
    )

with st.expander("Ajustes de periodo de ingresos", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        anio_emp, mes_emp = _select_anio_mes(
            empleo, "ANIO", "MES", "Empleos Universidad", "universidad"
        )
    with c2:
        anio_ing_u, mes_ing_u = _select_anio_mes(
            ingresos_udla, "anio", "mes", "Ingresos UDLA", "udla"
        )

if anio_emp is None or mes_emp is None:
    st.info("No hay periodo de empleos de la universidad disponible.")
    st.stop()
if anio_ing_u is None or mes_ing_u is None:
    st.info("No hay periodo de ingresos UDLA disponible.")
    st.stop()

if estudiantes_filtrados.empty:
    st.info("No hay estudiantes para la universidad seleccionada.")
    st.stop()

label_local = (
    universidad_sel
    if universidad_sel != "Todas las universidades"
    else "Todas las universidades"
)

title_placeholder.title(f"Quintiles Similares: {label_local} vs UDLA")
caption_placeholder.caption(
    f"Comparacion de distribucion de hogares por quintil entre {label_local} y UDLA."
)

# Universidad: hogares por familiares de estudiantes
familias_i, mapa_i = build_familias(
    estudiantes_filtrados[["IDENTIFICACION"]].copy(),
    universo_familiares,
    id_col="IDENTIFICACION",
    padre_col="CED_PADRE",
    madre_col="CED_MADRE",
)
empleo_i = empleo[(empleo["ANIO"] == anio_emp) & (empleo["MES"] == mes_emp)].copy()
empleo_i["IDENTIFICACION"] = norm_id(empleo_i["IDENTIFICACION"])
empleo_i["SALARIO"] = pd.to_numeric(empleo_i["SALARIO"], errors="coerce").fillna(0.0)
salario_map_i = salario_por_id(empleo_i, "IDENTIFICACION", "SALARIO")
hogares_i = _hogares_salario(mapa_i, salario_map_i)
rangos_universidad = calcular_rangos_quintiles(hogares_i["salario"])
hogares_i["quintil"] = hogares_i["salario"].apply(
    lambda x: _asignar_quintil_custom(x, rangos_universidad)
)

# UDLA: hogares por familiares de estudiantes UDLA
personas_u = personas_udla[personas_udla["tipo"] == tipo_udla].copy()
if periodo_udla_sel != "Todos" and "periodo" in personas_u.columns:
    periodo_norm = _norm_period(periodo_udla_sel)
    personas_u = personas_u[personas_u["periodo"].map(_norm_period) == periodo_norm]

if personas_u.empty:
    st.info("No hay estudiantes UDLA para el filtro seleccionado.")
    st.stop()

familias_u, mapa_u = build_familias(
    personas_u[["identificacion"]].copy(),
    familiares_udla,
    id_col="identificacion",
    padre_col="ced_padre",
    madre_col="ced_madre",
)

ing_u = ingresos_udla[
    (ingresos_udla["anio"] == anio_ing_u) & (ingresos_udla["mes"] == mes_ing_u)
].copy()
salario_map_u = salario_por_id(ing_u, "identificacion", "salario")
hogares_u = _hogares_salario(mapa_u, salario_map_u)
hogares_u["quintil"] = hogares_u["salario"].apply(
    lambda x: _asignar_quintil_custom(x, QUINTILES_UDLA)
)

res_i = _resumen_quintiles(hogares_i, "universidad")
res_u = _resumen_quintiles(hogares_u, "udla")

res = res_i.merge(res_u, on="quintil", how="outer").fillna(0)
res["hogares_universidad"] = res["hogares_universidad"].astype(int)
res["hogares_udla"] = res["hogares_udla"].astype(int)

chart_df = pd.DataFrame(
    {
        "quintil": QUINTIL_ORDER + QUINTIL_ORDER,
        "grupo": [label_local] * len(QUINTIL_ORDER) + ["UDLA"] * len(QUINTIL_ORDER),
        "porcentaje": res["pct_universidad"].tolist() + res["pct_udla"].tolist(),
        "hogares": res["hogares_universidad"].tolist() + res["hogares_udla"].tolist(),
    }
)

st.markdown("### Grafica por quintiles")
chart = (
    alt.Chart(chart_df)
    .mark_bar()
    .encode(
        x=alt.X("quintil:N", title="Quintil", sort=QUINTIL_ORDER),
        xOffset="grupo:N",
        y=alt.Y("porcentaje:Q", title="% de hogares"),
        color=alt.Color("grupo:N", scale=alt.Scale(range=["#2563eb", "#16a34a"])),
        tooltip=[
            alt.Tooltip("grupo:N", title="Grupo"),
            alt.Tooltip("quintil:N", title="Quintil"),
            alt.Tooltip("hogares:Q", title="Hogares", format=","),
            alt.Tooltip("porcentaje:Q", title="% del total", format=".2f"),
        ],
    )
    .properties(height=360)
)
st.altair_chart(chart, use_container_width=True)

st.markdown("### Tarjetas por quintil")
cards = []
for q in QUINTIL_ORDER:
    row = res[res["quintil"] == q].iloc[0]
    cards.append(
        {
            "quintil": q,
            "hog_i": int(row["hogares_universidad"]),
            "pct_i": float(row["pct_universidad"]),
            "hog_u": int(row["hogares_udla"]),
            "pct_u": float(row["pct_udla"]),
            "rango_i": _rango_text(q, rangos_universidad),
            "rango_u": _rango_text(q, QUINTILES_UDLA),
        }
    )

for i in range(0, len(cards), 2):
    col1, col2 = st.columns(2)
    with col1:
        c = cards[i]
        st.markdown(
            _card_html(
                c["quintil"],
                c["hog_i"],
                c["pct_i"],
                c["hog_u"],
                c["pct_u"],
                c["rango_i"],
                c["rango_u"],
                label_local,
            ),
            unsafe_allow_html=True,
        )
    if i + 1 < len(cards):
        with col2:
            c = cards[i + 1]
            st.markdown(
                _card_html(
                    c["quintil"],
                    c["hog_i"],
                    c["pct_i"],
                    c["hog_u"],
                    c["pct_u"],
                    c["rango_i"],
                    c["rango_u"],
                    label_local,
                ),
                unsafe_allow_html=True,
            )

comp = res[res["quintil"].isin(["1", "2", "3", "4", "5"])].copy()
comp["dif_pp"] = (comp["pct_universidad"] - comp["pct_udla"]).abs()
comp["similitud"] = 100.0 - comp["dif_pp"]
best = comp.sort_values(["dif_pp", "quintil"], ascending=[True, True]).iloc[0]

st.markdown("### Quintil mas relacionado")
st.success(
    f"Quintil {best['quintil']} - Similitud {best['similitud']:.2f}/100 "
    f"(diferencia {best['dif_pp']:.2f} puntos porcentuales)."
)
