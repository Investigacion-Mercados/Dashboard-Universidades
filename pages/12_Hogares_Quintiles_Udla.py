from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.comparacion_helpers import build_familias, norm_id, salario_por_id
from utils.excel_loader import load_excel_sheet
from utils.student_columns import normalize_university_column


QUINTIL_ORDER = ["Sin empleo", "1", "2", "3", "4", "5"]

# Rangos UDLA alineados a pages/11_Quintiles_Similares.py
QUINTILES_UDLA = {
    1: {"min": 105.75, "max": 482.00},
    2: {"min": 482.01, "max": 850.92},
    3: {"min": 850.93, "max": 1542.92},
    4: {"min": 1542.93, "max": 2525.00},
    5: {"min": 2525.01, "max": 15392.53},
}


def _normalizar_universidad(df: pd.DataFrame) -> pd.DataFrame:
    return normalize_university_column(df)


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    estudiantes = load_excel_sheet("Estudiantes")
    universo_familiares = load_excel_sheet("Universo Familiares")
    empleo = load_excel_sheet("Empleos")

    if "Cedula" in estudiantes.columns:
        estudiantes = estudiantes.rename(columns={"Cedula": "IDENTIFICACION"})
    elif "CEDULA" in estudiantes.columns:
        estudiantes = estudiantes.rename(columns={"CEDULA": "IDENTIFICACION"})

    estudiantes = _normalizar_universidad(estudiantes)

    return estudiantes, universo_familiares, empleo


def _select_anio_mes(
    df: pd.DataFrame, anio_col: str, mes_col: str, label: str
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
        f"Anio de empleos ({label})", options=anios, index=len(anios) - 1
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
        f"Mes de empleos ({label})", options=meses, index=len(meses) - 1
    )
    return anio, mes


def _asignar_quintil_udla(salario: float) -> str:
    if pd.isna(salario) or float(salario) <= 0:
        return "Sin empleo"

    val = float(salario)
    q_min = QUINTILES_UDLA[1]["min"]
    q_max = QUINTILES_UDLA[5]["max"]
    if val < q_min:
        return "1"
    if val > q_max:
        return "5"

    for q in [1, 2, 3, 4, 5]:
        r = QUINTILES_UDLA[q]
        if r["min"] <= val <= r["max"]:
            return str(q)
    return "Sin empleo"


def _hogares_salario(df_mapa: pd.DataFrame, salario_map: dict) -> pd.DataFrame:
    if df_mapa.empty:
        return pd.DataFrame(columns=["hogar_id", "salario"])
    tmp = df_mapa.copy()
    tmp["salario"] = tmp["fam_id"].map(salario_map).fillna(0.0)
    return tmp.groupby("hogar_id", as_index=False)["salario"].sum()


def _resumen_quintiles(df_hogares: pd.DataFrame) -> pd.DataFrame:
    total = int(df_hogares["hogar_id"].nunique()) if not df_hogares.empty else 0
    counts = (
        df_hogares["quintil"].value_counts().reindex(QUINTIL_ORDER, fill_value=0).astype(int)
        if total > 0
        else pd.Series([0] * len(QUINTIL_ORDER), index=QUINTIL_ORDER, dtype=int)
    )
    pct = (counts / total * 100.0).round(2) if total > 0 else counts.astype(float)
    return pd.DataFrame(
        {"quintil": QUINTIL_ORDER, "hogares": counts.values.tolist(), "pct": pct.values.tolist()}
    )


def _fmt_money(v: float) -> str:
    return f"${v:,.2f}"


def _rango_text(quintil: str) -> str:
    if quintil == "Sin empleo":
        return "Sin ingreso laboral"
    q = int(quintil)
    r = QUINTILES_UDLA[q]
    return f"{_fmt_money(r['min'])} - {_fmt_money(r['max'])}"


def _card_html(
    quintil: str, hogares: int, pct: float, rango_udla: str, label_grupo: str
) -> str:
    titulo = f"Quintil {quintil}" if quintil != "Sin empleo" else "Sin empleo"
    return f"""
    <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;padding:14px 16px;margin-bottom:12px;">
      <div style="font-size:18px;font-weight:700;color:#111827;margin-bottom:8px;">{titulo}</div>
      <div style="font-size:12px;color:#4b5563;margin-bottom:10px;">Rango UDLA: {rango_udla}</div>
      <div style="display:flex;gap:10px;">
        <div style="flex:1;background:#eff6ff;border-radius:8px;padding:10px;">
          <div style="font-size:11px;color:#1d4ed8;font-weight:600;">{label_grupo}</div>
          <div style="font-size:22px;font-weight:700;color:#1e3a8a;line-height:1.1;">{hogares:,}</div>
          <div style="font-size:12px;color:#1e40af;">{pct:.2f}% del total</div>
        </div>
      </div>
    </div>
    """


st.set_page_config(page_title="Quintiles vs UDLA", page_icon="Q", layout="wide")
title_placeholder = st.empty()
caption_placeholder = st.empty()

with st.spinner("Cargando datos..."):
    estudiantes, universo_familiares, empleo = load_data()

if estudiantes.empty or universo_familiares.empty or empleo.empty:
    st.info("No hay datos suficientes para calcular quintiles de hogares.")
    st.stop()

estudiantes["IDENTIFICACION"] = norm_id(estudiantes["IDENTIFICACION"])
universo_familiares["IDENTIFICACION"] = norm_id(universo_familiares["IDENTIFICACION"])

st.markdown("### Filtros")
col_filtro_1, col_filtro_2 = st.columns(2)
estudiantes_filtrados = estudiantes
universidad_sel = "Todas las universidades"

with col_filtro_1:
    if "Universidad" in estudiantes.columns:
        universidades_disponibles = sorted(
            estudiantes["Universidad"].dropna().astype(str).str.strip().unique().tolist()
        )
        universidad_sel = st.selectbox(
            "Universidad",
            options=["Todas las universidades"] + universidades_disponibles,
            index=0,
        )

        if universidad_sel != "Todas las universidades":
            estudiantes_filtrados = estudiantes[estudiantes["Universidad"] == universidad_sel]
    else:
        st.warning("La hoja Estudiantes no contiene la columna 'Universidad'.")

titulo_universidad = (
    universidad_sel
    if universidad_sel != "Todas las universidades"
    else "Todas las universidades"
)

title_placeholder.title(f"Quintiles: {titulo_universidad} vs UDLA")
caption_placeholder.caption(
    f"Distribucion de hogares de {titulo_universidad} usando los rangos de quintiles UDLA."
)

with col_filtro_2:
    anio_emp, mes_emp = _select_anio_mes(empleo, "ANIO", "MES", titulo_universidad)

if anio_emp is None or mes_emp is None:
    st.info("No hay periodo de empleos disponible.")
    st.stop()

if estudiantes_filtrados.empty:
    st.info("No hay estudiantes para la universidad seleccionada.")
    st.stop()

label_grupo = (
    f"Hogares {titulo_universidad}"
    if universidad_sel != "Todas las universidades"
    else "Hogares"
)

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
if hogares_i.empty:
    st.info("No hay hogares para los filtros seleccionados.")
    st.stop()

hogares_i["quintil"] = hogares_i["salario"].apply(_asignar_quintil_udla)
res = _resumen_quintiles(hogares_i)
total_hogares = int(hogares_i["hogar_id"].nunique())

st.markdown(
    f"""
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:14px 16px;margin-bottom:14px;">
      <div style="font-size:12px;color:#334155;">Total {label_grupo}</div>
      <div style="font-size:28px;font-weight:700;color:#0f172a;line-height:1.1;">{total_hogares:,}</div>
      <div style="font-size:12px;color:#475569;">Periodo: {int(anio_emp)}-{int(mes_emp):02d}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Tarjetas por quintil (Rangos UDLA)")
cards = []
for q in QUINTIL_ORDER:
    row = res[res["quintil"] == q].iloc[0]
    cards.append(
        {
            "quintil": q,
            "hogares": int(row["hogares"]),
            "pct": float(row["pct"]),
            "rango": _rango_text(q),
        }
    )

for i in range(0, len(cards), 2):
    col1, col2 = st.columns(2)
    with col1:
        c = cards[i]
        st.markdown(
            _card_html(c["quintil"], c["hogares"], c["pct"], c["rango"], label_grupo),
            unsafe_allow_html=True,
        )
    if i + 1 < len(cards):
        with col2:
            c = cards[i + 1]
            st.markdown(
                _card_html(c["quintil"], c["hogares"], c["pct"], c["rango"], label_grupo),
                unsafe_allow_html=True,
            )
