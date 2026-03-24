import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.excel_loader import get_active_excel_filename, load_excel_sheet


def _parse_rgba_str(rgba_str: str) -> Tuple[int, int, int, float]:
    m = re.match(r"rgba\((\d+),\s*(\d+),\s*(\d+),\s*([0-9.]+)\)", rgba_str.strip())
    if not m:
        raise ValueError(f"RGBA invalido: {rgba_str}")
    r, g, b, a = m.groups()
    return int(r), int(g), int(b), float(a)


def _mix_with_white(rgb: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    # t=0 => color base, t=1 => blanco
    r0, g0, b0 = rgb
    r = int(round(r0 * (1.0 - t) + 255 * t))
    g = int(round(g0 * (1.0 - t) + 255 * t))
    b = int(round(b0 * (1.0 - t) + 255 * t))
    return (r, g, b)


def generar_paleta_pastel(desde_rgba: str, n: int = 10) -> List[str]:
    base_r, base_g, base_b, _ = _parse_rgba_str(desde_rgba)
    base_rgb: Tuple[int, int, int] = (base_r, base_g, base_b)
    ts = np.linspace(0.0, 0.82, n)  # 0 = mas fuerte, 0.82 = muy pastel
    colores: List[str] = []
    for t in ts:
        r, g, b = _mix_with_white(base_rgb, float(t))
        colores.append(f"rgba({r},{g},{b},1.0)")
    return colores


PALETA_AZUL_PASTEL_10: List[str] = generar_paleta_pastel("rgba(0,112,192,1.0)", n=10)
ANIO_CORTE = 2025
MES_CORTE = 11
AFILIACION_VOLUNTARIA_CODE = "32-AFILIACION VOLUNTARIA(TIPEM-32)"
TIPOS_EMPLEO = [
    "Todos",
    "Relacion de Dependencia",
    "Afiliacion Voluntaria",
    "Desconocido",
]


@st.cache_data(show_spinner=False)
def load_data(excel_filename: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    estudiantes = load_excel_sheet("Estudiantes", excel_filename)
    universo_familiares = load_excel_sheet("Universo Familiares", excel_filename)
    empleo = load_excel_sheet("Empleos", excel_filename)
    deudas = load_excel_sheet("Deudas", excel_filename)

    if "Cedula" in estudiantes.columns:
        estudiantes = estudiantes.rename(columns={"Cedula": "IDENTIFICACION"})
    elif "CEDULA" in estudiantes.columns:
        estudiantes = estudiantes.rename(columns={"CEDULA": "IDENTIFICACION"})

    return estudiantes, universo_familiares, empleo, deudas


def _normalizar_ids_familia(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["CED_PADRE", "CED_MADRE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def _empleo_reciente(empleo: pd.DataFrame) -> pd.DataFrame:
    if empleo.empty:
        return empleo.copy()
    emp = empleo.sort_values(
        ["IDENTIFICACION", "ANIO", "MES"], ascending=[True, False, False]
    ).drop_duplicates("IDENTIFICACION", keep="first")
    emp["TIPO_EMPRESA"] = emp["TIPO_EMPRESA"].astype(str).str.strip()
    return emp


def _filtrar_mes(df: pd.DataFrame, anio: int, mes: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df[(df["ANIO"] == anio) & (df["MES"] == mes)].copy()


def construir_familiares_filtrado(
    universo_familiares: pd.DataFrame,
    empleo: pd.DataFrame,
    ids_base: set[int],
    cant_papas: int | None,
    cant_papas_trab: int | None,
    tipo_empleo_sel: str,
) -> set[int]:
    u = universo_familiares[universo_familiares["IDENTIFICACION"].isin(ids_base)].copy()
    if u.empty:
        return set()

    u = _normalizar_ids_familia(u)
    u["n_papas"] = (u["CED_PADRE"] != 0).astype(int) + (u["CED_MADRE"] != 0).astype(int)
    u = u[u["n_papas"] > 0]
    if cant_papas in (1, 2):
        u = u[u["n_papas"] == cant_papas]
    if u.empty:
        return set()

    u["hogar_id"] = u.apply(
        lambda r: "|".join(sorted([str(r["CED_PADRE"]), str(r["CED_MADRE"])])), axis=1
    )

    pares = []
    for _, r in u.iterrows():
        if r["CED_PADRE"] != 0:
            pares.append((r["hogar_id"], r["CED_PADRE"]))
        if r["CED_MADRE"] != 0:
            pares.append((r["hogar_id"], r["CED_MADRE"]))
    if not pares:
        return set()

    df_mapa = pd.DataFrame(pares, columns=["hogar_id", "fam_id"]).drop_duplicates()

    emp = _empleo_reciente(empleo)
    if emp.empty:
        df_emp = df_mapa.copy()
        df_emp["tipo_empleo"] = "Desconocido"
        df_emp["trabaja"] = False
    else:
        df_emp = df_mapa.merge(
            emp[["IDENTIFICACION", "TIPO_EMPRESA"]],
            left_on="fam_id",
            right_on="IDENTIFICACION",
            how="left",
            indicator=True,
        )
        tipo_emp_norm = df_emp["TIPO_EMPRESA"].astype(str).str.strip()
        df_emp["tipo_empleo"] = np.where(
            df_emp["_merge"] != "both",
            "Desconocido",
            np.where(
                tipo_emp_norm == AFILIACION_VOLUNTARIA_CODE,
                "Afiliacion Voluntaria",
                "Relacion de Dependencia",
            ),
        )
        df_emp["trabaja"] = df_emp["_merge"] == "both"

    if tipo_empleo_sel != "Todos":
        df_emp = df_emp[df_emp["tipo_empleo"] == tipo_empleo_sel]
    if df_emp.empty:
        return set()

    agg = df_emp.groupby("hogar_id", as_index=False).agg(n_trab=("trabaja", "sum"))
    if cant_papas_trab in (0, 1, 2):
        agg = agg[agg["n_trab"] == cant_papas_trab]
    if agg.empty:
        return set()

    hogares_ok = set(agg["hogar_id"].unique().tolist())
    fam_ok = set(df_emp.loc[df_emp["hogar_id"].isin(hogares_ok), "fam_id"].unique())
    return fam_ok


def crear_pie_chart_tipos_deuda(
    df_deudas: pd.DataFrame, grupo_label: str
) -> go.Figure | None:
    if df_deudas.empty:
        return None

    df = df_deudas.copy()
    df["VALOR"] = pd.to_numeric(df["VALOR"], errors="coerce").fillna(0)

    tipo = df["TIPO"].fillna("").astype(str).str.strip()
    cod_tipo = df["COD_TIPO"].fillna("").astype(str).str.strip()
    df["TIPO_FINAL"] = tipo.where(tipo != "", cod_tipo)
    df["TIPO_FINAL"] = df["TIPO_FINAL"].replace("", "Sin tipo")

    top_deudas = df.groupby("TIPO_FINAL")["VALOR"].sum().reset_index()
    top_deudas = top_deudas.sort_values("VALOR", ascending=False).head(10)

    if top_deudas.empty:
        return None

    k = min(10, len(top_deudas))

    fig = px.pie(
        top_deudas,
        values="VALOR",
        names="TIPO_FINAL",
        title=f"Top 10 Tipos de Deuda - {grupo_label}",
        color_discrete_sequence=PALETA_AZUL_PASTEL_10[:k],
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>"
        + "Valor: $%{value:,.0f}<br>"
        + "Porcentaje: %{percent}<br>"
        + "<extra></extra>",
        marker=dict(line=dict(color="white", width=1)),
    )

    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
    )

    return fig


def crear_bar_chart_calificacion_descriptiva(
    df_deudas: pd.DataFrame, grupo_label: str
) -> go.Figure | None:
    if df_deudas.empty:
        return None

    df = df_deudas.copy()
    df["VALOR"] = pd.to_numeric(df["VALOR"], errors="coerce").fillna(0)

    df["COD_CAL_NORM"] = (
        df["COD_CALIFICACION"]
        .astype(str)
        .str.upper()
        .str.strip()
        .replace({"": np.nan, "NAN": np.nan})
    )
    df = df.dropna(subset=["COD_CAL_NORM"])

    mapa_desc = {
        "A1": "Riesgo estable",
        "A2": "Riesgo moderado",
        "A3": "Riesgo moderado",
        "AL": "Riesgo moderado",
        "B1": "Riesgo moderado",
        "B2": "Riesgo moderado",
        "C1": "Riesgo moderado",
        "C2": "Riesgo moderado",
        "D": "Alto Riesgo",
        "E": "Alto Riesgo",
    }

    df["CALIFICACION_DESC"] = (
        df["COD_CAL_NORM"].map(mapa_desc).fillna("Desconocido")
    )

    df_grouped = df.groupby("CALIFICACION_DESC", as_index=False).agg(
        conteo_deudas=("VALOR", "count"), valor_total=("VALOR", "sum")
    )
    df_grouped = df_grouped.sort_values("conteo_deudas", ascending=False)

    if df_grouped.empty:
        return None

    colores = {
        "Riesgo estable": "rgba(0,112,192,1.0)",
        "Riesgo moderado": "rgba(0,112,192,0.70)",
        "Alto Riesgo": "rgba(0,112,192,0.40)",
        "Desconocido": "rgba(0,112,192,0.20)",
    }

    fig = px.bar(
        df_grouped,
        x="conteo_deudas",
        y="CALIFICACION_DESC",
        orientation="h",
        title=f"Deudas por Calificacion Crediticia - {grupo_label}",
        text="conteo_deudas",
        color="CALIFICACION_DESC",
        color_discrete_map=colores,
        hover_data={"valor_total": ":$,.0f", "conteo_deudas": True},
    )

    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="inside",
        hovertemplate="<b>%{y}</b><br>"
        "Numero de deudas: %{x:,}<br>"
        "Valor total: %{customdata[0]:$,.0f}<extra></extra>",
    )
    fig.update_layout(
        xaxis_title="Numero de Deudas",
        yaxis_title="Calificacion Crediticia",
        height=350,
        showlegend=False,
        margin=dict(l=80, r=50, t=80, b=60),
        plot_bgcolor="white",
        title=dict(x=0.5, xanchor="center", font=dict(size=14)),
    )
    return fig


st.set_page_config(page_title="Deudas", page_icon="💰", layout="wide")
st.title("💰 Deudas")

with st.spinner("Cargando datos..."):
    excel_filename = get_active_excel_filename()
    estudiantes, universo_familiares, empleo, deudas = load_data(excel_filename)

empleo = _filtrar_mes(empleo, ANIO_CORTE, MES_CORTE)
deudas = _filtrar_mes(deudas, ANIO_CORTE, MES_CORTE)

ids_estudiantes = set(estudiantes["IDENTIFICACION"].dropna().astype(int).unique().tolist())

grupo_label = "Estudiantes"

c1, c2, c3 = st.columns(3)
with c1:
    cant_papas_opt = st.selectbox(
        "Cantidad de papas en el hogar",
        options=["Todos", 1, 2],
        index=0,
    )
    cant_papas = None if cant_papas_opt == "Todos" else int(cant_papas_opt)
with c2:
    cant_papas_trab_opt = st.selectbox(
        "Cantidad de papas trabajando",
        options=["Todos", 0, 1, 2],
        index=0,
    )
    cant_papas_trab = (
        None if cant_papas_trab_opt == "Todos" else int(cant_papas_trab_opt)
    )
with c3:
    tipo_empleo_sel = st.selectbox(
        "Tipo de empleo",
        options=TIPOS_EMPLEO,
        index=0,
    )

ids_grupo = construir_familiares_filtrado(
    universo_familiares,
    empleo,
    ids_base=ids_estudiantes,
    cant_papas=cant_papas,
    cant_papas_trab=cant_papas_trab,
    tipo_empleo_sel=tipo_empleo_sel,
)

df_deudas_filtrado = deudas[deudas["IDENTIFICACION"].isin(ids_grupo)].copy()

if df_deudas_filtrado.empty:
    st.info("No hay datos de deudas disponibles para el grupo seleccionado.")
    st.stop()

fig_pie = crear_pie_chart_tipos_deuda(df_deudas_filtrado, grupo_label)
if fig_pie:
    st.plotly_chart(fig_pie, use_container_width=True)

fig_bar = crear_bar_chart_calificacion_descriptiva(df_deudas_filtrado, grupo_label)
if fig_bar:
    st.plotly_chart(fig_bar, use_container_width=True)
