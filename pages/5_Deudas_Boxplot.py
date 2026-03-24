import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.excel_loader import get_active_excel_filename, load_excel_sheet
from utils.student_columns import normalize_university_column
from utils.student_filters import render_student_academic_filters


ANIO_CORTE = 2025
MES_CORTE = 11
AFILIACION_VOLUNTARIA_CODE = "32-AFILIACION VOLUNTARIA(TIPEM-32)"
TIPOS_EMPLEO = [
    "Todos",
    "Relacion de Dependencia",
    "Afiliacion Voluntaria",
    "Desconocido",
]

QUINTILES = {
    1: {"min": 1.13, "max": 642.03},
    2: {"min": 642.04, "max": 909.07},
    3: {"min": 909.09, "max": 1415.89},
    4: {"min": 1415.92, "max": 2491.60},
    5: {"min": 2491.61, "max": 20009.99},
}

QUINTIL_ORDER = [
    "Sin informacion de empleo",
    "Quintil 1",
    "Quintil 2",
    "Quintil 3",
    "Quintil 4",
    "Quintil 5",
]


def _parse_valor_deuda(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    has_comma = s.str.contains(",", regex=False)
    s = s.where(
        ~has_comma,
        s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
    )
    return pd.to_numeric(s, errors="coerce").fillna(0)


def _normalizar_universidad(df: pd.DataFrame) -> pd.DataFrame:
    return normalize_university_column(df)


@st.cache_data(show_spinner=False)
def load_data(excel_filename: str) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    estudiantes = load_excel_sheet("Estudiantes", excel_filename)
    universo_familiares = load_excel_sheet("Universo Familiares", excel_filename)
    empleo = load_excel_sheet("Empleos", excel_filename)
    deudas = load_excel_sheet("Deudas", excel_filename)

    if "Cedula" in estudiantes.columns:
        estudiantes = estudiantes.rename(columns={"Cedula": "IDENTIFICACION"})
    elif "CEDULA" in estudiantes.columns:
        estudiantes = estudiantes.rename(columns={"CEDULA": "IDENTIFICACION"})
    estudiantes = _normalizar_universidad(estudiantes)

    return estudiantes, universo_familiares, empleo, deudas


def _normalizar_ids_familia(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["CED_PADRE", "CED_MADRE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def _empleo_agrupado(empleo: pd.DataFrame) -> pd.DataFrame:
    if empleo.empty:
        return empleo.copy()
    emp = empleo.copy()
    emp["IDENTIFICACION"] = pd.to_numeric(emp["IDENTIFICACION"], errors="coerce")
    emp = emp.dropna(subset=["IDENTIFICACION"]).copy()
    emp["IDENTIFICACION"] = emp["IDENTIFICACION"].astype(int)
    emp["SALARIO"] = pd.to_numeric(emp["SALARIO"], errors="coerce").fillna(0)
    emp["TIPO_EMPRESA"] = emp["TIPO_EMPRESA"].astype(str).str.strip()
    emp["ES_AFILIACION_VOL"] = emp["TIPO_EMPRESA"] == AFILIACION_VOLUNTARIA_CODE

    agg = emp.groupby("IDENTIFICACION", as_index=False).agg(
        SALARIO=("SALARIO", "sum"),
        ES_AFILIACION_VOL=("ES_AFILIACION_VOL", "all"),
    )
    agg["tipo_empleo"] = np.where(
        agg["ES_AFILIACION_VOL"], "Afiliacion Voluntaria", "Relacion de Dependencia"
    )
    return agg


def _filtrar_mes(df: pd.DataFrame, anio: int, mes: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df[(df["ANIO"] == anio) & (df["MES"] == mes)].copy()


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


def construir_hogares_familia(
    universo_familiares: pd.DataFrame,
    empleo: pd.DataFrame,
    ids_base: set[int],
    cant_papas: int | None,
    cant_papas_trab: int | None,
    tipo_empleo_sel: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    u = universo_familiares[universo_familiares["IDENTIFICACION"].isin(ids_base)].copy()
    if u.empty:
        return (
            pd.DataFrame(
                columns=["hogar_id", "salario_hogar", "QUINTIL", "GRUPO_QUINTIL"]
            ),
            pd.DataFrame(columns=["hogar_id", "fam_id"]),
        )

    u = _normalizar_ids_familia(u)
    u["n_papas"] = (u["CED_PADRE"] != 0).astype(int) + (u["CED_MADRE"] != 0).astype(int)
    u = u[u["n_papas"] > 0]
    if cant_papas in (1, 2):
        u = u[u["n_papas"] == cant_papas]
    if u.empty:
        return (
            pd.DataFrame(
                columns=["hogar_id", "salario_hogar", "QUINTIL", "GRUPO_QUINTIL"]
            ),
            pd.DataFrame(columns=["hogar_id", "fam_id"]),
        )

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
        return (
            pd.DataFrame(
                columns=["hogar_id", "salario_hogar", "QUINTIL", "GRUPO_QUINTIL"]
            ),
            pd.DataFrame(columns=["hogar_id", "fam_id"]),
        )

    df_mapa = pd.DataFrame(pares, columns=["hogar_id", "fam_id"]).drop_duplicates()

    emp = _empleo_agrupado(empleo)
    if emp.empty:
        df_emp = df_mapa.copy()
        df_emp["tipo_empleo"] = "Desconocido"
        df_emp["trabaja"] = False
    else:
        df_emp = df_mapa.merge(
            emp[["IDENTIFICACION", "tipo_empleo"]],
            left_on="fam_id",
            right_on="IDENTIFICACION",
            how="left",
            indicator=True,
        )
        df_emp["tipo_empleo"] = df_emp["tipo_empleo"].where(
            df_emp["_merge"] == "both", "Desconocido"
        )
        df_emp["trabaja"] = df_emp["_merge"] == "both"

    if tipo_empleo_sel != "Todos":
        df_emp = df_emp[df_emp["tipo_empleo"] == tipo_empleo_sel]
    if df_emp.empty:
        return (
            pd.DataFrame(
                columns=["hogar_id", "salario_hogar", "QUINTIL", "GRUPO_QUINTIL"]
            ),
            pd.DataFrame(columns=["hogar_id", "fam_id"]),
        )

    agg = df_emp.groupby("hogar_id", as_index=False).agg(n_trab=("trabaja", "sum"))
    if cant_papas_trab in (0, 1, 2):
        agg = agg[agg["n_trab"] == cant_papas_trab]
    if agg.empty:
        return (
            pd.DataFrame(
                columns=["hogar_id", "salario_hogar", "QUINTIL", "GRUPO_QUINTIL"]
            ),
            pd.DataFrame(columns=["hogar_id", "fam_id"]),
        )

    hogares_ok = set(agg["hogar_id"].unique().tolist())
    df_mapa_ok = df_mapa[df_mapa["hogar_id"].isin(hogares_ok)].copy()
    if df_mapa_ok.empty:
        return (
            pd.DataFrame(
                columns=["hogar_id", "salario_hogar", "QUINTIL", "GRUPO_QUINTIL"]
            ),
            pd.DataFrame(columns=["hogar_id", "fam_id"]),
        )

    if emp.empty:
        df_hogares = (
            df_mapa_ok.groupby("hogar_id", as_index=False)
            .agg(salario_hogar=("fam_id", "size"))
            .assign(salario_hogar=0)
        )
    else:
        emp_sal = emp.copy()
        emp_sal["IDENTIFICACION"] = pd.to_numeric(
            emp_sal["IDENTIFICACION"], errors="coerce"
        )
        emp_sal = emp_sal.dropna(subset=["IDENTIFICACION"]).copy()
        emp_sal["IDENTIFICACION"] = emp_sal["IDENTIFICACION"].astype(int)
        emp_sal["SALARIO"] = pd.to_numeric(emp_sal["SALARIO"], errors="coerce").fillna(
            0
        )
        df_sal = df_mapa_ok.merge(
            emp_sal[["IDENTIFICACION", "SALARIO"]],
            left_on="fam_id",
            right_on="IDENTIFICACION",
            how="left",
        )
        df_sal["SALARIO"] = df_sal["SALARIO"].fillna(0)
        df_hogares = (
            df_sal.groupby("hogar_id", as_index=False)["SALARIO"]
            .sum()
            .rename(columns={"SALARIO": "salario_hogar"})
        )

    df_hogares["QUINTIL"] = df_hogares["salario_hogar"].apply(asignar_quintil)
    df_hogares["GRUPO_QUINTIL"] = df_hogares["QUINTIL"].apply(
        lambda q: f"Quintil {int(q)}" if pd.notna(q) else "Sin informacion de empleo"
    )

    return df_hogares, df_mapa_ok


st.set_page_config(
    page_title="Deudas por Quintil - Boxplot", page_icon="D", layout="wide"
)
st.title("Deudas por Quintil - Boxplot")

with st.spinner("Cargando datos..."):
    excel_filename = get_active_excel_filename()
    estudiantes, universo_familiares, empleo, deudas = load_data(excel_filename)

empleo = _filtrar_mes(empleo, ANIO_CORTE, MES_CORTE)
deudas = _filtrar_mes(deudas, ANIO_CORTE, MES_CORTE)

estudiantes_filtrados = estudiantes
st.markdown("### Filtros academicos")
estudiantes_filtrados, _filtros_estudiantes = render_student_academic_filters(
    estudiantes, key_prefix="deudas_boxplot"
)

ids_estudiantes = set(
    estudiantes_filtrados["IDENTIFICACION"].dropna().astype(int).unique().tolist()
)

cant_papas = None
cant_papas_trab = None
tipo_empleo_sel = "Todos"

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

ids_base = ids_estudiantes
df_hogares, df_mapa = construir_hogares_familia(
    universo_familiares,
    empleo=empleo,
    ids_base=ids_base,
    cant_papas=cant_papas,
    cant_papas_trab=cant_papas_trab,
    tipo_empleo_sel=tipo_empleo_sel,
)

if df_hogares.empty or df_mapa.empty:
    st.info("No hay datos disponibles para el grupo seleccionado.")
    st.stop()

df_deudas = deudas.merge(
    df_mapa, left_on="IDENTIFICACION", right_on="fam_id", how="inner"
).copy()
df_deudas = df_deudas.merge(
    df_hogares[["hogar_id", "GRUPO_QUINTIL"]], on="hogar_id", how="left"
)

if df_deudas.empty:
    st.info("No hay datos de deudas para el grupo seleccionado.")
    st.stop()

df_deudas["VALOR"] = _parse_valor_deuda(df_deudas["VALOR"])
df_plot = (
    df_deudas.groupby(["GRUPO_QUINTIL", "hogar_id"], as_index=False)["VALOR"]
    .sum()
    .copy()
)
df_plot["GRUPO_QUINTIL"] = pd.Categorical(
    df_plot["GRUPO_QUINTIL"], categories=QUINTIL_ORDER, ordered=True
)
df_plot = df_plot.dropna(subset=["GRUPO_QUINTIL"])

if df_plot.empty:
    st.info("No hay datos de quintiles disponibles para el boxplot.")
    st.stop()

mostrar_outliers = st.toggle("Mostrar outliers", value=True)
points_mode = "outliers" if mostrar_outliers else False

if not mostrar_outliers:

    def _clip_grupo(group: pd.DataFrame) -> pd.DataFrame:
        q1 = group["VALOR"].quantile(0.25)
        q3 = group["VALOR"].quantile(0.75)
        iqr = q3 - q1
        lower = max(0, q1 - 1.5 * iqr)
        upper = q3 + 1.5 * iqr
        group = group.copy()
        group["VALOR"] = group["VALOR"].clip(lower=lower, upper=upper)
        return group

    df_plot_chart = df_plot.groupby("GRUPO_QUINTIL", group_keys=False).apply(
        _clip_grupo
    )
else:
    df_plot_chart = df_plot

fig = px.box(
    df_plot_chart,
    x="GRUPO_QUINTIL",
    y="VALOR",
    color="GRUPO_QUINTIL",
    title="Distribucion de la deuda por quintil - Estudiantes - Familiares",
    points=points_mode,
    category_orders={"GRUPO_QUINTIL": QUINTIL_ORDER},
)

fig.update_layout(
    height=520,
    xaxis_title="Quintil",
    yaxis_title="Valor de deuda",
    showlegend=False,
    margin=dict(l=60, r=30, t=60, b=60),
)

st.plotly_chart(fig, use_container_width=True)
