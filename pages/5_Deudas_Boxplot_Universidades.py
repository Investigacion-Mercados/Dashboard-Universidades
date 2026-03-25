import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.excel_loader import load_excel_sheet
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
SOURCE_FILES = {
    "UDLA": "Udla.xlsx",
    "Universidades": "Universidades.xlsx",
}

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

QUINTIL_COLORS = {
    "Sin informacion de empleo": "#95a5a6",
    "Quintil 1": "#e74c3c",
    "Quintil 2": "#e67e22",
    "Quintil 3": "#f39c12",
    "Quintil 4": "#2ecc71",
    "Quintil 5": "#27ae60",
}


def _rgba(hex_color: str, alpha: float) -> str:
    color = hex_color.strip().lstrip("#")
    if len(color) != 6:
        return f"rgba(0,0,0,{alpha})"
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


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


def _normalizar_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def _normalizar_estudiantes(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    estudiantes = df.copy()
    if "Cedula" in estudiantes.columns:
        estudiantes = estudiantes.rename(columns={"Cedula": "IDENTIFICACION"})
    elif "CEDULA" in estudiantes.columns:
        estudiantes = estudiantes.rename(columns={"CEDULA": "IDENTIFICACION"})

    estudiantes = _normalizar_universidad(estudiantes)
    if "Universidad" not in estudiantes.columns:
        estudiantes["Universidad"] = source_name

    estudiantes["IDENTIFICACION"] = _normalizar_id(estudiantes["IDENTIFICACION"])
    estudiantes["fuente_archivo"] = source_name
    estudiantes = estudiantes[estudiantes["IDENTIFICACION"] != 0].copy()
    return estudiantes


def _normalizar_universo_familiares(
    df: pd.DataFrame, source_name: str
) -> pd.DataFrame:
    familia = df.copy()
    familia["IDENTIFICACION"] = _normalizar_id(familia["IDENTIFICACION"])
    for col in ["CED_PADRE", "CED_MADRE"]:
        familia[col] = _normalizar_id(familia[col])
    familia["fuente_archivo"] = source_name
    familia = familia[familia["IDENTIFICACION"] != 0].copy()
    return familia


def _normalizar_empleo(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    empleo = df.copy()
    empleo["IDENTIFICACION"] = _normalizar_id(empleo["IDENTIFICACION"])
    empleo["SALARIO"] = pd.to_numeric(empleo["SALARIO"], errors="coerce").fillna(0)
    empleo["fuente_archivo"] = source_name
    empleo = empleo[empleo["IDENTIFICACION"] != 0].copy()
    return empleo


def _normalizar_deudas(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    deudas = df.copy()
    deudas["IDENTIFICACION"] = _normalizar_id(deudas["IDENTIFICACION"])
    deudas["fuente_archivo"] = source_name
    deudas = deudas[deudas["IDENTIFICACION"] != 0].copy()
    return deudas


@st.cache_data(show_spinner=False)
def load_combined_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    students_frames = []
    familia_frames = []
    empleo_frames = []
    deuda_frames = []

    for source_name, filename in SOURCE_FILES.items():
        students_frames.append(
            _normalizar_estudiantes(load_excel_sheet("Estudiantes", filename), source_name)
        )
        familia_frames.append(
            _normalizar_universo_familiares(
                load_excel_sheet("Universo Familiares", filename), source_name
            )
        )
        empleo_frames.append(
            _normalizar_empleo(load_excel_sheet("Empleos", filename), source_name)
        )
        deuda_frames.append(
            _normalizar_deudas(load_excel_sheet("Deudas", filename), source_name)
        )

    estudiantes = pd.concat(students_frames, ignore_index=True, sort=False)
    universo_familiares = pd.concat(familia_frames, ignore_index=True, sort=False)
    empleo = pd.concat(empleo_frames, ignore_index=True, sort=False)
    deudas = pd.concat(deuda_frames, ignore_index=True, sort=False)

    return estudiantes, universo_familiares, empleo, deudas


def _filtrar_mes(df: pd.DataFrame, anio: int, mes: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df[(df["ANIO"] == anio) & (df["MES"] == mes)].copy()


def _empleo_agrupado(empleo: pd.DataFrame) -> pd.DataFrame:
    if empleo.empty:
        return empleo.copy()

    emp = empleo.copy()
    emp["TIPO_EMPRESA"] = emp["TIPO_EMPRESA"].astype(str).str.strip()
    emp["ES_AFILIACION_VOL"] = emp["TIPO_EMPRESA"] == AFILIACION_VOLUNTARIA_CODE

    agg = emp.groupby(["fuente_archivo", "IDENTIFICACION"], as_index=False).agg(
        SALARIO=("SALARIO", "sum"),
        ES_AFILIACION_VOL=("ES_AFILIACION_VOL", "all"),
    )
    agg["tipo_empleo"] = np.where(
        agg["ES_AFILIACION_VOL"], "Afiliacion Voluntaria", "Relacion de Dependencia"
    )
    return agg


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
    estudiantes_base: pd.DataFrame,
    universo_familiares: pd.DataFrame,
    empleo: pd.DataFrame,
    cant_papas: int | None,
    cant_papas_trab: int | None,
    tipo_empleo_sel: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ids_base = estudiantes_base[
        ["fuente_archivo", "IDENTIFICACION", "Universidad"]
    ].drop_duplicates()
    if ids_base.empty:
        return (
            pd.DataFrame(columns=["hogar_id", "salario_hogar", "QUINTIL", "GRUPO_QUINTIL"]),
            pd.DataFrame(columns=["fuente_archivo", "hogar_id", "fam_id"]),
            pd.DataFrame(columns=["hogar_id", "Universidad"]),
        )

    u = universo_familiares.merge(
        ids_base[["fuente_archivo", "IDENTIFICACION"]],
        on=["fuente_archivo", "IDENTIFICACION"],
        how="inner",
    )
    if u.empty:
        return (
            pd.DataFrame(columns=["hogar_id", "salario_hogar", "QUINTIL", "GRUPO_QUINTIL"]),
            pd.DataFrame(columns=["fuente_archivo", "hogar_id", "fam_id"]),
            pd.DataFrame(columns=["hogar_id", "Universidad"]),
        )

    u["n_papas"] = (u["CED_PADRE"] != 0).astype(int) + (u["CED_MADRE"] != 0).astype(int)
    u = u[u["n_papas"] > 0]
    if cant_papas in (1, 2):
        u = u[u["n_papas"] == cant_papas]
    if u.empty:
        return (
            pd.DataFrame(columns=["hogar_id", "salario_hogar", "QUINTIL", "GRUPO_QUINTIL"]),
            pd.DataFrame(columns=["fuente_archivo", "hogar_id", "fam_id"]),
            pd.DataFrame(columns=["hogar_id", "Universidad"]),
        )

    u["hogar_id"] = u.apply(
        lambda r: (
            f"{r['fuente_archivo']}|"
            f"{'|'.join(sorted([str(r['CED_PADRE']), str(r['CED_MADRE'])]))}"
        ),
        axis=1,
    )

    pares = []
    for _, row in u.iterrows():
        if row["CED_PADRE"] != 0:
            pares.append((row["fuente_archivo"], row["hogar_id"], row["CED_PADRE"]))
        if row["CED_MADRE"] != 0:
            pares.append((row["fuente_archivo"], row["hogar_id"], row["CED_MADRE"]))
    if not pares:
        return (
            pd.DataFrame(columns=["hogar_id", "salario_hogar", "QUINTIL", "GRUPO_QUINTIL"]),
            pd.DataFrame(columns=["fuente_archivo", "hogar_id", "fam_id"]),
            pd.DataFrame(columns=["hogar_id", "Universidad"]),
        )

    df_mapa = pd.DataFrame(
        pares, columns=["fuente_archivo", "hogar_id", "fam_id"]
    ).drop_duplicates()

    emp = _empleo_agrupado(empleo)
    if emp.empty:
        df_emp = df_mapa.copy()
        df_emp["tipo_empleo"] = "Desconocido"
        df_emp["trabaja"] = False
    else:
        df_emp = df_mapa.merge(
            emp[["fuente_archivo", "IDENTIFICACION", "tipo_empleo"]],
            left_on=["fuente_archivo", "fam_id"],
            right_on=["fuente_archivo", "IDENTIFICACION"],
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
            pd.DataFrame(columns=["hogar_id", "salario_hogar", "QUINTIL", "GRUPO_QUINTIL"]),
            pd.DataFrame(columns=["fuente_archivo", "hogar_id", "fam_id"]),
            pd.DataFrame(columns=["hogar_id", "Universidad"]),
        )

    agg = df_emp.groupby("hogar_id", as_index=False).agg(n_trab=("trabaja", "sum"))
    if cant_papas_trab in (0, 1, 2):
        agg = agg[agg["n_trab"] == cant_papas_trab]
    if agg.empty:
        return (
            pd.DataFrame(columns=["hogar_id", "salario_hogar", "QUINTIL", "GRUPO_QUINTIL"]),
            pd.DataFrame(columns=["fuente_archivo", "hogar_id", "fam_id"]),
            pd.DataFrame(columns=["hogar_id", "Universidad"]),
        )

    hogares_ok = set(agg["hogar_id"].unique().tolist())
    df_mapa_ok = df_mapa[df_mapa["hogar_id"].isin(hogares_ok)].copy()
    if df_mapa_ok.empty:
        return (
            pd.DataFrame(columns=["hogar_id", "salario_hogar", "QUINTIL", "GRUPO_QUINTIL"]),
            pd.DataFrame(columns=["fuente_archivo", "hogar_id", "fam_id"]),
            pd.DataFrame(columns=["hogar_id", "Universidad"]),
        )

    if emp.empty:
        df_hogares = (
            df_mapa_ok.groupby("hogar_id", as_index=False)
            .agg(salario_hogar=("fam_id", "size"))
            .assign(salario_hogar=0)
        )
    else:
        df_sal = df_mapa_ok.merge(
            emp[["fuente_archivo", "IDENTIFICACION", "SALARIO"]],
            left_on=["fuente_archivo", "fam_id"],
            right_on=["fuente_archivo", "IDENTIFICACION"],
            how="left",
        )
        df_sal["SALARIO"] = df_sal["SALARIO"].fillna(0)
        df_hogares = (
            df_sal.groupby("hogar_id", as_index=False)["SALARIO"]
            .sum()
            .rename(columns={"SALARIO": "salario_hogar"})
        )

    df_hogar_universidad = (
        u.merge(
            ids_base[["fuente_archivo", "IDENTIFICACION", "Universidad"]],
            on=["fuente_archivo", "IDENTIFICACION"],
            how="left",
        )[["hogar_id", "Universidad"]]
        .dropna(subset=["Universidad"])
        .drop_duplicates()
    )

    df_hogares["QUINTIL"] = df_hogares["salario_hogar"].apply(asignar_quintil)
    df_hogares["GRUPO_QUINTIL"] = df_hogares["QUINTIL"].apply(
        lambda q: f"Quintil {int(q)}" if pd.notna(q) else "Sin informacion de empleo"
    )

    return df_hogares, df_mapa_ok, df_hogar_universidad


def construir_boxplot_universidades(
    df_box: pd.DataFrame,
    df_universidad: pd.DataFrame,
    *,
    mostrar_outliers: bool,
    mostrar_etiquetas: bool,
) -> go.Figure:
    x_positions = {grupo: idx for idx, grupo in enumerate(QUINTIL_ORDER)}

    if not mostrar_outliers:
        def _limites_grupo(group: pd.DataFrame) -> pd.Series:
            q1 = group["VALOR"].quantile(0.25)
            q3 = group["VALOR"].quantile(0.75)
            iqr = q3 - q1
            lower = max(0, q1 - 1.5 * iqr)
            upper = q3 + 1.5 * iqr
            return pd.Series({"lower": lower, "upper": upper})

        limites = (
            df_box.groupby("GRUPO_QUINTIL", observed=False)
            .apply(_limites_grupo)
            .reset_index()
        )

        df_box_chart = df_box.merge(limites, on="GRUPO_QUINTIL", how="left")
        df_box_chart["VALOR"] = df_box_chart["VALOR"].clip(
            lower=df_box_chart["lower"], upper=df_box_chart["upper"]
        )
        df_box_chart = df_box_chart.drop(columns=["lower", "upper"])

        df_universidad_chart = df_universidad.merge(
            limites, on="GRUPO_QUINTIL", how="left"
        )
        df_universidad_chart["valor_referencia"] = df_universidad_chart[
            "valor_referencia"
        ].clip(
            lower=df_universidad_chart["lower"], upper=df_universidad_chart["upper"]
        )
        df_universidad_chart["promedio_deuda"] = df_universidad_chart[
            "promedio_deuda"
        ].clip(lower=df_universidad_chart["lower"], upper=df_universidad_chart["upper"])
        df_universidad_chart = df_universidad_chart.drop(columns=["lower", "upper"])
    else:
        df_box_chart = df_box.copy()
        df_universidad_chart = df_universidad.copy()

    fig = go.Figure()
    for grupo in QUINTIL_ORDER:
        subset = df_box_chart[df_box_chart["GRUPO_QUINTIL"] == grupo]
        if subset.empty:
            continue

        color = QUINTIL_COLORS[grupo]
        fig.add_trace(
            go.Box(
                x=[x_positions[grupo]] * len(subset),
                y=subset["VALOR"],
                name=grupo,
                boxpoints="outliers" if mostrar_outliers else False,
                marker=dict(color=color, size=4, opacity=0.35),
                line=dict(color=color, width=1.6),
                fillcolor=_rgba(color, 0.18),
                hovertemplate=(
                    f"{grupo}<br>Deuda hogar: "
                    "$%{y:,.2f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    palette = (
        px.colors.qualitative.Plotly
        + px.colors.qualitative.Dark24
        + px.colors.qualitative.Set3
    )
    universidades = sorted(df_universidad["Universidad"].dropna().unique().tolist())
    color_map = {
        universidad: palette[idx % len(palette)]
        for idx, universidad in enumerate(universidades)
    }

    rng = np.random.default_rng(7)
    point_mode = "markers+text" if mostrar_etiquetas else "markers"

    for universidad in universidades:
        subset = df_universidad_chart[
            df_universidad_chart["Universidad"] == universidad
        ].copy()
        if subset.empty:
            continue

        subset["x_plot"] = subset["GRUPO_QUINTIL"].map(x_positions).astype(float)
        subset["x_plot"] = subset["x_plot"] + rng.normal(0, 0.08, size=len(subset))
        subset["etiqueta"] = universidad
        subset["hover_text"] = subset.apply(
            lambda row: (
                f"<b>{universidad}</b><br>"
                f"Quintil: {row['GRUPO_QUINTIL']}<br>"
                f"Mediana deuda: ${row['valor_referencia']:,.2f}<br>"
                f"Promedio deuda: ${row['promedio_deuda']:,.2f}<br>"
                f"Hogares: {int(row['hogares'])}"
            ),
            axis=1,
        )

        fig.add_trace(
            go.Scatter(
                x=subset["x_plot"],
                y=subset["valor_referencia"],
                mode=point_mode,
                name=universidad,
                text=subset["etiqueta"] if mostrar_etiquetas else None,
                textposition="top center",
                marker=dict(
                    size=11,
                    color=color_map[universidad],
                    line=dict(width=0),
                ),
                hovertext=subset["hover_text"],
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Distribucion de deuda por quintil con referencia por universidad",
        height=680,
        xaxis_title="Quintil",
        yaxis_title="Valor de deuda",
        margin=dict(l=60, r=30, t=70, b=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title_text="Universidad",
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(x_positions.values()),
        ticktext=QUINTIL_ORDER,
        gridcolor="rgba(0,0,0,0.06)",
        zeroline=False,
    )
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.10)", zeroline=False)

    return fig


st.set_page_config(
    page_title="Deudas por Quintil - Universidades", page_icon="D", layout="wide"
)
st.title("Deudas por Quintil - Universidades")
st.caption(
    "Esta pagina combina siempre `db/Udla.xlsx` y `db/Universidades.xlsx`. "
    "El boxplot resume deuda total por hogar; cada punto coloreado representa "
    "la mediana de deuda de los hogares vinculados a una universidad en ese quintil."
)

with st.spinner("Cargando datos combinados..."):
    estudiantes, universo_familiares, empleo, deudas = load_combined_data()

empleo = _filtrar_mes(empleo, ANIO_CORTE, MES_CORTE)
deudas = _filtrar_mes(deudas, ANIO_CORTE, MES_CORTE)

st.markdown("### Filtros academicos")
estudiantes_filtrados, _filtros_estudiantes = render_student_academic_filters(
    estudiantes,
    key_prefix="deudas_boxplot_universidades",
    lock_single_option_keys={"universidad"},
)

if estudiantes_filtrados.empty:
    st.info("No hay estudiantes para los filtros seleccionados.")
    st.stop()

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
    cant_papas_trab = None if cant_papas_trab_opt == "Todos" else int(cant_papas_trab_opt)
with c3:
    tipo_empleo_sel = st.selectbox(
        "Tipo de empleo",
        options=TIPOS_EMPLEO,
        index=0,
    )

o1, o2 = st.columns(2)
with o1:
    mostrar_outliers = st.toggle("Mostrar outliers", value=True)
with o2:
    mostrar_etiquetas = st.toggle("Mostrar etiquetas de universidades", value=False)

df_hogares, df_mapa, df_hogar_universidad = construir_hogares_familia(
    estudiantes_filtrados,
    universo_familiares,
    empleo=empleo,
    cant_papas=cant_papas,
    cant_papas_trab=cant_papas_trab,
    tipo_empleo_sel=tipo_empleo_sel,
)

if df_hogares.empty or df_mapa.empty:
    st.info("No hay hogares disponibles para el grupo seleccionado.")
    st.stop()

if df_hogar_universidad.empty:
    st.info("No hay universidades vinculadas a los hogares seleccionados.")
    st.stop()

df_deudas = deudas.merge(
    df_mapa,
    left_on=["fuente_archivo", "IDENTIFICACION"],
    right_on=["fuente_archivo", "fam_id"],
    how="inner",
).copy()
df_deudas = df_deudas.merge(
    df_hogares[["hogar_id", "GRUPO_QUINTIL"]], on="hogar_id", how="left"
)

if df_deudas.empty:
    st.info("No hay datos de deudas para el grupo seleccionado.")
    st.stop()

df_deudas["VALOR"] = _parse_valor_deuda(df_deudas["VALOR"])
df_box = (
    df_deudas.groupby(["GRUPO_QUINTIL", "hogar_id"], as_index=False)["VALOR"]
    .sum()
    .copy()
)
df_box["GRUPO_QUINTIL"] = pd.Categorical(
    df_box["GRUPO_QUINTIL"], categories=QUINTIL_ORDER, ordered=True
)
df_box = df_box.dropna(subset=["GRUPO_QUINTIL"]).sort_values("GRUPO_QUINTIL")

if df_box.empty:
    st.info("No hay datos de quintiles disponibles para el boxplot.")
    st.stop()

df_universidad = df_box.merge(df_hogar_universidad, on="hogar_id", how="inner")
df_universidad = (
    df_universidad.groupby(["GRUPO_QUINTIL", "Universidad"], as_index=False)
    .agg(
        hogares=("hogar_id", "nunique"),
        valor_referencia=("VALOR", "median"),
        promedio_deuda=("VALOR", "mean"),
    )
    .copy()
)
df_universidad["GRUPO_QUINTIL"] = pd.Categorical(
    df_universidad["GRUPO_QUINTIL"], categories=QUINTIL_ORDER, ordered=True
)
df_universidad = df_universidad.dropna(subset=["GRUPO_QUINTIL"]).sort_values(
    ["GRUPO_QUINTIL", "Universidad"]
)

if df_universidad.empty:
    st.info("No hay puntos por universidad disponibles para el grafico.")
    st.stop()

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Universidades visibles", int(df_universidad["Universidad"].nunique()))
metric_col2.metric("Hogares con deuda", int(df_box["hogar_id"].nunique()))
metric_col3.metric("Puntos universidad-quintil", int(len(df_universidad)))

fig = construir_boxplot_universidades(
    df_box,
    df_universidad,
    mostrar_outliers=mostrar_outliers,
    mostrar_etiquetas=mostrar_etiquetas,
)

st.plotly_chart(fig, use_container_width=True)
