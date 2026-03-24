import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import gaussian_kde

from utils.excel_loader import load_excel_sheet
from utils.student_columns import normalize_university_column


st.set_page_config(page_title="Distribución de edades", layout="wide")
st.title("Distribución de edades")
st.caption("Estudiantes - Familiares.")


def _normalizar_universidad(df: pd.DataFrame) -> pd.DataFrame:
    return normalize_university_column(df)


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    estudiantes = load_excel_sheet("Estudiantes")
    universo_familiares = load_excel_sheet("Universo Familiares")
    info_personal = load_excel_sheet("Informacion Personal")

    if "Cedula" in estudiantes.columns:
        estudiantes = estudiantes.rename(columns={"Cedula": "IDENTIFICACION"})
    elif "CEDULA" in estudiantes.columns:
        estudiantes = estudiantes.rename(columns={"CEDULA": "IDENTIFICACION"})
    estudiantes = _normalizar_universidad(estudiantes)

    return estudiantes, universo_familiares, info_personal


def _coerce_ids(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna().astype(int)


def _extraer_ids_estudiantes(estudiantes_df: pd.DataFrame) -> set[int]:
    if "IDENTIFICACION" not in estudiantes_df.columns:
        return set()
    ids = _coerce_ids(estudiantes_df["IDENTIFICACION"]).unique().tolist()
    return set(ids)


def _extraer_ids_padres(
    universo_df: pd.DataFrame, ids_estudiantes: set[int]
) -> set[int]:
    if not ids_estudiantes or universo_df.empty:
        return set()

    if "IDENTIFICACION" not in universo_df.columns:
        return set()

    df = universo_df[universo_df["IDENTIFICACION"].isin(ids_estudiantes)].copy()
    if df.empty:
        return set()

    padres = set()
    for col in ["CED_PADRE", "CED_MADRE"]:
        if col not in df.columns:
            continue
        valores = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
        padres.update([v for v in valores.tolist() if v > 0])
    return padres


def _calcular_edades(info_df: pd.DataFrame, ids: set[int]) -> np.ndarray:
    if not ids or info_df.empty:
        return np.array([])

    if (
        "IDENTIFICACION" not in info_df.columns
        or "FECHA_NACIMIENTO" not in info_df.columns
    ):
        return np.array([])

    df = info_df.copy()
    df["IDENTIFICACION"] = pd.to_numeric(df["IDENTIFICACION"], errors="coerce")
    df = df.dropna(subset=["IDENTIFICACION"]).copy()
    df["IDENTIFICACION"] = df["IDENTIFICACION"].astype(int)
    df = df[df["IDENTIFICACION"].isin(ids)]
    if df.empty:
        return np.array([])

    df["FECHA_NACIMIENTO"] = pd.to_datetime(
        df["FECHA_NACIMIENTO"], errors="coerce", dayfirst=True
    )
    today = pd.Timestamp.today().normalize()
    df["EDAD"] = np.floor((today - df["FECHA_NACIMIENTO"]).dt.days / 365.25)
    df = df[df["EDAD"].notna()]
    df = df[df["EDAD"] >= 0]
    return df["EDAD"].astype(int).to_numpy()


with st.spinner("Cargando datos..."):
    estudiantes, universo_familiares, info_personal = load_data()

estudiantes_filtrados = estudiantes
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

ids_estudiantes = _extraer_ids_estudiantes(estudiantes_filtrados)
ids_padres = _extraer_ids_padres(universo_familiares, ids_estudiantes)

edades_padres = _calcular_edades(info_personal, ids_padres)

if len(edades_padres) == 0:
    st.info("No hay datos suficientes para construir el gráfico de dispersión.")
    st.stop()

st.metric("Estudiantes - Familiares", int(len(edades_padres)))


def _gauss_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std <= 0 or not np.isfinite(std):
        return np.zeros_like(x)
    coef = 1.0 / (std * np.sqrt(2 * np.pi))
    return coef * np.exp(-0.5 * ((x - mean) / std) ** 2)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = hex_color.strip().lstrip("#")
    if len(color) != 6:
        return f"rgba(0,0,0,{alpha})"
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


fig = go.Figure()

if len(edades_padres) >= 2:
    x_min = max(0, int(np.percentile(edades_padres, 1)) - 2)
    x_max = int(np.percentile(edades_padres, 99)) + 2
else:
    x_min, x_max = 0, 90

x_vals = np.linspace(x_min, x_max, 1000)
x_tab = np.arange(x_min, x_max + 1, 1)
curvas_tab: dict[str, np.ndarray] = {}

if len(edades_padres) >= 2:
    kde = gaussian_kde(edades_padres, bw_method=0.25)
    y_padres = kde(x_vals) * 100
    curvas_tab["Estudiantes - Familiares (%)"] = kde(x_tab) * 100
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_padres,
            mode="lines",
            name="Estudiantes - Familiares",
            line=dict(color="#3498db", width=3),
            fill="tozeroy",
            fillcolor=_hex_to_rgba("#3498db", 0.18),
            hovertemplate="Edad: %{x:.1f}<br>Participación: %{y:.2f}%<extra></extra>",
        )
    )
else:
    st.info("No hay suficientes datos para la curva de familiares.")

if len(fig.data) == 0:
    st.info("No hay datos suficientes para construir el gráfico de Gauss.")
    st.stop()

fig.update_layout(
    height=520,
    margin=dict(l=40, r=20, t=30, b=40),
    xaxis_title="Edad (años)",
    yaxis_title="Participación (%)",
    xaxis=dict(range=[x_min, x_max]),
    legend_title="Grupo",
)

st.plotly_chart(fig, use_container_width=True)

if curvas_tab:
    dx = float(x_tab[1] - x_tab[0]) if len(x_tab) > 1 else 1.0
    tabla = pd.DataFrame({"Edad": x_tab.astype(int)})
    for nombre, y_vals in curvas_tab.items():
        area_vals = y_vals * dx
        total_area = float(area_vals.sum())
        if total_area > 0:
            tabla[nombre] = np.round(area_vals / total_area * 100, 6)
        else:
            tabla[nombre] = 0.0
    total_row = {"Edad": "Total"}
    for nombre in curvas_tab.keys():
        total_row[nombre] = float(tabla[nombre].sum())
    tabla = pd.concat([tabla, pd.DataFrame([total_row])], ignore_index=True)

    st.markdown("---")
    st.subheader("Tabla de participacion (%)")
    tabla_display = tabla.copy()
    tabla_display["Edad"] = tabla_display["Edad"].astype(str)
    for nombre in curvas_tab.keys():
        tabla_display[nombre] = tabla_display[nombre].apply(
            lambda v: (
                f"{v:.6f}".replace(".", ",")
                if isinstance(v, (int, float, np.floating))
                else v
            )
        )
    st.dataframe(tabla_display, use_container_width=True, hide_index=True)
