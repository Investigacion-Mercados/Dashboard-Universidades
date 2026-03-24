import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import gaussian_kde

from utils.excel_loader import load_excel_sheet
from utils.student_columns import normalize_university_column


st.set_page_config(page_title="Distribucion de hijos", layout="wide")
st.title("Distribucion de hijos")
st.caption("Estudiantes - Familiares (cantidad de hijos).")


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


def _extraer_ids_familiares(
    universo_df: pd.DataFrame, ids_estudiantes: set[int]
) -> set[int]:
    if not ids_estudiantes or universo_df.empty:
        return set()

    if "IDENTIFICACION" not in universo_df.columns:
        return set()

    df = universo_df[universo_df["IDENTIFICACION"].isin(ids_estudiantes)].copy()
    if df.empty:
        return set()

    familiares = set()
    for col in ["CED_PADRE", "CED_MADRE"]:
        if col not in df.columns:
            continue
        valores = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
        familiares.update([v for v in valores.tolist() if v > 0])
    return familiares


def _info_hijos(info_df: pd.DataFrame, ids: set[int]) -> pd.DataFrame:
    if not ids or info_df.empty:
        return pd.DataFrame(columns=["IDENTIFICACION", "HIJOS"])
    if "IDENTIFICACION" not in info_df.columns or "HIJOS" not in info_df.columns:
        return pd.DataFrame(columns=["IDENTIFICACION", "HIJOS"])

    df = info_df.copy()
    df["IDENTIFICACION"] = pd.to_numeric(df["IDENTIFICACION"], errors="coerce")
    df = df.dropna(subset=["IDENTIFICACION"]).copy()
    df["IDENTIFICACION"] = df["IDENTIFICACION"].astype(int)
    df = df[df["IDENTIFICACION"].isin(ids)]
    if df.empty:
        return pd.DataFrame(columns=["IDENTIFICACION", "HIJOS"])

    df["HIJOS"] = pd.to_numeric(df["HIJOS"], errors="coerce").fillna(0)
    df = df[df["HIJOS"] >= 0]
    return df[["IDENTIFICACION", "HIJOS"]].copy()


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
ids_familiares = _extraer_ids_familiares(universo_familiares, ids_estudiantes)
df_hijos = _info_hijos(info_personal, ids_familiares)

if df_hijos.empty:
    st.info("No hay datos suficientes para construir el grafico de dispersion.")
    st.stop()

hijos_familiares = df_hijos["HIJOS"].to_numpy()

st.metric("Estudiantes - Familiares", int(len(hijos_familiares)))

max_val = 0
if len(hijos_familiares) > 0:
    max_val = max(max_val, int(np.max(hijos_familiares)))
max_val = max(max_val, 5)

fig = go.Figure()

if len(hijos_familiares) >= 2:
    x_min = max(0, int(np.percentile(hijos_familiares, 1)))
    x_max = max(int(np.percentile(hijos_familiares, 99)) + 1, 5)
else:
    x_min, x_max = 0, max_val

x_vals = np.linspace(x_min, x_max, 1000)
x_tab = np.arange(x_min, x_max + 1, 1)
curvas_tab: dict[str, np.ndarray] = {}

if len(hijos_familiares) >= 2:
    kde = gaussian_kde(hijos_familiares, bw_method=0.25)
    y_est = kde(x_vals) * 100
    curvas_tab["Estudiantes - Familiares (%)"] = kde(x_tab) * 100
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_est,
            mode="lines",
            name="Estudiantes - Familiares",
            line=dict(color="#3498db", width=3),
            fill="tozeroy",
            fillcolor=_hex_to_rgba("#3498db", 0.18),
            hovertemplate="Hijos: %{x:.1f}<br>Participacion: %{y:.2f}%<extra></extra>",
        )
    )
else:
    st.info("No hay suficientes datos para la curva de familiares.")

if len(fig.data) == 0:
    st.info("No hay datos suficientes para construir el grafico de Gauss.")
    st.stop()

fig.update_layout(
    height=520,
    margin=dict(l=40, r=20, t=30, b=40),
    xaxis_title="Cantidad de hijos",
    yaxis_title="Participacion (%)",
    xaxis=dict(range=[x_min, x_max]),
    legend_title="Grupo",
)

st.plotly_chart(fig, use_container_width=True)

if curvas_tab:
    dx = float(x_tab[1] - x_tab[0]) if len(x_tab) > 1 else 1.0
    tabla = pd.DataFrame({"Hijos": x_tab.astype(int)})
    for nombre, y_vals in curvas_tab.items():
        area_vals = y_vals * dx
        total_area = float(area_vals.sum())
        if total_area > 0:
            tabla[nombre] = np.round(area_vals / total_area * 100, 6)
        else:
            tabla[nombre] = 0.0
    total_row = {"Hijos": "Total"}
    for nombre in curvas_tab.keys():
        total_row[nombre] = float(tabla[nombre].sum())
    tabla = pd.concat([tabla, pd.DataFrame([total_row])], ignore_index=True)

    st.markdown("---")
    st.subheader("Tabla de participacion (%)")
    tabla_display = tabla.copy()
    tabla_display["Hijos"] = tabla_display["Hijos"].astype(str)
    for nombre in curvas_tab.keys():
        tabla_display[nombre] = tabla_display[nombre].apply(
            lambda v: (
                f"{v:.6f}".replace(".", ",")
                if isinstance(v, (int, float, np.floating))
                else v
            )
        )
    st.dataframe(tabla_display, use_container_width=True, hide_index=True)
