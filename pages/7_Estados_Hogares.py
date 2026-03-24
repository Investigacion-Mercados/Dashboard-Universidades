import pandas as pd
import streamlit as st

from utils.excel_loader import get_active_excel_filename, load_excel_sheet
from utils.student_columns import normalize_university_column
from utils.student_filters import render_student_academic_filters


st.set_page_config(page_title="Estados de Hogares", layout="wide")
st.title("Estados de Hogares")

COLORES = [
    "#3498db",
    "#2ecc71",
    "#e67e22",
    "#9b59b6",
    "#e74c3c",
    "#1abc9c",
    "#f39c12",
    "#34495e",
]


def tarjeta_estado(titulo: str, valor: int, color: str) -> None:
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}15 0%, {color}30 100%);
            border-left: 5px solid {color};
            border-radius: 8px;
            padding: 12px 16px;
            margin: 6px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
            text-align: center;
        ">
            <div style="color: {color}; font-size: 0.95em; margin-bottom: 6px;">
                {titulo}
            </div>
            <div style="color: {color}; font-size: 1.8em; font-weight: bold;">
                {valor:,}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _normalizar_universidad(df: pd.DataFrame) -> pd.DataFrame:
    return normalize_university_column(df)


@st.cache_data(show_spinner=False)
def load_data(excel_filename: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    estudiantes = load_excel_sheet("Estudiantes", excel_filename)
    universo_familiares = load_excel_sheet("Universo Familiares", excel_filename)
    info_personal = load_excel_sheet("Informacion Personal", excel_filename)

    if "Cedula" in estudiantes.columns:
        estudiantes = estudiantes.rename(columns={"Cedula": "IDENTIFICACION"})
    elif "CEDULA" in estudiantes.columns:
        estudiantes = estudiantes.rename(columns={"CEDULA": "IDENTIFICACION"})
    estudiantes = _normalizar_universidad(estudiantes)

    return estudiantes, universo_familiares, info_personal


def _build_hogares(universo_df: pd.DataFrame, ids_base: set[int]) -> pd.DataFrame:
    u = universo_df[universo_df["IDENTIFICACION"].isin(ids_base)].copy()
    if u.empty:
        return pd.DataFrame(columns=["hogar_id", "fam_ids", "ced_padre", "ced_madre"])

    for col in ["CED_PADRE", "CED_MADRE"]:
        u[col] = pd.to_numeric(u[col], errors="coerce").fillna(0).astype(int)

    rows = []
    for _, r in u.iterrows():
        ids = []
        if r["CED_PADRE"] != 0:
            ids.append(int(r["CED_PADRE"]))
        if r["CED_MADRE"] != 0:
            ids.append(int(r["CED_MADRE"]))
        if not ids:
            continue
        hogar_id = "|".join(sorted([str(i) for i in ids]))
        rows.append(
            {
                "hogar_id": hogar_id,
                "fam_ids": ids,
                "ced_padre": int(r["CED_PADRE"]),
                "ced_madre": int(r["CED_MADRE"]),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["hogar_id", "fam_ids", "ced_padre", "ced_madre"])

    df = pd.DataFrame(rows)
    df = (
        df.groupby("hogar_id", as_index=False)
        .agg(
            fam_ids=("fam_ids", lambda lst: sorted({x for sub in lst for x in sub})),
            ced_padre=("ced_padre", "max"),
            ced_madre=("ced_madre", "max"),
        )
    )
    return df


def _latest_info(info_df: pd.DataFrame) -> pd.DataFrame:
    info = info_df.copy()
    info["IDENTIFICACION"] = pd.to_numeric(info["IDENTIFICACION"], errors="coerce")
    info = info.dropna(subset=["IDENTIFICACION"]).copy()

    info["FECHA_EXP"] = pd.to_datetime(
        info["FECHA EXPEDICION"], errors="coerce", dayfirst=True
    )

    info = info.sort_values(["IDENTIFICACION", "FECHA_EXP"])
    info_latest = info.groupby("IDENTIFICACION", as_index=False).tail(1)

    return info_latest[["IDENTIFICACION", "FECHA_EXP", "ESTADO_CIVIL"]]


def _estado_hogar(
    fam_ids: list[int], info_latest: pd.DataFrame, ced_padre: int = 0, ced_madre: int = 0
) -> str:
    candidatos = info_latest[info_latest["IDENTIFICACION"].isin(fam_ids)].copy()
    if candidatos.empty:
        return "Desconocido"

    def _estado_valido(valor) -> str | None:
        estado = str(valor).strip()
        if not estado or estado.lower() == "nan":
            return None
        return estado

    if candidatos["FECHA_EXP"].isna().all():
        for fam_id in [ced_padre, ced_madre]:
            if not fam_id:
                continue
            candidato = candidatos[candidatos["IDENTIFICACION"] == fam_id]
            if candidato.empty:
                continue
            estado = _estado_valido(candidato.iloc[-1].get("ESTADO_CIVIL", ""))
            if estado:
                return estado
        return "Desconocido"

    candidatos["FECHA_EXP_ORD"] = candidatos["FECHA_EXP"].fillna(pd.Timestamp.min)
    best = candidatos.sort_values("FECHA_EXP_ORD").iloc[-1]

    estado = _estado_valido(best.get("ESTADO_CIVIL", ""))
    if not estado:
        return "Desconocido"
    return estado


with st.spinner("Cargando datos..."):
    excel_filename = get_active_excel_filename()
    estudiantes, universo_familiares, info_personal = load_data(excel_filename)

st.markdown("### Filtros")
estudiantes_filtrados, _filtros_estudiantes = render_student_academic_filters(
    estudiantes, key_prefix="estados_hogares"
)

ids_base = set(
    estudiantes_filtrados["IDENTIFICACION"].dropna().astype(int).unique().tolist()
)

df_hogares = _build_hogares(universo_familiares, ids_base)
if df_hogares.empty:
    st.info("No hay hogares disponibles para el analisis.")
    st.stop()

info_latest = _latest_info(info_personal)

estados = []
for _, row in df_hogares.iterrows():
    estado = _estado_hogar(
        row["fam_ids"], info_latest, row["ced_padre"], row["ced_madre"]
    )
    estados.append(estado)

resultado = pd.DataFrame({"ESTADO_HOGAR": estados})

conteo = resultado["ESTADO_HOGAR"].value_counts().reset_index()
conteo.columns = ["ESTADO_HOGAR", "CANTIDAD"]

st.subheader("Resumen por estado del hogar")

total_hogares = int(conteo["CANTIDAD"].sum())
st.markdown(
    f"""
<div style="
    background: linear-gradient(135deg, #34495e15 0%, #34495e30 100%);
    border-left: 5px solid #34495e;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0 12px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    text-align: center;
">
    <div style="color: #34495e; font-size: 0.95em; margin-bottom: 6px;">
        Total Hogares
    </div>
    <div style="color: #34495e; font-size: 1.8em; font-weight: bold;">
        {total_hogares:,}
    </div>
</div>
""",
    unsafe_allow_html=True,
)

cols = st.columns(4)
for i, (_, r) in enumerate(conteo.iterrows()):
    col = cols[i % 4]
    color = COLORES[i % len(COLORES)]
    with col:
        tarjeta_estado(str(r["ESTADO_HOGAR"]), int(r["CANTIDAD"]), color)

st.markdown("---")

st.subheader("Detalle de hogares por estado")
st.dataframe(conteo, use_container_width=True)
