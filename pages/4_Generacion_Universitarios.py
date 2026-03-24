import streamlit as st
import pandas as pd
import os
import sys

# Añadir utils al path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils"))
from excel_loader import get_active_excel_filename, load_excel_sheet
from student_filters import apply_student_academic_filters, render_student_academic_filters
from student_columns import normalize_university_column

# Configuración de la página
st.set_page_config(
    page_title="Generación Universitarios", page_icon="🎓", layout="wide"
)

st.title("🎓 Primera Generación de Universitarios")

def _normalizar_identificacion(df, columnas_posibles):
    for columna in columnas_posibles:
        if columna in df.columns:
            return df.rename(columns={columna: "IDENTIFICACION"})
    return df


def _normalizar_universidad(df):
    return normalize_university_column(df)


@st.cache_data
def load_estudiantes(excel_filename: str):
    df_estudiantes = load_excel_sheet("Estudiantes", excel_filename)
    df_estudiantes = _normalizar_identificacion(
        df_estudiantes, ["IDENTIFICACION", "Cedula", "CEDULA"]
    )
    return _normalizar_universidad(df_estudiantes)


@st.cache_data
def calcular_primera_generacion(
    excel_filename: str, filtros_estudiantes: dict[str, str | None]
):
    """
    Calcula el porcentaje de estudiantes que son primera generación de universitarios.

    Un estudiante es considerado "Primera Generación" si NINGUNO de sus padres tiene
    NIVEL_ESTUDIO = "SUPERIOR".
    """
    # Cargar datos
    df_estudiantes = apply_student_academic_filters(
        load_estudiantes(excel_filename), filtros_estudiantes
    )
    df_familiares = load_excel_sheet("Universo Familiares", excel_filename)
    df_info = load_excel_sheet("Informacion Personal", excel_filename)

    # Obtener IDs únicos de estudiantes
    ids_estudiantes = df_estudiantes["IDENTIFICACION"].dropna().unique()

    primera_generacion = 0
    no_primera_generacion = 0

    for id_estudiante in ids_estudiantes:
        # Buscar familiares del estudiante
        fila_familiar = df_familiares[df_familiares["IDENTIFICACION"] == id_estudiante]

        if fila_familiar.empty:
            continue

        # Obtener cedulas de padres
        ced_padre = fila_familiar.iloc[0].get("CED_PADRE")
        ced_madre = fila_familiar.iloc[0].get("CED_MADRE")

        # Filtrar cedulas válidas (no nulas y no 0)
        cedulas_padres = []
        if pd.notna(ced_padre) and ced_padre != 0:
            cedulas_padres.append(ced_padre)
        if pd.notna(ced_madre) and ced_madre != 0:
            cedulas_padres.append(ced_madre)

        # Buscar nivel de estudio de los padres
        es_primera_generacion = True
        for ced in cedulas_padres:
            fila_padre = df_info[df_info["IDENTIFICACION"] == ced]
            if not fila_padre.empty:
                nivel_estudio = fila_padre.iloc[0].get("NIVEL_ESTUDIO")
                # Si algún padre tiene NIVEL_ESTUDIO = "SUPERIOR", NO es primera generación
                if pd.notna(nivel_estudio) and str(nivel_estudio).upper() == "SUPERIOR":
                    es_primera_generacion = False
                    break

        if es_primera_generacion:
            primera_generacion += 1
        else:
            no_primera_generacion += 1

    # Calcular totales y porcentajes
    total = primera_generacion + no_primera_generacion
    if total > 0:
        pct_primera = (primera_generacion / total) * 100
        pct_no_primera = (no_primera_generacion / total) * 100
    else:
        pct_primera = 0
        pct_no_primera = 0

    return primera_generacion, no_primera_generacion, pct_primera, pct_no_primera, total


try:
    excel_filename = get_active_excel_filename()
    df_estudiantes = load_estudiantes(excel_filename)
    st.markdown("### Filtros")
    _, filtros_estudiantes = render_student_academic_filters(
        df_estudiantes, key_prefix="primera_generacion"
    )

    primera_gen, no_primera_gen, pct_primera, pct_no_primera, total = (
        calcular_primera_generacion(excel_filename, filtros_estudiantes)
    )

    # Mostrar en dos columnas con tarjetas
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
        <div style="
            background: linear-gradient(135deg, #2ecc7115 0%, #2ecc7130 100%);
            border-left: 5px solid #2ecc71;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        ">
            <h3 style="color: #2ecc71; margin: 0 0 10px 0; font-size: 1.3em;">🌟 Primera Generación</h3>
            <p style="margin: 0 0 15px 0; color: #666; font-size: 0.95em;">De universitarios</p>
            <p style="margin: 0 0 10px 0; color: #2ecc71; font-size: 3em; font-weight: bold;">
                {pct_primera:.1f}%
            </p>
            <p style="margin: 0; color: #666; font-size: 0.9em;">
                {primera_gen} de {total} estudiantes
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div style="
            background: linear-gradient(135deg, #3498db15 0%, #3498db30 100%);
            border-left: 5px solid #3498db;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        ">
            <h3 style="color: #3498db; margin: 0 0 10px 0; font-size: 1.3em;">👨‍👩‍👧 No Primera Generación</h3>
            <p style="margin: 0 0 15px 0; color: #666; font-size: 0.95em;">De universitarios</p>
            <p style="margin: 0 0 10px 0; color: #3498db; font-size: 3em; font-weight: bold;">
                {pct_no_primera:.1f}%
            </p>
            <p style="margin: 0; color: #666; font-size: 0.9em;">
                {no_primera_gen} de {total} estudiantes
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

except Exception as e:
    st.error(f"❌ Error al calcular generación de universitarios: {str(e)}")
    st.exception(e)
