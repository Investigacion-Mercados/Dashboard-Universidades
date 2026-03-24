from __future__ import annotations

import streamlit as st

from utils.excel_loader import (
    get_active_excel_filename,
    list_excel_files,
    set_active_excel_filename,
)

st.set_page_config(page_title="Dashboard Universidades", page_icon="D", layout="wide")

archivos_excel = list_excel_files()
if not archivos_excel:
    st.error("No se encontraron archivos Excel en la carpeta db.")
    st.stop()

archivo_activo = get_active_excel_filename()
indice_actual = (
    archivos_excel.index(archivo_activo) if archivo_activo in archivos_excel else 0
)

with st.sidebar:
    archivo_seleccionado = st.selectbox(
        "Archivo Excel fuente",
        options=archivos_excel,
        index=indice_actual,
        key="dashboard_excel_selector",
    )

archivo_activo = set_active_excel_filename(archivo_seleccionado)
