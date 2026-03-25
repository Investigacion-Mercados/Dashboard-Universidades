import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import pandas as pd
import os
from shapely.geometry import Point

from utils.excel_loader import get_active_excel_filename, load_excel_sheet
from utils.student_columns import find_column, normalize_university_column
from utils.student_filters import render_student_academic_filters

# Configuracion de la pagina
st.set_page_config(
    page_title="Mapa de Calor - Estudiantes",
    page_icon="\U0001F5FA\uFE0F",
    layout="wide",
)

st.title("\U0001F5FA\uFE0F Mapa de Parroquias")

# Rutas a los archivos GeoJSON
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
GJSON_RURAL = os.path.join(BASE_DIR, "db", "parroquiasRurales.geojson")
GJSON_URBANA = os.path.join(BASE_DIR, "db", "parroquiasUrbanas.geojson")
GJSON_OTRAS = os.path.join(BASE_DIR, "db", "otras.geojson")


@st.cache_data
def cargar_geojson():
    """Carga y combina todos los GeoJSON de parroquias"""
    gdf_rurales = gpd.read_file(GJSON_RURAL)
    gdf_urbanas = gpd.read_file(GJSON_URBANA)
    gdf_otras = gpd.read_file(GJSON_OTRAS)

    # Asegurar que todas esten en EPSG:4326 (WGS84)
    if gdf_rurales.crs != "EPSG:4326":
        gdf_rurales = gdf_rurales.to_crs("EPSG:4326")
    if gdf_urbanas.crs != "EPSG:4326":
        gdf_urbanas = gdf_urbanas.to_crs("EPSG:4326")
    if gdf_otras.crs != "EPSG:4326":
        gdf_otras = gdf_otras.to_crs("EPSG:4326")

    return pd.concat([gdf_rurales, gdf_urbanas, gdf_otras], ignore_index=True)


def _normalizar_identificacion(df, columnas_posibles):
    for columna in columnas_posibles:
        if columna in df.columns:
            return df.rename(columns={columna: "IDENTIFICACION"})
    return df


def _normalizar_universidad(df):
    return normalize_university_column(df)


@st.cache_data
def cargar_datos(excel_filename: str):
    """Carga estudiantes e informacion personal del archivo activo."""
    df_estudiantes = load_excel_sheet("Estudiantes", excel_filename)
    df_estudiantes = _normalizar_identificacion(
        df_estudiantes, ["IDENTIFICACION", "Cedula", "CEDULA"]
    )
    df_estudiantes = _normalizar_universidad(df_estudiantes)

    df_info = load_excel_sheet("Informacion Personal", excel_filename)
    df_info = _normalizar_identificacion(df_info, ["IDENTIFICACION", "Cedula", "CEDULA"])

    return df_estudiantes, df_info


def filtrar_ubicaciones(df_info, df_estudiantes):
    """Filtra ubicaciones solo para los estudiantes visibles en los filtros."""
    if "IDENTIFICACION" not in df_info.columns or "IDENTIFICACION" not in df_estudiantes.columns:
        return pd.DataFrame(columns=["IDENTIFICACION", "LATITUD", "LONGITUD"])

    df_ubicaciones = df_info[
        df_info["IDENTIFICACION"].isin(df_estudiantes["IDENTIFICACION"])
    ].copy()
    if "LATITUD" not in df_ubicaciones.columns or "LONGITUD" not in df_ubicaciones.columns:
        return pd.DataFrame(columns=["IDENTIFICACION", "LATITUD", "LONGITUD"])

    df_ubicaciones["LATITUD"] = pd.to_numeric(df_ubicaciones["LATITUD"], errors="coerce")
    df_ubicaciones["LONGITUD"] = pd.to_numeric(df_ubicaciones["LONGITUD"], errors="coerce")
    df_ubicaciones = df_ubicaciones.dropna(subset=["LATITUD", "LONGITUD"])

    return df_ubicaciones


def obtener_nombre(row):
    """Obtiene el nombre de la parroquia desde multiples campos posibles"""
    for campo in ["nombre", "DPA_DESPAR", "dpa_despar"]:
        valor = row.get(campo, None)
        if pd.notna(valor) and valor != "":
            return valor
    return "Sin nombre"


def enriquecer_estudiantes_con_parroquia(gdf_todas, df_estudiantes):
    """Asigna parroquia a cada estudiante segun sus coordenadas."""
    detalle = df_estudiantes.copy()
    detalle["PARROQUIA"] = "Sin parroquia"
    detalle["PARROQUIA_IDX"] = pd.Series(pd.NA, index=detalle.index, dtype="Int64")

    if detalle.empty:
        return detalle

    parroquias = [
        (idx, obtener_nombre(row), row.geometry) for idx, row in gdf_todas.iterrows()
    ]

    for idx_estudiante, est_row in detalle.iterrows():
        punto = Point(est_row["LONGITUD"], est_row["LATITUD"])
        for idx_parroquia, nombre_parroquia, geometry in parroquias:
            if geometry.contains(punto):
                detalle.at[idx_estudiante, "PARROQUIA"] = nombre_parroquia
                detalle.at[idx_estudiante, "PARROQUIA_IDX"] = idx_parroquia
                break

    return detalle


def calcular_estudiantes_por_parroquia(df_estudiantes):
    """Calcula cuantos estudiantes hay en cada parroquia ya asignada."""
    if df_estudiantes.empty or "PARROQUIA_IDX" not in df_estudiantes.columns:
        return {}

    conteos = (
        df_estudiantes.dropna(subset=["PARROQUIA_IDX"])
        .groupby("PARROQUIA_IDX")
        .size()
        .to_dict()
    )
    return {int(idx): int(cantidad) for idx, cantidad in conteos.items()}


def get_color_estudiantes(cantidad, max_cantidad):
    """Obtiene el color segun la cantidad de estudiantes"""
    if cantidad == 0:
        return "white"

    percentil = cantidad / max_cantidad if max_cantidad > 0 else 0

    if percentil < 0.2:
        return "#E6F2FF"
    if percentil < 0.4:
        return "#99CCFF"
    if percentil < 0.6:
        return "#4DA6FF"
    if percentil < 0.8:
        return "#0073E6"
    return "#004C99"


def preparar_tabla_ubicaciones(df_estudiantes):
    """Prepara la tabla de detalle de estudiantes con ubicacion."""
    nombre_col = find_column(df_estudiantes, ["NOMBRE", "NOMBRES", "Nombre"])
    ubicacion_col = find_column(df_estudiantes, ["UBICACION", "Ubicacion"])

    if nombre_col is None:
        estudiantes = df_estudiantes["IDENTIFICACION"].astype(str)
    else:
        estudiantes = df_estudiantes[nombre_col].fillna("").astype(str).str.strip()
        estudiantes = estudiantes.where(estudiantes != "", df_estudiantes["IDENTIFICACION"].astype(str))

    if ubicacion_col is None:
        ubicaciones = pd.Series([""] * len(df_estudiantes), index=df_estudiantes.index)
    else:
        ubicaciones = df_estudiantes[ubicacion_col].fillna("").astype(str).str.strip()

    tabla = pd.DataFrame(
        {
            "Identificacion": df_estudiantes["IDENTIFICACION"].astype(str),
            "Estudiante": estudiantes,
            "Parroquia": df_estudiantes["PARROQUIA"].fillna("Sin parroquia"),
            "Ubicacion": ubicaciones,
            "Latitud": df_estudiantes["LATITUD"].round(6),
            "Longitud": df_estudiantes["LONGITUD"].round(6),
        }
    )

    return tabla.sort_values(["Parroquia", "Estudiante"]).reset_index(drop=True)


def crear_mapa(gdf_todas, df_estudiantes):
    """Crea el mapa con parroquias y mapa de calor de estudiantes"""
    if not df_estudiantes.empty:
        centro_lat = df_estudiantes["LATITUD"].mean()
        centro_lon = df_estudiantes["LONGITUD"].mean()
        centro = [centro_lat, centro_lon]
    else:
        centro = [-0.20, -78.50]

    m = folium.Map(location=centro, zoom_start=11, tiles="cartodbpositron")

    estudiantes_por_parroquia = calcular_estudiantes_por_parroquia(df_estudiantes)
    max_estudiantes = (
        max(estudiantes_por_parroquia.values()) if estudiantes_por_parroquia else 1
    )

    # Capa de parroquias
    fg_parroquias = folium.FeatureGroup(name="Parroquias", show=True).add_to(m)
    for _, row in gdf_todas.iterrows():
        nombre_original = obtener_nombre(row)
        folium.GeoJson(
            {
                "type": "Feature",
                "geometry": row.geometry.__geo_interface__,
                "properties": {"nombre": nombre_original},
            },
            style_function=lambda _: {
                "fillColor": "white",
                "color": "black",
                "weight": 0.25,
                "fillOpacity": 0.3,
            },
            tooltip=folium.GeoJsonTooltip(fields=["nombre"], aliases=["Parroquia:"]),
        ).add_to(fg_parroquias)

    # Capa de mapa de calor de estudiantes (parroquias coloreadas)
    fg_estudiantes = folium.FeatureGroup(name="Mapa Calor - Estudiantes", show=True)
    fg_estudiantes.add_to(m)

    for idx, row in gdf_todas.iterrows():
        nombre_original = obtener_nombre(row)
        cantidad = estudiantes_por_parroquia.get(idx, 0)
        color = get_color_estudiantes(cantidad, max_estudiantes)

        folium.GeoJson(
            {
                "type": "Feature",
                "geometry": row.geometry.__geo_interface__,
                "properties": {"nombre": nombre_original, "estudiantes": cantidad},
            },
            style_function=lambda _, c=color: {
                "fillColor": c,
                "color": "black",
                "weight": 0.25,
                "fillOpacity": 0.7,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["nombre", "estudiantes"], aliases=["Parroquia:", "Estudiantes:"]
            ),
        ).add_to(fg_estudiantes)

    folium.LayerControl(position="topleft", collapsed=False).add_to(m)

    return m


# Cargar datos
try:
    gdf_todas = cargar_geojson()
    excel_filename = get_active_excel_filename()
    df_estudiantes, df_info = cargar_datos(excel_filename)

    st.markdown("### Filtros")
    estudiantes_filtrados, _filtros_estudiantes = render_student_academic_filters(
        df_estudiantes,
        key_prefix="mapa_calor",
        lock_single_option_keys={"universidad"},
    )
    df_ubicaciones = filtrar_ubicaciones(df_info, estudiantes_filtrados)
    df_ubicaciones = enriquecer_estudiantes_con_parroquia(gdf_todas, df_ubicaciones)

    mapa = crear_mapa(gdf_todas, df_ubicaciones)

    # Informacion
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Parroquias", len(gdf_todas))
    with col2:
        st.metric("Estudiantes", len(df_ubicaciones))

    st_folium(mapa, width=1400, height=800, returned_objects=[])

    st.markdown("### Detalle de ubicaciones")
    tabla_ubicaciones = preparar_tabla_ubicaciones(df_ubicaciones)
    if tabla_ubicaciones.empty:
        st.info("No hay estudiantes con ubicacion para mostrar en la tabla.")
    else:
        st.dataframe(tabla_ubicaciones, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"Error al cargar el mapa: {str(e)}")
    st.exception(e)
