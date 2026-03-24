import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import pandas as pd
import os
import sys
from shapely.geometry import Point

# Anadir utils al path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils"))
from excel_loader import get_active_excel_filename, load_excel_sheet

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


@st.cache_data
def cargar_estudiantes(excel_filename: str):
    """Carga los estudiantes y sus ubicaciones"""
    df_estudiantes = load_excel_sheet("Estudiantes", excel_filename)
    df_estudiantes = _normalizar_identificacion(
        df_estudiantes, ["IDENTIFICACION", "Cedula", "CEDULA"]
    )

    df_info = load_excel_sheet("Informacion Personal", excel_filename)

    # Filtrar solo estudiantes con ubicacion
    df_ubicaciones = df_info[
        df_info["IDENTIFICACION"].isin(df_estudiantes["IDENTIFICACION"])
    ].copy()
    df_ubicaciones = df_ubicaciones.dropna(subset=["LATITUD", "LONGITUD"])

    return df_ubicaciones


def obtener_nombre(row):
    """Obtiene el nombre de la parroquia desde multiples campos posibles"""
    for campo in ["nombre", "DPA_DESPAR", "dpa_despar"]:
        valor = row.get(campo, None)
        if pd.notna(valor) and valor != "":
            return valor
    return "Sin nombre"


def calcular_estudiantes_por_parroquia(gdf_todas, df_estudiantes):
    """Calcula cuantos estudiantes hay en cada parroquia"""
    estudiantes_por_parroquia = {}

    for idx, parr_row in gdf_todas.iterrows():
        count = 0
        for _, est_row in df_estudiantes.iterrows():
            punto = Point(est_row["LONGITUD"], est_row["LATITUD"])
            if parr_row.geometry.contains(punto):
                count += 1
        estudiantes_por_parroquia[idx] = count

    return estudiantes_por_parroquia


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


def crear_mapa(gdf_todas, df_estudiantes):
    """Crea el mapa con parroquias y mapa de calor de estudiantes"""
    if not df_estudiantes.empty:
        centro_lat = df_estudiantes["LATITUD"].mean()
        centro_lon = df_estudiantes["LONGITUD"].mean()
        centro = [centro_lat, centro_lon]
    else:
        centro = [-0.20, -78.50]

    m = folium.Map(location=centro, zoom_start=11, tiles="cartodbpositron")

    estudiantes_por_parroquia = calcular_estudiantes_por_parroquia(
        gdf_todas, df_estudiantes
    )
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
    df_estudiantes = cargar_estudiantes(excel_filename)

    mapa = crear_mapa(gdf_todas, df_estudiantes)

    # Informacion
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Parroquias", len(gdf_todas))
    with col2:
        st.metric("Estudiantes", len(df_estudiantes))

    st_folium(mapa, width=1400, height=800, returned_objects=[])

except Exception as e:
    st.error(f"Error al cargar el mapa: {str(e)}")
    st.exception(e)
