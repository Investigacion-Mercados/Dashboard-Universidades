import streamlit as st
import pandas as pd
import numpy as np
from utils.excel_loader import load_excel_sheet
from utils.student_columns import normalize_university_column

st.set_page_config(page_title="Perfil Demográfico", page_icon="👥", layout="wide")

# Definir los rangos de quintiles
QUINTILES_ECUADOR = {
    1: {"min": 1.13, "max": 642.03},
    2: {"min": 642.04, "max": 909.07},
    3: {"min": 909.09, "max": 1415.89},
    4: {"min": 1415.92, "max": 2491.60},
    5: {"min": 2491.61, "max": 20009.99},
}

QUINTILES_UDLA = {
    1: {"min": 105.75, "max": 482.00},
    2: {"min": 482.01, "max": 850.92},
    3: {"min": 850.93, "max": 1542.92},
    4: {"min": 1542.93, "max": 2525.00},
    5: {"min": 2525.01, "max": 15392.53},
}

OPCIONES_QUINTILES = {
    "Ecuador (actual)": QUINTILES_ECUADOR,
    "UDLA": QUINTILES_UDLA,
}


def _normalizar_identificacion(df, columnas_posibles):
    for columna in columnas_posibles:
        if columna in df.columns:
            return df.rename(columns={columna: "IDENTIFICACION"})
    return df


def _normalizar_universidad(df):
    return normalize_university_column(df)


def _normalizar_tipo(df):
    if "Tipo" not in df.columns:
        return df

    tipo = df["Tipo"].fillna("").astype(str).str.strip().str.upper()
    tipo = tipo.replace("", "SIN DATO")

    return df.assign(Tipo=tipo)


def _clasificar_tipo_hogar(series):
    valores = (
        pd.Series(series, dtype="object")
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    valores = valores[valores != ""]

    if "DOCUMENTADO" in valores.values:
        return "DOCUMENTADO"
    if "LEAD" in valores.values:
        return "LEAD"
    return "SIN DATO"


def _conteos_tipo_hogar(df):
    if len(df) == 0 or "tipo_hogar" not in df.columns:
        return 0, 0

    conteos = df["tipo_hogar"].value_counts()
    return int(conteos.get("LEAD", 0)), int(conteos.get("DOCUMENTADO", 0))


@st.cache_data
def load_data():
    """Carga todas las hojas necesarias"""
    estudiantes = load_excel_sheet("Estudiantes")
    universo_familiares = load_excel_sheet("Universo Familiares")
    empleo = load_excel_sheet("Empleos")
    info_personal = load_excel_sheet("Informacion Personal")

    estudiantes = _normalizar_identificacion(
        estudiantes, ["IDENTIFICACION", "Cedula", "CEDULA"]
    )
    estudiantes = _normalizar_universidad(estudiantes)
    estudiantes = _normalizar_tipo(estudiantes)

    return estudiantes, universo_familiares, empleo, info_personal


def asignar_quintil(salario, rangos_quintiles):
    """Asigna un quintil según el salario"""
    if pd.isna(salario):
        return None

    if salario < rangos_quintiles[1]["min"]:
        return 1
    if salario > rangos_quintiles[5]["max"]:
        return 5

    for quintil, rango in rangos_quintiles.items():
        if rango["min"] <= salario <= rango["max"]:
            return quintil
    return None


def calcular_metricas_estudiantes_familia(
    estudiantes_df, universo_fam_df, empleo_df, info_personal_df, rangos_quintiles
):
    """Calcula métricas de quintiles para Estudiantes usando análisis por HOGAR de sus familias"""

    # Obtener IDs de estudiantes
    estudiantes_grad = estudiantes_df["IDENTIFICACION"].unique()

    # Obtener información de padres y madres de cada estudiante
    familiares_estudiantes = universo_fam_df[
        universo_fam_df["IDENTIFICACION"].isin(estudiantes_grad)
    ].copy()

    # Reemplazar 0 con NaN para facilitar el procesamiento
    familiares_estudiantes["CED_PADRE"] = familiares_estudiantes["CED_PADRE"].replace(
        0, np.nan
    )
    familiares_estudiantes["CED_MADRE"] = familiares_estudiantes["CED_MADRE"].replace(
        0, np.nan
    )

    # Obtener salarios por familiar tomando su periodo más reciente
    # y sumando todos los registros que tenga en ese periodo.
    empleo_familiares = empleo_df.copy()
    empleo_familiares["SALARIO"] = pd.to_numeric(
        empleo_familiares["SALARIO"], errors="coerce"
    ).fillna(0)
    empleo_familiares["ANIO"] = pd.to_numeric(
        empleo_familiares["ANIO"], errors="coerce"
    )
    empleo_familiares["MES"] = pd.to_numeric(empleo_familiares["MES"], errors="coerce")
    empleo_familiares = empleo_familiares.dropna(subset=["ANIO", "MES"])

    if len(empleo_familiares) > 0:
        empleo_familiares["PERIODO"] = (
            empleo_familiares["ANIO"].astype(int) * 100
            + empleo_familiares["MES"].astype(int)
        )
        max_periodo_por_id = empleo_familiares.groupby("IDENTIFICACION")[
            "PERIODO"
        ].transform("max")
        empleo_ultimo_periodo = empleo_familiares[
            empleo_familiares["PERIODO"] == max_periodo_por_id
        ].copy()
        salario_dict = (
            empleo_ultimo_periodo.groupby("IDENTIFICACION")["SALARIO"].sum().to_dict()
        )
    else:
        salario_dict = {}

    # Crear diccionario de hijos por identificación
    hijos_dict = dict(
        zip(info_personal_df["IDENTIFICACION"], info_personal_df["HIJOS"])
    )
    tipo_estudiante_dict = {}
    if "Tipo" in estudiantes_df.columns:
        tipo_estudiante_dict = (
            estudiantes_df.groupby("IDENTIFICACION")["Tipo"]
            .agg(_clasificar_tipo_hogar)
            .to_dict()
        )

    # Analizar cada hogar
    hogares = []
    hogares_sin_empleo = []

    for _, row in familiares_estudiantes.iterrows():
        estudiante_id = row["IDENTIFICACION"]
        ced_padre = row["CED_PADRE"]
        ced_madre = row["CED_MADRE"]

        # Si no tiene padre ni madre, saltar
        if pd.isna(ced_padre) and pd.isna(ced_madre):
            continue

        # Obtener salarios del hogar
        salario_padre = salario_dict.get(ced_padre, 0) if pd.notna(ced_padre) else 0
        salario_madre = salario_dict.get(ced_madre, 0) if pd.notna(ced_madre) else 0
        salario_hogar = salario_padre + salario_madre

        # Obtener hijos (máximo entre padre y madre)
        hijos_padre = hijos_dict.get(ced_padre, 0) if pd.notna(ced_padre) else 0
        hijos_madre = hijos_dict.get(ced_madre, 0) if pd.notna(ced_madre) else 0
        hijos_hogar = max(hijos_padre, hijos_madre)

        # Identificador único del hogar
        hogar_id = f"{ced_padre}_{ced_madre}"

        if salario_hogar > 0:
            # Hogar con empleo formal
            hogares.append(
                {
                    "hogar_id": hogar_id,
                    "estudiante_id": estudiante_id,
                    "salario_hogar": salario_hogar,
                    "hijos": hijos_hogar,
                    "tipo_hogar": tipo_estudiante_dict.get(estudiante_id, "SIN DATO"),
                }
            )
        else:
            # Hogar sin empleo formal
            hogares_sin_empleo.append(
                {
                    "hogar_id": hogar_id,
                    "estudiante_id": estudiante_id,
                    "hijos": hijos_hogar,
                    "tipo_hogar": tipo_estudiante_dict.get(estudiante_id, "SIN DATO"),
                }
            )

    # Convertir a DataFrame y eliminar duplicados de hogares
    df_hogares = pd.DataFrame(hogares)
    df_sin_empleo = pd.DataFrame(hogares_sin_empleo)

    if len(df_hogares) > 0:
        df_hogares = (
            df_hogares.groupby("hogar_id", as_index=False)
            .agg(
                salario_hogar=("salario_hogar", "max"),
                hijos=("hijos", "max"),
                tipo_hogar=("tipo_hogar", _clasificar_tipo_hogar),
            )
            .copy()
        )
        # Asignar quintil a cada hogar
        df_hogares["QUINTIL_CALCULADO"] = df_hogares["salario_hogar"].apply(
            lambda salario: asignar_quintil(salario, rangos_quintiles)
        )

    if len(df_sin_empleo) > 0:
        df_sin_empleo = (
            df_sin_empleo.groupby("hogar_id", as_index=False)
            .agg(
                hijos=("hijos", "max"),
                tipo_hogar=("tipo_hogar", _clasificar_tipo_hogar),
            )
            .copy()
        )

    hogares_lead_sin_empleo, hogares_documentado_sin_empleo = _conteos_tipo_hogar(
        df_sin_empleo
    )

    # Calcular métricas de hogares sin empleo formal
    no_empleo_formal = {
        "cantidad": len(df_sin_empleo),
        "promedio_hijos": int(
            round(df_sin_empleo["hijos"].mean() if len(df_sin_empleo) > 0 else 0)
        ),
        "hogares_lead": hogares_lead_sin_empleo,
        "hogares_documentado": hogares_documentado_sin_empleo,
    }

    # Calcular métricas por quintil
    metricas = {}
    for quintil in range(1, 6):
        if len(df_hogares) > 0:
            hogares_quintil = df_hogares[df_hogares["QUINTIL_CALCULADO"] == quintil]
            hogares_lead, hogares_documentado = _conteos_tipo_hogar(hogares_quintil)

            if len(hogares_quintil) > 0:
                metricas[quintil] = {
                    "cantidad": len(hogares_quintil),
                    "promedio_hijos": int(round(hogares_quintil["hijos"].mean())),
                    "promedio_salario": hogares_quintil["salario_hogar"].mean(),
                    "mediana_salario": hogares_quintil["salario_hogar"].median(),
                    "hogares_lead": hogares_lead,
                    "hogares_documentado": hogares_documentado,
                }
            else:
                metricas[quintil] = {
                    "cantidad": 0,
                    "promedio_hijos": 0,
                    "promedio_salario": 0,
                    "mediana_salario": 0,
                    "hogares_lead": 0,
                    "hogares_documentado": 0,
                }
        else:
            metricas[quintil] = {
                "cantidad": 0,
                "promedio_hijos": 0,
                "promedio_salario": 0,
                "mediana_salario": 0,
                "hogares_lead": 0,
                "hogares_documentado": 0,
            }

    return metricas, no_empleo_formal


def mostrar_tarjeta_no_empleo(
    no_empleo_data, label="Personas", mostrar_sexo=False, mostrar_desglose_tipo=False
):
    """Muestra una tarjeta para personas/hogares sin empleo formal"""
    cantidad = no_empleo_data["cantidad"]
    promedio_hijos = no_empleo_data["promedio_hijos"]
    porcentaje_hombres = no_empleo_data.get("porcentaje_hombres", 0)
    porcentaje_mujeres = no_empleo_data.get("porcentaje_mujeres", 0)
    hogares_lead = no_empleo_data.get("hogares_lead", 0)
    hogares_documentado = no_empleo_data.get("hogares_documentado", 0)

    color = "#95a5a6"  # Gris

    tipo_block = ""
    if mostrar_desglose_tipo:
        tipo_block = f'<div style="background: white; padding: 15px; border-radius: 8px; margin-top: 15px;"><p style="margin: 0; color: #666; font-size: 0.85em;">🏷️ Tipo de hogar</p><div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 10px;"><div style="background: {color}12; padding: 12px; border-radius: 8px;"><p style="margin: 0; color: #666; font-size: 0.8em;">👥 Hogares Leads</p><p style="margin: 5px 0 0 0; color: {color}; font-size: 1.5em; font-weight: bold;">{hogares_lead:,}</p></div><div style="background: {color}12; padding: 12px; border-radius: 8px;"><p style="margin: 0; color: #666; font-size: 0.8em;">📄 Hogares Documentados</p><p style="margin: 5px 0 0 0; color: {color}; font-size: 1.5em; font-weight: bold;">{hogares_documentado:,}</p></div></div></div>'

    sexo_block = ""
    if mostrar_sexo:
        sexo_block = f'<div style="background: white; padding: 15px; border-radius: 8px; margin-top: 15px;"><p style="margin: 0; color: #666; font-size: 0.85em;">👫 Distribución por sexo</p><div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 10px;"><div><p style="margin: 0; color: #666; font-size: 0.85em;">♂ Hombres</p><p style="margin: 5px 0 0 0; color: {color}; font-size: 1.5em; font-weight: bold;">{porcentaje_hombres:.2f}%</p></div><div><p style="margin: 0; color: #666; font-size: 0.85em;">♀ Mujeres</p><p style="margin: 5px 0 0 0; color: {color}; font-size: 1.5em; font-weight: bold;">{porcentaje_mujeres:.2f}%</p></div></div></div>'

    html_content = f'<div style="background: linear-gradient(135deg, {color}15 0%, {color}30 100%); border-left: 5px solid {color}; border-radius: 10px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"><h3 style="color: {color}; margin: 0 0 15px 0;">❌ Sin Empleo Formal</h3><p style="color: #666; font-size: 0.9em; margin: 0 0 15px 0;">Sin registro en empleo formal</p><div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;"><div style="background: white; padding: 15px; border-radius: 8px;"><p style="margin: 0; color: #666; font-size: 0.85em;">👥 {label}</p><p style="margin: 5px 0 0 0; color: {color}; font-size: 1.8em; font-weight: bold;">{cantidad:,}</p></div><div style="background: white; padding: 15px; border-radius: 8px;"><p style="margin: 0; color: #666; font-size: 0.85em;">👶 Promedio hijos</p><p style="margin: 5px 0 0 0; color: {color}; font-size: 1.8em; font-weight: bold;">{promedio_hijos}</p></div></div>{tipo_block}{sexo_block}</div>'

    st.markdown(html_content, unsafe_allow_html=True)


def mostrar_tarjeta_quintil(
    quintil,
    metricas,
    tipo,
    rangos_quintiles,
    label="Personas",
    mostrar_sexo=False,
    mostrar_desglose_tipo=False,
):
    """Muestra una tarjeta con el diseño para un quintil"""
    rango = rangos_quintiles[quintil]
    cantidad = metricas["cantidad"]
    promedio_hijos = metricas["promedio_hijos"]
    promedio_salario = metricas["promedio_salario"]
    mediana_salario = metricas.get("mediana_salario", 0)
    porcentaje_hombres = metricas.get("porcentaje_hombres", 0)
    porcentaje_mujeres = metricas.get("porcentaje_mujeres", 0)
    hogares_lead = metricas.get("hogares_lead", 0)
    hogares_documentado = metricas.get("hogares_documentado", 0)

    # Colores según quintil
    colores = {
        1: "#e74c3c",  # Rojo
        2: "#e67e22",  # Naranja
        3: "#f39c12",  # Amarillo/Dorado
        4: "#2ecc71",  # Verde
        5: "#27ae60",  # Verde oscuro
    }

    color = colores.get(quintil, "#3498db")

    tipo_block = ""
    if mostrar_desglose_tipo:
        tipo_block = f'<div style="background: white; padding: 15px; border-radius: 8px; margin-top: 15px;"><p style="margin: 0; color: #666; font-size: 0.85em;">🏷️ Tipo de hogar</p><div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 10px;"><div style="background: {color}12; padding: 12px; border-radius: 8px;"><p style="margin: 0; color: #666; font-size: 0.8em;">👥 Hogares Leads</p><p style="margin: 5px 0 0 0; color: {color}; font-size: 1.5em; font-weight: bold;">{hogares_lead:,}</p></div><div style="background: {color}12; padding: 12px; border-radius: 8px;"><p style="margin: 0; color: #666; font-size: 0.8em;">📄 Hogares Documentados</p><p style="margin: 5px 0 0 0; color: {color}; font-size: 1.5em; font-weight: bold;">{hogares_documentado:,}</p></div></div></div>'

    sexo_block = ""
    if mostrar_sexo:
        sexo_block = f'<div style="background: white; padding: 15px; border-radius: 8px; margin-top: 15px;"><p style="margin: 0; color: #666; font-size: 0.85em;">👫 Distribución por sexo</p><div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 10px;"><div><p style="margin: 0; color: #666; font-size: 0.85em;">♂ Hombres</p><p style="margin: 5px 0 0 0; color: {color}; font-size: 1.5em; font-weight: bold;">{porcentaje_hombres:.2f}%</p></div><div><p style="margin: 0; color: #666; font-size: 0.85em;">♀ Mujeres</p><p style="margin: 5px 0 0 0; color: {color}; font-size: 1.5em; font-weight: bold;">{porcentaje_mujeres:.2f}%</p></div></div></div>'

    html_content = f'<div style="background: linear-gradient(135deg, {color}15 0%, {color}30 100%); border-left: 5px solid {color}; border-radius: 10px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"><h3 style="color: {color}; margin: 0 0 15px 0;">🏆 Quintil {quintil}</h3><p style="color: #666; font-size: 0.9em; margin: 0 0 15px 0;">Rango salarial: ${rango["min"]:,.2f} - ${rango["max"]:,.2f}</p><div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;"><div style="background: white; padding: 15px; border-radius: 8px;"><p style="margin: 0; color: #666; font-size: 0.85em;">👥 {label}</p><p style="margin: 5px 0 0 0; color: {color}; font-size: 1.8em; font-weight: bold;">{cantidad:,}</p></div><div style="background: white; padding: 15px; border-radius: 8px;"><p style="margin: 0; color: #666; font-size: 0.85em;">👶 Promedio hijos</p><p style="margin: 5px 0 0 0; color: {color}; font-size: 1.8em; font-weight: bold;">{promedio_hijos}</p></div></div><div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;"><div style="background: white; padding: 15px; border-radius: 8px;"><p style="margin: 0; color: #666; font-size: 0.85em;">💰 Salario promedio</p><p style="margin: 5px 0 0 0; color: {color}; font-size: 1.8em; font-weight: bold;">${promedio_salario:,.2f}</p></div><div style="background: white; padding: 15px; border-radius: 8px;"><p style="margin: 0; color: #666; font-size: 0.85em;">📊 Mediana salario</p><p style="margin: 5px 0 0 0; color: {color}; font-size: 1.8em; font-weight: bold;">${mediana_salario:,.2f}</p></div></div>{tipo_block}{sexo_block}</div>'

    st.markdown(html_content, unsafe_allow_html=True)


# Título principal
st.title("👥 Perfil Demográfico por Quintiles")

# Cargar datos
with st.spinner("Cargando datos..."):
    estudiantes_df, universo_fam_df, empleo_df, info_personal_df = load_data()

col_filtro_1, col_filtro_2 = st.columns(2)
universidad_sel = "Todas las universidades"

with col_filtro_1:
    estudiantes_filtrados = estudiantes_df
    if "Universidad" in estudiantes_df.columns:
        universidades_disponibles = sorted(
            estudiantes_df["Universidad"].dropna().astype(str).str.strip().unique().tolist()
        )
        universidad_sel = st.selectbox(
            "Universidad",
            options=["Todas las universidades"] + universidades_disponibles,
            index=0,
        )

        if universidad_sel != "Todas las universidades":
            estudiantes_filtrados = estudiantes_df[
                estudiantes_df["Universidad"] == universidad_sel
            ]
    else:
        st.warning("La hoja Estudiantes no contiene la columna 'Universidad'.")

with col_filtro_2:
    quintil_sel = st.selectbox(
        "Quintiles a usar",
        options=list(OPCIONES_QUINTILES.keys()),
        index=0,
    )

rangos_quintiles = OPCIONES_QUINTILES[quintil_sel]
st.caption(f"Rangos activos: {quintil_sel}")

st.header("📊 Análisis por Quintiles - Estudiantes (Familia)")
st.info(
    "📌 Análisis basado en los salarios de los familiares (padres/madres) de los estudiantes"
)

with st.spinner("Calculando métricas..."):
    metricas, no_empleo_formal = calcular_metricas_estudiantes_familia(
        estudiantes_filtrados,
        universo_fam_df,
        empleo_df,
        info_personal_df,
        rangos_quintiles,
    )

tiene_solo_innova = (
    "Universidad" in estudiantes_filtrados.columns
    and not estudiantes_filtrados.empty
    and estudiantes_filtrados["Universidad"].astype(str).str.strip().eq("INNOVA").all()
)
mostrar_desglose_tipo = "Tipo" in estudiantes_filtrados.columns and not tiene_solo_innova

# Tarjeta de población total
total_poblacion = estudiantes_filtrados["IDENTIFICACION"].nunique()

# Calcular total de hogares
total_hogares = (
    sum(metricas[q]["cantidad"] for q in range(1, 6)) + no_empleo_formal["cantidad"]
)

# Mostrar tarjetas en 2 columnas
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
    <div style="
        background: linear-gradient(135deg, #3498db15 0%, #3498db30 100%);
        border-left: 5px solid #3498db;
        border-radius: 8px;
        padding: 12px 20px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    ">
        <h3 style="color: #3498db; margin: 0 0 5px 0; font-size: 1.2em;">👥 Total Estudiantes</h3>
        <p style="margin: 0; color: #3498db; font-size: 2em; font-weight: bold;">
            {total_poblacion:,}
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
        padding: 12px 20px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    ">
        <h3 style="color: #3498db; margin: 0 0 5px 0; font-size: 1.2em;">🏠 Total Hogares</h3>
        <p style="margin: 0; color: #3498db; font-size: 2em; font-weight: bold;">
            {total_hogares:,}
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("### 💰 Distribución por Quintiles")

# Primera fila: Sin empleo formal y Quintil 1
col1, col2 = st.columns(2)
with col1:
    mostrar_tarjeta_no_empleo(
        no_empleo_formal,
        label="Hogares",
        mostrar_desglose_tipo=mostrar_desglose_tipo,
    )
with col2:
    mostrar_tarjeta_quintil(
        1,
        metricas[1],
        "Estudiantes",
        rangos_quintiles,
        label="Hogares",
        mostrar_desglose_tipo=mostrar_desglose_tipo,
    )

# Resto de quintiles
for i in range(2, 6, 2):
    col1, col2 = st.columns(2)
    with col1:
        mostrar_tarjeta_quintil(
            i,
            metricas[i],
            "Estudiantes",
            rangos_quintiles,
            label="Hogares",
            mostrar_desglose_tipo=mostrar_desglose_tipo,
        )
    if i + 1 <= 5:
        with col2:
            mostrar_tarjeta_quintil(
                i + 1,
                metricas[i + 1],
                "Estudiantes",
                rangos_quintiles,
                label="Hogares",
                mostrar_desglose_tipo=mostrar_desglose_tipo,
            )
