import numpy as np
import pandas as pd
import streamlit as st

from utils.excel_loader import load_excel_sheet


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

COLORES_QUINTIL = {
    1: "#e74c3c",
    2: "#e67e22",
    3: "#f39c12",
    4: "#2ecc71",
    5: "#27ae60",
    "sin_info": "#95a5a6",
}

CALIFICACIONES_TARJETA = ["Riesgo estable", "Riesgo moderado", "Alto Riesgo"]
RANK_RIESGO = {
    "Riesgo estable": 1,
    "Riesgo moderado": 2,
    "Alto Riesgo": 3,
}


def _parse_valor_deuda(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    has_comma = s.str.contains(",", regex=False)
    s = s.where(
        ~has_comma,
        s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
    )
    return pd.to_numeric(s, errors="coerce").fillna(0)


def _normalizar_colegio(df: pd.DataFrame) -> pd.DataFrame:
    if "Colegio" not in df.columns:
        return df

    colegio = df["Colegio"].copy()
    if colegio.dtype == "O":
        colegio = colegio.fillna("").astype(str).str.strip()
        colegio = colegio.replace("", "Sin dato")
    else:
        colegio = colegio.fillna("Sin dato")

    return df.assign(Colegio=colegio)


@st.cache_data(show_spinner=False)
def load_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    estudiantes = load_excel_sheet("Estudiantes")
    universo_familiares = load_excel_sheet("Universo Familiares")
    empleo = load_excel_sheet("Empleos")
    deudas = load_excel_sheet("Deudas")

    if "Cedula" in estudiantes.columns:
        estudiantes = estudiantes.rename(columns={"Cedula": "IDENTIFICACION"})
    elif "CEDULA" in estudiantes.columns:
        estudiantes = estudiantes.rename(columns={"CEDULA": "IDENTIFICACION"})
    estudiantes = _normalizar_colegio(estudiantes)

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


def agregar_calificacion_desc(df_deudas: pd.DataFrame) -> pd.DataFrame:
    if df_deudas.empty:
        return df_deudas.copy()

    df = df_deudas.copy()
    df["COD_CAL_NORM"] = (
        df["COD_CALIFICACION"]
        .astype(str)
        .str.upper()
        .str.strip()
        .replace({"": np.nan, "NAN": np.nan})
    )
    df = df.dropna(subset=["COD_CAL_NORM"])

    mapa_desc = {
        "A1": "Riesgo estable",
        "A2": "Riesgo moderado",
        "A3": "Riesgo moderado",
        "AL": "Riesgo moderado",
        "B1": "Riesgo moderado",
        "B2": "Riesgo moderado",
        "C1": "Riesgo moderado",
        "C2": "Riesgo moderado",
        "D": "Alto Riesgo",
        "E": "Alto Riesgo",
    }

    df["CALIFICACION_DESC"] = df["COD_CAL_NORM"].map(mapa_desc).fillna("Desconocido")
    return df


def _conteos_por_riesgo_monto(
    df_deudas_calif: pd.DataFrame, key_col: str
) -> dict[tuple[str, str], int]:
    if df_deudas_calif.empty:
        return {}

    df_riesgo = df_deudas_calif.copy()
    df_riesgo["VALOR"] = _parse_valor_deuda(df_riesgo["VALOR"])
    df_riesgo["RANK_RIESGO"] = df_riesgo["CALIFICACION_DESC"].map(RANK_RIESGO)
    df_riesgo = df_riesgo.dropna(subset=["RANK_RIESGO", key_col])
    if df_riesgo.empty:
        return {}

    df_sum = (
        df_riesgo.groupby(
            ["GRUPO_QUINTIL", key_col, "CALIFICACION_DESC"], as_index=False
        )["VALOR"]
        .sum()
        .copy()
    )
    df_sum["RANK_RIESGO"] = df_sum["CALIFICACION_DESC"].map(RANK_RIESGO)
    df_sum = df_sum.sort_values(
        ["GRUPO_QUINTIL", key_col, "VALOR", "RANK_RIESGO"],
        ascending=[True, True, False, False],
    )
    df_persona_riesgo = df_sum.drop_duplicates(["GRUPO_QUINTIL", key_col], keep="first")
    return (
        df_persona_riesgo.groupby(["GRUPO_QUINTIL", "CALIFICACION_DESC"])[key_col]
        .nunique()
        .to_dict()
    )


def mostrar_tarjeta_quintil_deudas(
    titulo: str,
    subtitulo: str,
    color: str,
    hogares_total: int,
    hogares_con_deuda: int,
    conteos_calificacion: dict[str, int],
    label_entidad: str,
) -> None:
    calificacion_cards = ""
    for calificacion in CALIFICACIONES_TARJETA:
        valor = conteos_calificacion.get(calificacion, 0)
        calificacion_cards += (
            f'<div style="background: white; padding: 12px; border-radius: 8px; flex: 1; min-width: 0;">'
            f'<p style="margin: 0; color: #666; font-size: 0.8em;">{calificacion}</p>'
            f'<p style="margin: 5px 0 0 0; color: {color}; font-size: 1.4em; font-weight: bold;">{valor:,}</p>'
            f"</div>"
        )

    html_content = (
        f'<div style="background: linear-gradient(135deg, {color}15 0%, {color}30 100%);'
        f" border-left: 5px solid {color}; border-radius: 10px; padding: 20px;"
        f' margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">'
        f'<h3 style="color: {color}; margin: 0 0 10px 0;">{titulo}</h3>'
        f'<p style="color: #666; font-size: 0.9em; margin: 0 0 15px 0;">{subtitulo}</p>'
        f'<div style="display: flex; gap: 15px;">'
        f'<div style="background: white; padding: 15px; border-radius: 8px; flex: 1;">'
        f'<p style="margin: 0; color: #666; font-size: 0.85em;">{label_entidad}</p>'
        f'<p style="margin: 5px 0 0 0; color: {color}; font-size: 1.8em; font-weight: bold;">{hogares_total:,}</p>'
        f"</div>"
        f'<div style="background: white; padding: 15px; border-radius: 8px; flex: 1;">'
        f'<p style="margin: 0; color: #666; font-size: 0.85em;">{label_entidad} con deuda</p>'
        f'<p style="margin: 5px 0 0 0; color: {color}; font-size: 1.8em; font-weight: bold;">{hogares_con_deuda:,}</p>'
        f"</div>"
        f"</div>"
        f'<div style="display: flex; gap: 15px; margin-top: 15px;">'
        f"{calificacion_cards}"
        f"</div>"
        f"</div>"
    )

    st.markdown(html_content, unsafe_allow_html=True)


st.set_page_config(page_title="Deudas por Quintiles", page_icon="D", layout="wide")
st.title("Deudas por Quintiles")

with st.spinner("Cargando datos..."):
    estudiantes, universo_familiares, empleo, deudas = load_data()

empleo = _filtrar_mes(empleo, ANIO_CORTE, MES_CORTE)
deudas = _filtrar_mes(deudas, ANIO_CORTE, MES_CORTE)

estudiantes_filtrados = estudiantes
if "Colegio" in estudiantes.columns:
    colegios_disponibles = sorted(
        estudiantes["Colegio"].dropna().astype(str).str.strip().unique().tolist()
    )
    colegio_sel = st.selectbox(
        "Colegio",
        options=["Todos los colegios"] + colegios_disponibles,
        index=0,
    )

    if colegio_sel != "Todos los colegios":
        estudiantes_filtrados = estudiantes[estudiantes["Colegio"] == colegio_sel]
else:
    st.warning("La hoja Estudiantes no contiene la columna 'Colegio'.")

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

st.markdown("### Tarjetas por quintil")

ids_base = ids_estudiantes

df_hogares, df_mapa = construir_hogares_familia(
    universo_familiares,
    empleo=empleo,
    ids_base=ids_base,
    cant_papas=cant_papas,
    cant_papas_trab=cant_papas_trab,
    tipo_empleo_sel=tipo_empleo_sel,
)

if df_hogares.empty:
    st.info("No hay hogares disponibles para el grupo seleccionado.")
    st.stop()

if df_mapa.empty:
    st.info("No hay familiares disponibles para el grupo seleccionado.")
    st.stop()

df_deudas_hogar = deudas.merge(
    df_mapa, left_on="IDENTIFICACION", right_on="fam_id", how="inner"
).copy()
df_deudas_hogar = df_deudas_hogar.merge(
    df_hogares[["hogar_id", "GRUPO_QUINTIL"]], on="hogar_id", how="left"
)

entidad_total = df_hogares.groupby("GRUPO_QUINTIL")["hogar_id"].nunique().to_dict()
entidad_con_deuda = (
    df_deudas_hogar.groupby("GRUPO_QUINTIL")["hogar_id"].nunique().to_dict()
    if not df_deudas_hogar.empty
    else {}
)

df_deudas_calif = agregar_calificacion_desc(df_deudas_hogar)
conteos_calificacion = _conteos_por_riesgo_monto(df_deudas_calif, "hogar_id")
label_entidad = "Hogares"


def _conteos_por_grupo(grupo: str) -> dict[str, int]:
    return {
        calif: int(conteos_calificacion.get((grupo, calif), 0))
        for calif in CALIFICACIONES_TARJETA
    }


grupos_orden = [
    (
        "Sin informacion de empleo",
        COLORES_QUINTIL["sin_info"],
        "Sin registro de empleo",
    ),
    ("Quintil 1", COLORES_QUINTIL[1], QUINTILES[1]),
    ("Quintil 2", COLORES_QUINTIL[2], QUINTILES[2]),
    ("Quintil 3", COLORES_QUINTIL[3], QUINTILES[3]),
    ("Quintil 4", COLORES_QUINTIL[4], QUINTILES[4]),
    ("Quintil 5", COLORES_QUINTIL[5], QUINTILES[5]),
]

cards = []
for grupo, color, rango in grupos_orden:
    if isinstance(rango, dict):
        subtitulo = f"Rango salarial: ${rango['min']:,.2f} - ${rango['max']:,.2f}"
    else:
        subtitulo = str(rango)
    cards.append(
        {
            "grupo": grupo,
            "color": color,
            "subtitulo": subtitulo,
            "hogares": int(entidad_total.get(grupo, 0)),
            "hogares_deuda": int(entidad_con_deuda.get(grupo, 0)),
            "calif": _conteos_por_grupo(grupo),
        }
    )

for i in range(0, len(cards), 2):
    col1, col2 = st.columns(2)
    with col1:
        card = cards[i]
        mostrar_tarjeta_quintil_deudas(
            card["grupo"],
            card["subtitulo"],
            card["color"],
            card["hogares"],
            card["hogares_deuda"],
            card["calif"],
            label_entidad,
        )
    if i + 1 < len(cards):
        with col2:
            card = cards[i + 1]
            mostrar_tarjeta_quintil_deudas(
                card["grupo"],
                card["subtitulo"],
                card["color"],
                card["hogares"],
                card["hogares_deuda"],
                card["calif"],
                label_entidad,
            )
