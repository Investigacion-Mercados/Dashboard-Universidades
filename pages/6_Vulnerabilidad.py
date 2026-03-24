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
COLORES = {
    "azul": "#3498db",
    "rojo": "#e74c3c",
    "naranja": "#f39c12",
    "verde": "#2ecc71",
}

QUINTILES_ECUADOR = {
    1: {"min": 1.13, "max": 642.03},
    2: {"min": 642.04, "max": 909.07},
    3: {"min": 909.09, "max": 1415.89},
    4: {"min": 1415.92, "max": 2491.60},
    5: {"min": 2491.61, "max": 20009.99},
}

QUINTILES_INNOVA = {
    1: {"min": 470.00, "max": 482.00},
    2: {"min": 482.01, "max": 707.07},
    3: {"min": 707.08, "max": 1104.36},
    4: {"min": 1104.37, "max": 1856.00},
    5: {"min": 1856.01, "max": 5011.00},
}

QUINTILES_UDLA = {
    1: {"min": 105.75, "max": 482.00},
    2: {"min": 482.01, "max": 850.92},
    3: {"min": 850.93, "max": 1542.92},
    4: {"min": 1542.93, "max": 2525.00},
    5: {"min": 2525.01, "max": 15392.53},
}

GRUPO_QUINTIL_OPTS = ["Todos", "1", "2", "3", "4", "5", "sin info"]
RANGO_QUINTIL_OPTS = ["Ecuador", "Innova", "UDLA"]


def tarjeta_simple(titulo: str, valor: str, color: str) -> None:
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
                {valor}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def _filtrar_mes(df: pd.DataFrame, anio: int, mes: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df[(df["ANIO"] == anio) & (df["MES"] == mes)].copy()


def _normalizar_ids_familia(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["CED_PADRE", "CED_MADRE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def _empleo_reciente(empleo: pd.DataFrame) -> pd.DataFrame:
    if empleo.empty:
        return empleo.copy()
    emp = empleo.sort_values(
        ["IDENTIFICACION", "ANIO", "MES"], ascending=[True, False, False]
    ).drop_duplicates("IDENTIFICACION", keep="first")
    emp["TIPO_EMPRESA"] = emp["TIPO_EMPRESA"].astype(str).str.strip()
    return emp


def _asignar_quintil_custom(salario: float, rangos: dict[int, dict[str, float]]) -> str:
    if pd.isna(salario) or float(salario) <= 0:
        return "sin info"
    val = float(salario)
    q_min = rangos[1]["min"]
    q_max = rangos[5]["max"]
    if val < q_min:
        return "1"
    if val > q_max:
        return "5"
    for q in [1, 2, 3, 4, 5]:
        r = rangos[q]
        if r["min"] <= val <= r["max"]:
            return str(q)
    return "sin info"


def _rangos_por_nombre(nombre: str) -> dict[int, dict[str, float]]:
    if nombre == "Innova":
        return QUINTILES_INNOVA
    if nombre == "UDLA":
        return QUINTILES_UDLA
    return QUINTILES_ECUADOR


def _quintil_por_estudiante(
    universo_familiares: pd.DataFrame,
    empleo: pd.DataFrame,
    rangos: dict[int, dict[str, float]],
) -> pd.DataFrame:
    if universo_familiares.empty:
        return pd.DataFrame(columns=["IDENTIFICACION", "grupo_quintil"])

    u = _normalizar_ids_familia(universo_familiares.copy())
    u["hogar_id"] = u.apply(
        lambda r: "|".join(sorted([str(r["CED_PADRE"]), str(r["CED_MADRE"])])), axis=1
    )
    u = u[u["hogar_id"] != "0|0"].copy()
    if u.empty:
        return pd.DataFrame(columns=["IDENTIFICACION", "grupo_quintil"])

    pares = []
    for _, r in u.iterrows():
        if r["CED_PADRE"] != 0:
            pares.append((r["hogar_id"], int(r["CED_PADRE"])))
        if r["CED_MADRE"] != 0:
            pares.append((r["hogar_id"], int(r["CED_MADRE"])))
    if not pares:
        return pd.DataFrame(columns=["IDENTIFICACION", "grupo_quintil"])

    df_mapa = pd.DataFrame(pares, columns=["hogar_id", "fam_id"]).drop_duplicates()

    if empleo.empty:
        salario_map: dict[int, float] = {}
    else:
        emp = empleo.copy()
        emp["IDENTIFICACION"] = pd.to_numeric(emp["IDENTIFICACION"], errors="coerce")
        emp = emp.dropna(subset=["IDENTIFICACION"]).copy()
        emp["IDENTIFICACION"] = emp["IDENTIFICACION"].astype(int)
        emp["SALARIO"] = pd.to_numeric(emp["SALARIO"], errors="coerce").fillna(0)
        salario_map = emp.groupby("IDENTIFICACION")["SALARIO"].sum().to_dict()

    df_mapa["salario"] = df_mapa["fam_id"].map(salario_map).fillna(0.0)
    df_hogares = df_mapa.groupby("hogar_id", as_index=False)["salario"].sum()
    df_hogares["grupo_quintil"] = df_hogares["salario"].apply(
        lambda x: _asignar_quintil_custom(x, rangos)
    )

    out = (
        u[["IDENTIFICACION", "hogar_id"]]
        .drop_duplicates()
        .merge(df_hogares[["hogar_id", "grupo_quintil"]], on="hogar_id", how="left")
    )
    out["grupo_quintil"] = out["grupo_quintil"].fillna("sin info")
    return out[["IDENTIFICACION", "grupo_quintil"]].drop_duplicates()


def _filtrar_universo_por_filtros(
    universo_familiares: pd.DataFrame,
    empleo: pd.DataFrame,
    ids_base: set[int],
    cant_papas: int | None,
    cant_papas_trab: int | None,
    tipo_empleo_sel: str,
) -> pd.DataFrame:
    u = universo_familiares[universo_familiares["IDENTIFICACION"].isin(ids_base)].copy()
    if u.empty:
        return u

    sin_filtros = (
        cant_papas is None and cant_papas_trab is None and tipo_empleo_sel == "Todos"
    )
    if sin_filtros:
        return u

    u = _normalizar_ids_familia(u)
    u["n_papas"] = (u["CED_PADRE"] != 0).astype(int) + (u["CED_MADRE"] != 0).astype(int)

    if cant_papas in (1, 2):
        u = u[u["n_papas"] == cant_papas]
        if u.empty:
            return u

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
        return u.iloc[0:0]

    df_mapa = pd.DataFrame(pares, columns=["hogar_id", "fam_id"]).drop_duplicates()

    emp = _empleo_reciente(empleo)
    if emp.empty:
        df_emp = df_mapa.copy()
        df_emp["tipo_empleo"] = "Desconocido"
        df_emp["trabaja"] = False
    else:
        df_emp = df_mapa.merge(
            emp[["IDENTIFICACION", "TIPO_EMPRESA"]],
            left_on="fam_id",
            right_on="IDENTIFICACION",
            how="left",
            indicator=True,
        )
        tipo_emp_norm = df_emp["TIPO_EMPRESA"].astype(str).str.strip()
        df_emp["tipo_empleo"] = np.where(
            df_emp["_merge"] != "both",
            "Desconocido",
            np.where(
                tipo_emp_norm == AFILIACION_VOLUNTARIA_CODE,
                "Afiliacion Voluntaria",
                "Relacion de Dependencia",
            ),
        )
        df_emp["trabaja"] = df_emp["_merge"] == "both"

    if tipo_empleo_sel != "Todos":
        df_emp = df_emp[df_emp["tipo_empleo"] == tipo_empleo_sel]
        if df_emp.empty:
            return u.iloc[0:0]

    agg = df_emp.groupby("hogar_id", as_index=False).agg(n_trab=("trabaja", "sum"))
    if cant_papas_trab in (0, 1, 2):
        agg = agg[agg["n_trab"] == cant_papas_trab]
        if agg.empty:
            return u.iloc[0:0]

    hogares_ok = set(agg["hogar_id"].unique().tolist())
    return u[u["hogar_id"].isin(hogares_ok)].copy()


def _contar_hogares(universo_familiares: pd.DataFrame) -> int:
    if universo_familiares.empty:
        return 0
    u = _normalizar_ids_familia(universo_familiares.copy())
    u["hogar_id"] = u.apply(
        lambda r: "|".join(sorted([str(r["CED_PADRE"]), str(r["CED_MADRE"])])), axis=1
    )
    mask_sin_info = (u["CED_PADRE"] == 0) & (u["CED_MADRE"] == 0)
    if mask_sin_info.any():
        u.loc[mask_sin_info, "hogar_id"] = (
            "sin_info_" + u.loc[mask_sin_info, "IDENTIFICACION"].astype(str)
        )
    return int(u["hogar_id"].nunique())


def calcular_vulnerabilidad_familias(
    ids_estudiantes: set[int],
    universo_familiares: pd.DataFrame,
    empleo: pd.DataFrame,
    deudas: pd.DataFrame,
) -> pd.DataFrame:
    if not ids_estudiantes:
        return pd.DataFrame()

    df_est = pd.DataFrame({"IDENTIFICACION": sorted(ids_estudiantes)})
    df = df_est.merge(universo_familiares, on="IDENTIFICACION", how="left")

    for col in ["CED_PADRE", "CED_MADRE"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["vulnerable"] = False
    df["en_riesgo"] = False
    df["motivos_vulnerabilidad"] = ""
    df["contador_riesgos"] = 0

    personas_con_ingresos: set[int] = set()
    ingreso_anual_por_id: dict[int, float] = {}
    if not empleo.empty and "SALARIO" in empleo.columns:
        emp = empleo.copy()
        emp["SALARIO"] = pd.to_numeric(emp["SALARIO"], errors="coerce").fillna(0)
        personas_con_ingresos = set(
            emp[emp["SALARIO"] > 0]["IDENTIFICACION"].astype(int).unique().tolist()
        )
        ingreso_anual_por_id = (
            emp.groupby("IDENTIFICACION")["SALARIO"].sum() * 14
        ).to_dict()

    deuda_total_por_id: dict[int, float] = {}
    deudores_criticos: set[int] = set()
    if not deudas.empty and "VALOR" in deudas.columns:
        deu = deudas.copy()
        deu["VALOR"] = pd.to_numeric(deu["VALOR"], errors="coerce").fillna(0)
        if "COD_CALIFICACION" in deu.columns:
            deu["COD_CALIFICACION"] = (
                deu["COD_CALIFICACION"].astype(str).str.upper().str.strip()
            )
            deu_crit = deu[deu["COD_CALIFICACION"].isin(["D", "E"])].copy()
        else:
            deu_crit = deu.iloc[0:0].copy()

        if not deu_crit.empty:
            deuda_total_por_id = (
                deu_crit.groupby("IDENTIFICACION")["VALOR"].sum().to_dict()
            )
            deudores_criticos = set(
                deu_crit["IDENTIFICACION"].astype(int).unique().tolist()
            )

    for idx, est in df.iterrows():
        motivos = []
        contador = 0

        ced_padre = int(est["CED_PADRE"])
        ced_madre = int(est["CED_MADRE"])
        tiene_padre = ced_padre != 0
        tiene_madre = ced_madre != 0

        if not tiene_padre and not tiene_madre:
            df.loc[idx, ["vulnerable", "en_riesgo"]] = [True, False]
            df.loc[idx, "contador_riesgos"] = 2
            df.loc[idx, "motivos_vulnerabilidad"] = "Sin informacion familiar"
            continue

        padre_sin_empleo = tiene_padre and (ced_padre not in personas_con_ingresos)
        madre_sin_empleo = tiene_madre and (ced_madre not in personas_con_ingresos)

        if tiene_padre and tiene_madre:
            if padre_sin_empleo and madre_sin_empleo:
                motivos.append("Familia sin empleo")
                contador += 1
        elif tiene_padre and not tiene_madre:
            if padre_sin_empleo:
                motivos.append("Familia sin empleo")
                contador += 1
        elif not tiene_padre and tiene_madre:
            if madre_sin_empleo:
                motivos.append("Familia sin empleo")
                contador += 1

        if deuda_total_por_id:
            tiene_deuda_critica = False
            deuda_total = 0.0
            ingreso_anual = 0.0

            if tiene_padre:
                deuda_total += deuda_total_por_id.get(ced_padre, 0.0)
                ingreso_anual += ingreso_anual_por_id.get(ced_padre, 0.0)
                if ced_padre in deudores_criticos:
                    tiene_deuda_critica = True
            if tiene_madre:
                deuda_total += deuda_total_por_id.get(ced_madre, 0.0)
                ingreso_anual += ingreso_anual_por_id.get(ced_madre, 0.0)
                if ced_madre in deudores_criticos:
                    tiene_deuda_critica = True

            if tiene_deuda_critica:
                if ingreso_anual > 0 and (deuda_total / ingreso_anual) >= 2.90:
                    motivos.append("Deuda familiar critica (D/E)")
                    contador += 1

        if contador > 0:
            if contador >= 2:
                df.loc[idx, ["vulnerable", "en_riesgo"]] = [True, False]
            else:
                df.loc[idx, ["vulnerable", "en_riesgo"]] = [False, True]
            df.loc[idx, "contador_riesgos"] = contador
            df.loc[idx, "motivos_vulnerabilidad"] = "; ".join(motivos)

    return df


st.set_page_config(page_title="Vulnerabilidad", page_icon="⚠️", layout="wide")
st.title("⚠️ Analisis de Vulnerabilidad")

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

c1, c2, c3, c4, c5 = st.columns(5)
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
    tipo_empleo_sel = st.selectbox("Tipo de empleo", options=TIPOS_EMPLEO, index=0)
with c4:
    grupo_quintil_sel = st.selectbox(
        "Grupo de quintil",
        options=GRUPO_QUINTIL_OPTS,
        index=0,
    )
with c5:
    rango_quintil_sel = st.selectbox(
        "Rango de quintil",
        options=RANGO_QUINTIL_OPTS,
        index=0,
    )

ids_base = set(
    estudiantes_filtrados["IDENTIFICACION"].dropna().astype(int).unique().tolist()
)

universo_filtrado = _filtrar_universo_por_filtros(
    universo_familiares,
    empleo,
    ids_base=ids_base,
    cant_papas=cant_papas,
    cant_papas_trab=cant_papas_trab,
    tipo_empleo_sel=tipo_empleo_sel,
)

if universo_filtrado.empty:
    st.info("No hay familias que cumplan los filtros seleccionados.")
    st.stop()

rangos_sel = _rangos_por_nombre(rango_quintil_sel)
quintiles_estudiante = _quintil_por_estudiante(universo_filtrado, empleo, rangos_sel)
if quintiles_estudiante.empty:
    universo_filtrado["grupo_quintil"] = "sin info"
else:
    universo_filtrado = universo_filtrado.merge(
        quintiles_estudiante,
        on="IDENTIFICACION",
        how="left",
    )
    universo_filtrado["grupo_quintil"] = (
        universo_filtrado["grupo_quintil"].astype(str).str.strip().str.lower()
    )
    universo_filtrado["grupo_quintil"] = universo_filtrado["grupo_quintil"].replace(
        {"nan": "sin info", "": "sin info"}
    )

if grupo_quintil_sel != "Todos":
    universo_filtrado = universo_filtrado[
        universo_filtrado["grupo_quintil"] == grupo_quintil_sel
    ].copy()
    if universo_filtrado.empty:
        st.info(
            f"No hay familias para el grupo de quintil `{grupo_quintil_sel}` usando rango `{rango_quintil_sel}`."
        )
        st.stop()

ids_estudiantes = set(universo_filtrado["IDENTIFICACION"].unique().tolist())

vulnerabilidad = calcular_vulnerabilidad_familias(
    ids_estudiantes, universo_familiares, empleo, deudas
)

if vulnerabilidad.empty:
    st.info("No hay datos disponibles para el grupo seleccionado.")
    st.stop()

total_estudiantes = len(vulnerabilidad)
total_hogares = _contar_hogares(universo_filtrado)
alta_vulnerabilidad = int(vulnerabilidad["vulnerable"].sum())
en_riesgo = int(vulnerabilidad["en_riesgo"].sum())
sin_riesgo = total_estudiantes - alta_vulnerabilidad - en_riesgo

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    tarjeta_simple("Total Personas", f"{total_estudiantes}", COLORES["azul"])
with col2:
    tarjeta_simple("Total Hogares", f"{total_hogares}", COLORES["azul"])
with col3:
    tarjeta_simple("Alta Vulnerabilidad", f"{alta_vulnerabilidad}", COLORES["rojo"])
with col4:
    tarjeta_simple("En Riesgo", f"{en_riesgo}", COLORES["naranja"])
with col5:
    tarjeta_simple("Sin Riesgo", f"{sin_riesgo}", COLORES["verde"])

if alta_vulnerabilidad > 0 or en_riesgo > 0:
    st.markdown("---")
    st.subheader("Detalle de Personas Vulnerables")

    df_alta = vulnerabilidad[vulnerabilidad["vulnerable"] == True][
        ["IDENTIFICACION", "motivos_vulnerabilidad"]
    ].copy()
    if not df_alta.empty:
        st.write("Alta Vulnerabilidad (2 o mas condiciones)")
        df_alta.columns = ["Identificacion", "Motivos de Vulnerabilidad"]
        st.dataframe(df_alta, width="stretch")

    df_riesgo = vulnerabilidad[vulnerabilidad["en_riesgo"] == True][
        ["IDENTIFICACION", "motivos_vulnerabilidad"]
    ].copy()
    if not df_riesgo.empty:
        st.write("En Riesgo (1 condicion)")
        df_riesgo.columns = ["Identificacion", "Motivos de Vulnerabilidad"]
        st.dataframe(df_riesgo, width="stretch")
