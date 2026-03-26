from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
import streamlit as st

from utils.excel_loader import load_excel_sheet
from utils.propensity_helpers import (
    REQUIRED_SHEETS,
    _canonical_sex,
    _choose_household_state,
    _household_id,
    _prepare_deuda,
    _prepare_empleo,
    _prepare_familia,
    _prepare_info,
    _prepare_students,
    _to_numeric,
)
from utils.quintile_ranges import asignar_quintil_por_rangos, calcular_rangos_quintiles
from utils.udla_clusters import run_udla_cluster_analysis

st.set_page_config(page_title="Clusters Universidad", page_icon="C", layout="wide")
REGISTRATION_BASE_YEAR = 2025


def _clean_series(series: pd.Series, default: str = "Sin dato") -> pd.Series:
    return (
        pd.Series(series)
        .fillna("")
        .astype(str)
        .str.strip()
        .replace({"": default, "nan": default, "None": default})
    )


def _round_half_up(value: float) -> int:
    number = float(pd.to_numeric(value, errors="coerce"))
    if pd.isna(number):
        return 0
    return int(math.floor(number + 0.5))


def _normalize_id(series: pd.Series) -> pd.Series:
    return (
        pd.Series(series)
        .astype("string")
        .fillna("0")
        .astype(str)
        .str.strip()
        .replace({"": "0", "nan": "0", "None": "0", "<NA>": "0"})
    )


def _find_column_name(df: pd.DataFrame, aliases: list[str]) -> str | None:
    normalized = {
        str(column).strip().upper().replace(" ", "_"): str(column)
        for column in df.columns
    }
    for alias in aliases:
        match = normalized.get(alias.strip().upper().replace(" ", "_"))
        if match is not None:
            return match
    return None


def _extract_student_registration_date(students_sheet: pd.DataFrame) -> pd.DataFrame:
    raw = students_sheet.copy()
    id_col = _find_column_name(raw, ["IDENTIFICACION", "CEDULA", "Cedula"])
    fecha_col = _find_column_name(raw, ["FECHA_REGISTRO", "FECHA REGISTRO"])

    if id_col is None:
        return pd.DataFrame(columns=["IDENTIFICACION", "est_FECHA_REGISTRO"])

    out = pd.DataFrame(index=raw.index)
    out["IDENTIFICACION"] = _normalize_id(raw[id_col])
    if fecha_col is None:
        out["est_FECHA_REGISTRO"] = pd.NaT
    else:
        out["est_FECHA_REGISTRO"] = pd.to_datetime(
            raw[fecha_col],
            errors="coerce",
            dayfirst=True,
        )
    out = out[out["IDENTIFICACION"] != "0"].copy()
    return out.drop_duplicates(subset=["IDENTIFICACION"], keep="first")


@st.cache_data(show_spinner=False)
def _build_universidades_base() -> pd.DataFrame:
    sheets = {
        sheet_name: load_excel_sheet(sheet_name, "Universidades.xlsx").copy()
        for sheet_name in REQUIRED_SHEETS
    }

    students = _prepare_students(sheets["Estudiantes"], default_university=None)
    students_reg = _extract_student_registration_date(sheets["Estudiantes"])
    students = students.merge(students_reg, on="IDENTIFICACION", how="left")
    familia = _prepare_familia(sheets["Universo Familiares"])
    info = _prepare_info(sheets["Informacion Personal"])
    empleo = _prepare_empleo(sheets["Empleos"])
    deuda = _prepare_deuda(sheets["Deudas"])

    base = students.merge(familia, on="IDENTIFICACION", how="left")
    base["CED_PADRE"] = base["CED_PADRE"].fillna("0")
    base["CED_MADRE"] = base["CED_MADRE"].fillna("0")
    base["hogar_id"] = _household_id(base)

    info_student = info.rename(
        columns={
            "SEXO_CANON": "est_SEXO_CANON",
            "EDAD": "est_EDAD",
        }
    )
    base = base.merge(
        info_student[["IDENTIFICACION", "est_SEXO_CANON", "est_EDAD"]],
        on="IDENTIFICACION",
        how="left",
    )

    padre_info = info.rename(
        columns={
            "IDENTIFICACION": "CED_PADRE",
            "ESTADO_CANON": "padre_ESTADO_CANON",
            "HIJOS_NUM": "padre_HIJOS_NUM",
            "NIVEL_CANON": "padre_NIVEL_CANON",
            "FECHA_EXP": "padre_FECHA_EXP",
        }
    )
    madre_info = info.rename(
        columns={
            "IDENTIFICACION": "CED_MADRE",
            "ESTADO_CANON": "madre_ESTADO_CANON",
            "HIJOS_NUM": "madre_HIJOS_NUM",
            "NIVEL_CANON": "madre_NIVEL_CANON",
            "FECHA_EXP": "madre_FECHA_EXP",
        }
    )
    base = base.merge(
        padre_info[
            [
                "CED_PADRE",
                "padre_ESTADO_CANON",
                "padre_HIJOS_NUM",
                "padre_NIVEL_CANON",
                "padre_FECHA_EXP",
            ]
        ],
        on="CED_PADRE",
        how="left",
    ).merge(
        madre_info[
            [
                "CED_MADRE",
                "madre_ESTADO_CANON",
                "madre_HIJOS_NUM",
                "madre_NIVEL_CANON",
                "madre_FECHA_EXP",
            ]
        ],
        on="CED_MADRE",
        how="left",
    )

    padre_emp = empleo.rename(
        columns={
            "IDENTIFICACION": "CED_PADRE",
            "salario": "salario_padre",
        }
    )
    madre_emp = empleo.rename(
        columns={
            "IDENTIFICACION": "CED_MADRE",
            "salario": "salario_madre",
        }
    )
    base = base.merge(
        padre_emp[["CED_PADRE", "salario_padre"]],
        on="CED_PADRE",
        how="left",
    ).merge(
        madre_emp[["CED_MADRE", "salario_madre"]],
        on="CED_MADRE",
        how="left",
    )

    padre_deu = deuda.rename(
        columns={
            "IDENTIFICACION": "CED_PADRE",
            "deuda_total": "deuda_padre",
        }
    )
    madre_deu = deuda.rename(
        columns={
            "IDENTIFICACION": "CED_MADRE",
            "deuda_total": "deuda_madre",
        }
    )
    base = base.merge(
        padre_deu[["CED_PADRE", "deuda_padre"]],
        on="CED_PADRE",
        how="left",
    ).merge(
        madre_deu[["CED_MADRE", "deuda_madre"]],
        on="CED_MADRE",
        how="left",
    )

    numeric_cols = [
        "salario_padre",
        "salario_madre",
        "deuda_padre",
        "deuda_madre",
        "padre_HIJOS_NUM",
        "madre_HIJOS_NUM",
    ]
    for column in numeric_cols:
        base[column] = _to_numeric(base.get(column, 0.0), default=0.0)

    base["sexo_estudiante"] = base["est_SEXO_CANON"].where(
        base["est_SEXO_CANON"].notna() & (base["est_SEXO_CANON"] != ""),
        base["GENERO_CANON"],
    )
    base["sexo_estudiante"] = base["sexo_estudiante"].map(_canonical_sex)
    registro_year = (
        pd.to_datetime(base["est_FECHA_REGISTRO"], errors="coerce", dayfirst=True).dt.year
        if "est_FECHA_REGISTRO" in base.columns
        else pd.Series(np.nan, index=base.index, dtype="float64")
    )
    ajuste_edad = 4 + (REGISTRATION_BASE_YEAR - registro_year)
    ajuste_edad = ajuste_edad.where(registro_year.notna(), 4)
    base["edad_estudiante"] = (
        _to_numeric(base["est_EDAD"], default=np.nan) - ajuste_edad
    ).clip(lower=0)
    base["salario_hogar"] = base["salario_padre"] + base["salario_madre"]
    base["deuda_hogar"] = base["deuda_padre"] + base["deuda_madre"]
    base["hijos_hogar"] = base[["padre_HIJOS_NUM", "madre_HIJOS_NUM"]].max(axis=1)
    base["padres_presentes"] = (
        (base["CED_PADRE"] != "0").astype(int) + (base["CED_MADRE"] != "0").astype(int)
    )
    base["padres_con_superior"] = (
        (base["padre_NIVEL_CANON"] == "SUPERIOR").astype(int)
        + (base["madre_NIVEL_CANON"] == "SUPERIOR").astype(int)
    )
    base["primera_generacion"] = (
        (base["padres_presentes"] > 0) & (base["padres_con_superior"] == 0)
    ).astype(int)
    base["estado_hogar"] = _choose_household_state(base)

    return base[
        [
            "IDENTIFICACION",
            "Universidad",
            "hogar_id",
            "carrera",
            "unidad_academica",
            "tipo_estudiante",
            "sexo_estudiante",
            "edad_estudiante",
            "salario_hogar",
            "deuda_hogar",
            "hijos_hogar",
            "primera_generacion",
            "estado_hogar",
        ]
    ].copy()


def _prepare_cluster_input(
    student_df: pd.DataFrame,
    income_ranges: dict[int, dict[str, float]],
    debt_ranges: dict[int, dict[str, float]],
) -> pd.DataFrame:
    base = student_df.copy()
    if base.empty:
        return base

    base["tipo_estudiante"] = _clean_series(base["tipo_estudiante"])
    base["facultad"] = _clean_series(base["unidad_academica"])
    base["carrera"] = _clean_series(base["carrera"])
    base["sexo_estudiante"] = _clean_series(base["sexo_estudiante"], default="DESCONOCIDO")
    base["estado_hogar"] = _clean_series(base["estado_hogar"], default="Desconocido")
    base["edad_estudiante"] = pd.to_numeric(base["edad_estudiante"], errors="coerce")

    base["hijos_hogar"] = (
        pd.to_numeric(base["hijos_hogar"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .round()
        .astype(int)
    )
    base["hijos_hogar_promedio"] = (
        base.groupby("hogar_id")["hijos_hogar"].transform("mean").fillna(0.0)
    )

    base["primera_generacion"] = (
        pd.to_numeric(base["primera_generacion"], errors="coerce")
        .fillna(0)
        .clip(lower=0, upper=1)
        .astype(int)
    )
    base["estudiante_quito"] = (
        base["IDENTIFICACION"].astype(str).str.strip().str.startswith("17").astype(int)
    )

    base["quintil_ingreso_hogar"] = base["salario_hogar"].apply(
        lambda value: asignar_quintil_por_rangos(value, income_ranges, vacio="Sin empleo")
    )
    base["quintil_ingreso_num"] = (
        pd.to_numeric(
            base["quintil_ingreso_hogar"].replace({"Sin empleo": "0"}),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    base["quintil_deuda_hogar"] = base["deuda_hogar"].apply(
        lambda value: asignar_quintil_por_rangos(value, debt_ranges, vacio="Sin deuda")
    )
    base["quintil_deuda_num"] = (
        pd.to_numeric(
            base["quintil_deuda_hogar"].replace({"Sin deuda": "0"}),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    return base


def _run_all_university_clusters(
    university_base_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if university_base_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    students_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []

    universities = sorted(
        [
            value
            for value in _clean_series(university_base_df["Universidad"]).unique().tolist()
            if value != "Sin dato"
        ]
    )

    for university_name in universities:
        subset = university_base_df[
            _clean_series(university_base_df["Universidad"]) == university_name
        ].copy()
        if subset.empty:
            continue

        ingreso_ranges = calcular_rangos_quintiles(subset["salario_hogar"])
        deuda_ranges = calcular_rangos_quintiles(subset["deuda_hogar"])
        cluster_input_df = _prepare_cluster_input(
            subset,
            ingreso_ranges,
            deuda_ranges,
        )
        analysis = run_udla_cluster_analysis(cluster_input_df)
        students_df = analysis["students"].copy()
        summary_df = analysis["summary"].copy()

        if students_df.empty or summary_df.empty:
            continue

        students_df["Universidad"] = university_name
        summary_df["universidad_referencia"] = university_name

        students_parts.append(students_df)
        summary_parts.append(summary_df)

    if not students_parts or not summary_parts:
        return pd.DataFrame(), pd.DataFrame()

    return (
        pd.concat(students_parts, ignore_index=True),
        pd.concat(summary_parts, ignore_index=True),
    )


def _detail_cluster_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["universidad_referencia"] = _clean_series(
        out.get("universidad_referencia", pd.Series(index=out.index, dtype="object"))
    )
    out["hijos_promedio_entero"] = out["hijos_promedio"].apply(_round_half_up)
    out["hogares_con_deuda_q"] = out.apply(
        lambda row: (
            f"{int(row['hogares_con_deuda']):,} ({row['quintil_deuda_modal']})"
            if str(row.get("quintil_deuda_modal", "Sin deuda")).strip()
            else f"{int(row['hogares_con_deuda']):,}"
        ),
        axis=1,
    )

    columns = [
        "cluster",
        "universidad_referencia",
        "estudiantes",
        "sexo_modal",
        "pct_edad_15_19",
        "pct_edad_20_22",
        "pct_edad_23_25",
        "pct_edad_mas_25",
        "pct_quito",
        "quintil_ingreso_modal",
        "hijos_promedio_entero",
        "hogares_con_deuda_q",
        "estado_hogar_modal",
        "tipo_estudiantes_pg",
    ]
    rename_map = {
        "cluster": "Cluster",
        "universidad_referencia": "Universidad referencia",
        "estudiantes": "Cantidad estudiantes",
        "sexo_modal": "Genero estudiante",
        "pct_edad_15_19": "15-19 anos",
        "pct_edad_20_22": "20-22 anos",
        "pct_edad_23_25": "23-25 anos",
        "pct_edad_mas_25": "Mas de 25 anos",
        "pct_quito": "Es de Quito (%)",
        "quintil_ingreso_modal": "Quintil ingresos",
        "hijos_promedio_entero": "Promedio hijos",
        "hogares_con_deuda_q": "Hogares con deuda (Q deuda)",
        "estado_hogar_modal": "Estado del hogar",
        "tipo_estudiantes_pg": "Tipo de estudiantes",
    }
    out["cluster_orden"] = (
        pd.to_numeric(
            out["cluster"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
        )
        .fillna(999)
        .astype(int)
    )
    out = out.sort_values(
        ["universidad_referencia", "cluster_orden", "cluster"]
    ).reset_index(drop=True)
    return out[columns].rename(columns=rename_map)


def _to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Detalle") -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()


st.title("Clusters Universidad")
st.caption("Vista de detalle sobre todo el universo de `db/Universidades.xlsx`.")

with st.spinner("Preparando datos de Universidades..."):
    university_base_df = _build_universidades_base()

if university_base_df.empty:
    st.info("No hay datos disponibles en `db/Universidades.xlsx`.")
    st.stop()

with st.spinner("Calculando clusters por universidad..."):
    students_df, summary_df = _run_all_university_clusters(university_base_df)
cluster_count = int(len(summary_df))

if students_df.empty or summary_df.empty:
    st.info("No fue posible construir clusters con los datos disponibles.")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Universidades",
    f"{int(_clean_series(students_df['Universidad']).nunique()):,}",
)
m2.metric("Estudiantes", f"{int(students_df['IDENTIFICACION'].nunique()):,}")
m3.metric("Hogares", f"{int(students_df['hogar_id'].nunique()):,}")
m4.metric("Clusters", cluster_count)

st.markdown("### Detalle")
detail_df = _detail_cluster_table(summary_df)
st.dataframe(
    detail_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Cantidad estudiantes": st.column_config.NumberColumn(
            "Cantidad estudiantes",
            format="%d",
        ),
        "15-19 anos": st.column_config.NumberColumn("15-19 anos", format="%.1f%%"),
        "20-22 anos": st.column_config.NumberColumn("20-22 anos", format="%.1f%%"),
        "23-25 anos": st.column_config.NumberColumn("23-25 anos", format="%.1f%%"),
        "Mas de 25 anos": st.column_config.NumberColumn("Mas de 25 anos", format="%.1f%%"),
        "Es de Quito (%)": st.column_config.NumberColumn(
            "Es de Quito (%)",
            format="%.1f%%",
        ),
        "Promedio hijos": st.column_config.NumberColumn("Promedio hijos", format="%d"),
    },
)
st.download_button(
    label="Descargar detalle en Excel (.xlsx)",
    data=_to_excel_bytes(detail_df, sheet_name="Detalle Universidades"),
    file_name="detalle_clusters_universidades.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    key="clusters_universidades_detalle_xlsx",
)
