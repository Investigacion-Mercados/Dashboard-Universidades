from __future__ import annotations

import html
import io
import math
import re

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


def _student_cluster_detail_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["Universidad"] = _clean_series(
        out.get("Universidad", pd.Series(index=out.index, dtype="object"))
    )
    out["cluster"] = _clean_series(
        out.get("cluster", pd.Series(index=out.index, dtype="object"))
    )
    out["estudiante_quito"] = _yes_no_series(out.get("estudiante_quito", 0))
    out["primera_generacion"] = _yes_no_series(out.get("primera_generacion", 0))
    out["edad_estudiante"] = pd.to_numeric(
        out.get("edad_estudiante", np.nan), errors="coerce"
    )
    out["hijos_hogar_promedio"] = pd.to_numeric(
        out.get("hijos_hogar_promedio", np.nan), errors="coerce"
    )
    out["cluster_orden"] = (
        pd.to_numeric(
            out["cluster"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
        )
        .fillna(999)
        .astype(int)
    )
    out = out.sort_values(
        ["Universidad", "cluster_orden", "cluster", "IDENTIFICACION"]
    ).reset_index(drop=True)

    columns = [
        "Universidad",
        "cluster",
        "IDENTIFICACION",
        "hogar_id",
        "tipo_estudiante",
        "facultad",
        "carrera",
        "sexo_estudiante",
        "edad_estudiante",
        "estudiante_quito",
        "primera_generacion",
        "quintil_ingreso_hogar",
        "quintil_deuda_hogar",
        "hijos_hogar_promedio",
        "estado_hogar",
    ]
    rename_map = {
        "Universidad": "Universidad",
        "cluster": "Cluster",
        "IDENTIFICACION": "Identificacion",
        "hogar_id": "Hogar",
        "tipo_estudiante": "Tipo",
        "facultad": "Facultad",
        "carrera": "Carrera",
        "sexo_estudiante": "Genero",
        "edad_estudiante": "Edad",
        "estudiante_quito": "Quito",
        "primera_generacion": "Primera generacion",
        "quintil_ingreso_hogar": "Quintil ingreso",
        "quintil_deuda_hogar": "Quintil deuda",
        "hijos_hogar_promedio": "Hijos hogar prom.",
        "estado_hogar": "Estado del hogar",
    }
    available = [column for column in columns if column in out.columns]
    return out[available].rename(columns=rename_map)


def _age_bucket(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series("Sin dato", index=values.index, dtype="object")
    out[(values >= 15) & (values <= 19)] = "15-19 anos"
    out[(values >= 20) & (values <= 22)] = "20-22 anos"
    out[(values >= 23) & (values <= 25)] = "23-25 anos"
    out[values >= 26] = "Mas de 25 anos"
    return out


def _hijos_bucket(series: pd.Series) -> pd.Series:
    rounded = pd.Series(series).apply(_round_half_up)
    out = rounded.astype(str)
    out = out.where(rounded < 5, "5+")
    return out


def _yes_no_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    return numeric.map({1: "Si", 0: "No"}).fillna("No")


def _cluster_distribution_matrix(
    df: pd.DataFrame,
    category_series: pd.Series,
    cluster_order: list[str],
    *,
    category_order: list[str] | None = None,
    unique_id_col: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if df.empty:
        return pd.DataFrame(index=cluster_order), (category_order or [])

    chart_df = pd.DataFrame(
        {
            "cluster": _clean_series(df["cluster"]),
            "categoria": _clean_series(category_series),
        }
    )

    if unique_id_col is not None and unique_id_col in df.columns:
        chart_df["id_unico"] = _clean_series(df[unique_id_col], default="0")
        grouped = (
            chart_df.groupby(["cluster", "categoria"], as_index=False)["id_unico"]
            .nunique()
            .rename(columns={"id_unico": "conteo"})
        )
        totals = (
            chart_df.groupby("cluster", as_index=False)["id_unico"]
            .nunique()
            .rename(columns={"id_unico": "total"})
        )
    else:
        grouped = (
            chart_df.groupby(["cluster", "categoria"], as_index=False)
            .size()
            .rename(columns={"size": "conteo"})
        )
        totals = (
            chart_df.groupby("cluster", as_index=False)
            .size()
            .rename(columns={"size": "total"})
        )

    grouped = grouped.merge(totals, on="cluster", how="left")
    grouped["valor"] = grouped["conteo"] / grouped["total"] * 100.0

    if category_order is None:
        category_order = (
            grouped.groupby("categoria", as_index=False)["conteo"]
            .sum()
            .sort_values("conteo", ascending=False)["categoria"]
            .tolist()
        )

    matrix = grouped.pivot_table(
        index="cluster",
        columns="categoria",
        values="valor",
        aggfunc="sum",
        fill_value=0.0,
    )
    matrix = matrix.reindex(index=cluster_order, fill_value=0.0)
    matrix = matrix.reindex(columns=category_order, fill_value=0.0)
    return matrix, category_order


def _detail_chart_table(
    students_df: pd.DataFrame, summary_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    if students_df.empty or summary_df.empty:
        return pd.DataFrame(), {}

    cluster_order = summary_df["cluster"].dropna().astype(str).tolist()
    summary_counts = (
        summary_df.set_index("cluster")["estudiantes"]
        .reindex(cluster_order)
        .fillna(0)
        .astype(int)
    )

    table = pd.DataFrame(
        {
            "Cluster": cluster_order,
            "Cantidad estudiantes": summary_counts.values,
        }
    )

    age_order = ["15-19 anos", "20-22 anos", "23-25 anos", "Mas de 25 anos", "Sin dato"]
    genero_order = ["MUJER", "HOMBRE", "DESCONOCIDO"]
    quito_order = ["Si", "No"]
    ingreso_order = ["Sin empleo", "1", "2", "3", "4", "5"]
    hijos_order = ["0", "1", "2", "3", "4", "5+"]
    deuda_order = ["Sin deuda", "1", "2", "3", "4", "5"]
    tipo_order = ["Primera generacion", "No primera generacion"]

    edad_matrix, _ = _cluster_distribution_matrix(
        students_df,
        _age_bucket(students_df["edad_estudiante"]),
        cluster_order,
        category_order=age_order,
    )
    genero_matrix, _ = _cluster_distribution_matrix(
        students_df,
        students_df["sexo_estudiante"],
        cluster_order,
        category_order=genero_order,
    )
    quito_matrix, _ = _cluster_distribution_matrix(
        students_df,
        _yes_no_series(students_df["estudiante_quito"]),
        cluster_order,
        category_order=quito_order,
    )
    ingreso_matrix, _ = _cluster_distribution_matrix(
        students_df,
        students_df["quintil_ingreso_hogar"],
        cluster_order,
        category_order=ingreso_order,
    )
    hijos_matrix, _ = _cluster_distribution_matrix(
        students_df,
        _hijos_bucket(students_df["hijos_hogar_promedio"]),
        cluster_order,
        category_order=hijos_order,
    )
    deuda_matrix, _ = _cluster_distribution_matrix(
        students_df,
        students_df["quintil_deuda_hogar"],
        cluster_order,
        category_order=deuda_order,
        unique_id_col="hogar_id",
    )
    estado_matrix, estado_order = _cluster_distribution_matrix(
        students_df,
        students_df["estado_hogar"],
        cluster_order,
        category_order=None,
    )
    tipo_matrix, _ = _cluster_distribution_matrix(
        students_df,
        _yes_no_series(students_df["primera_generacion"]).replace(
            {"Si": "Primera generacion", "No": "No primera generacion"}
        ),
        cluster_order,
        category_order=tipo_order,
    )

    if not estado_order:
        estado_order = ["Sin dato"]

    def _as_row_lists(matrix: pd.DataFrame, order: list[str]) -> list[list[float]]:
        aligned = matrix.reindex(index=cluster_order, columns=order, fill_value=0.0)
        return aligned.astype(float).values.tolist()

    table["Genero estudiante"] = _as_row_lists(genero_matrix, genero_order)
    table["Edad"] = _as_row_lists(edad_matrix, age_order)
    table["Es de Quito (%)"] = _as_row_lists(quito_matrix, quito_order)
    table["Quintil ingresos"] = _as_row_lists(ingreso_matrix, ingreso_order)
    table["Promedio hijos"] = _as_row_lists(hijos_matrix, hijos_order)
    table["Hogares con deuda (Q deuda)"] = _as_row_lists(deuda_matrix, deuda_order)
    table["Estado del hogar"] = _as_row_lists(estado_matrix, estado_order)
    table["Tipo de estudiantes"] = _as_row_lists(tipo_matrix, tipo_order)

    legend_map = {
        "Genero estudiante": genero_order,
        "Edad": age_order,
        "Es de Quito (%)": quito_order,
        "Quintil ingresos": ingreso_order,
        "Promedio hijos": hijos_order,
        "Hogares con deuda (Q deuda)": deuda_order,
        "Estado del hogar": estado_order,
        "Tipo de estudiantes": tipo_order,
    }
    return table, legend_map


def _stacked_cell_html(
    values: list[float],
    categories: list[str],
    colors: list[str],
) -> str:
    clean_values = [max(float(v), 0.0) for v in values]
    total = float(sum(clean_values))
    if total <= 0:
        return '<div class="detalle-bar-empty">Sin dato</div>'

    segments: list[str] = []
    for idx, (cat, value) in enumerate(zip(categories, clean_values)):
        if value <= 0:
            continue
        width = value / total * 100.0
        color = colors[idx % len(colors)]
        tooltip = html.escape(f"{cat}: {value:.1f}%")
        segments.append(
            (
                f'<span class="detalle-seg" style="width:{width:.4f}%;background:{color};" '
                f'title="{tooltip}"></span>'
            )
        )

    if not segments:
        return '<div class="detalle-bar-empty">Sin dato</div>'
    return f'<div class="detalle-bar">{"".join(segments)}</div>'


def _detail_chart_table_html(
    detail_chart_df: pd.DataFrame, legend_map: dict[str, list[str]]
) -> str:
    if detail_chart_df.empty:
        return "<p>Sin datos.</p>"

    chart_cols = [
        "Genero estudiante",
        "Edad",
        "Es de Quito (%)",
        "Quintil ingresos",
        "Promedio hijos",
        "Hogares con deuda (Q deuda)",
        "Estado del hogar",
        "Tipo de estudiantes",
    ]
    base_cols = ["Cluster", "Cantidad estudiantes"]
    columns = [*base_cols, *chart_cols]

    palette = [
        "#E8D677",
        "#E3B56F",
        "#DE926C",
        "#D96A6A",
        "#A78BFA",
        "#60A5FA",
        "#34D399",
        "#FBBF24",
    ]

    header_html = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    rows_html: list[str] = []

    for _, row in detail_chart_df.iterrows():
        cells: list[str] = []
        cells.append(f"<td>{html.escape(str(row['Cluster']))}</td>")
        cells.append(
            f"<td>{int(pd.to_numeric(row['Cantidad estudiantes'], errors='coerce') or 0):,}</td>"
        )

        for col in chart_cols:
            categories = legend_map.get(col, [])
            raw_values = row[col]
            if isinstance(raw_values, list):
                values = [float(v) for v in raw_values]
            else:
                values = []
            bar_html = _stacked_cell_html(values, categories, palette)
            cells.append(f"<td>{bar_html}</td>")

        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    return f"""
<style>
.detalle-wrap {{
  width: 100%;
  overflow-x: auto;
}}
.detalle-table {{
  border-collapse: collapse;
  width: 100%;
  min-width: 1300px;
}}
.detalle-table th, .detalle-table td {{
  border: 1px solid rgba(148, 163, 184, 0.25);
  padding: 8px 10px;
  text-align: left;
  vertical-align: middle;
}}
.detalle-table th {{
  font-weight: 600;
  background: rgba(15, 23, 42, 0.2);
}}
.detalle-table td:nth-child(2) {{
  text-align: right;
}}
.detalle-bar {{
  height: 18px;
  width: 100%;
  min-width: 170px;
  border-radius: 4px;
  overflow: hidden;
  background: rgba(148, 163, 184, 0.18);
  display: flex;
}}
.detalle-seg {{
  height: 100%;
  display: inline-block;
}}
.detalle-bar-empty {{
  color: #94a3b8;
  font-size: 12px;
}}
</style>
<div class="detalle-wrap">
  <table class="detalle-table">
    <thead><tr>{header_html}</tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>
</div>
"""


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

if students_df.empty or summary_df.empty:
    st.info("No fue posible construir clusters con los datos disponibles.")
    st.stop()

cluster_count = int(len(summary_df))
m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Universidades",
    f"{int(_clean_series(students_df['Universidad']).nunique()):,}",
)
m2.metric("Estudiantes", f"{int(students_df['IDENTIFICACION'].nunique()):,}")
m3.metric("Hogares", f"{int(students_df['hogar_id'].nunique()):,}")
m4.metric("Clusters", cluster_count)

tab_detalle, tab_detalle_graficos = st.tabs(["Detalle", "Detalle Graficos"])

with tab_detalle:
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

    st.markdown("#### Detalle estudiante")
    student_university_options = sorted(
        [
            value
            for value in _clean_series(students_df["Universidad"]).unique().tolist()
            if value != "Sin dato"
        ]
    )
    if not student_university_options:
        st.info("No hay universidades disponibles para el detalle de estudiantes.")
    else:
        selected_student_university = st.selectbox(
            "Universidad (detalle estudiante)",
            options=student_university_options,
            index=0,
            key="clusters_universidades_detalle_estudiantes_universidad",
        )
        university_students_df = students_df[
            _clean_series(students_df["Universidad"]) == selected_student_university
        ].copy()

        cluster_values = _clean_series(university_students_df["cluster"]).unique().tolist()

        def _cluster_sort_key(value: str) -> tuple[int, str]:
            match = re.search(r"(\d+)", str(value))
            order = int(match.group(1)) if match else 999
            return order, str(value)

        cluster_ordered = sorted(cluster_values, key=_cluster_sort_key)
        cluster_options = ["Todos"] + cluster_ordered
        selected_student_cluster = st.selectbox(
            "Cluster",
            options=cluster_options,
            index=0,
            key="clusters_universidades_detalle_estudiantes_cluster",
        )

        student_detail_df = (
            university_students_df.copy()
            if selected_student_cluster == "Todos"
            else university_students_df[
                _clean_series(university_students_df["cluster"])
                == selected_student_cluster
            ].copy()
        )
        student_detail_view = _student_cluster_detail_table(student_detail_df)
        st.dataframe(
            student_detail_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Edad": st.column_config.NumberColumn("Edad", format="%.0f"),
                "Hijos hogar prom.": st.column_config.NumberColumn(
                    "Hijos hogar prom.", format="%.1f"
                ),
            },
        )

        university_suffix = "".join(
            char if char.isalnum() else "_"
            for char in selected_student_university.lower()
        ).strip("_")
        if not university_suffix:
            university_suffix = "universidad"
        cluster_suffix = (
            "todos"
            if selected_student_cluster == "Todos"
            else "".join(
                char if char.isalnum() else "_"
                for char in selected_student_cluster.lower()
            ).strip("_")
        )
        if not cluster_suffix:
            cluster_suffix = "cluster"

        st.download_button(
            label="Descargar detalle estudiante en Excel (.xlsx)",
            data=_to_excel_bytes(
                student_detail_view,
                sheet_name="Detalle Estudiantes",
            ),
            file_name=(
                f"detalle_estudiantes_{university_suffix}_{cluster_suffix}.xlsx"
            ),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="clusters_universidades_detalle_estudiantes_xlsx",
        )

with tab_detalle_graficos:
    university_options = sorted(
        [
            value
            for value in _clean_series(summary_df["universidad_referencia"]).unique().tolist()
            if value != "Sin dato"
        ]
    )
    if not university_options:
        st.info("No hay universidades disponibles para mostrar detalle con graficos.")
    else:
        selected_university = st.selectbox(
            "Universidad",
            options=university_options,
            index=0,
            key="clusters_universidades_detalle_universidad",
        )
        students_view_df = students_df[
            _clean_series(students_df["Universidad"]) == selected_university
        ].copy()
        summary_view_df = summary_df[
            _clean_series(summary_df["universidad_referencia"]) == selected_university
        ].copy()

        st.caption(
            "Tabla de detalle con mini-graficos de distribucion (%) por cluster."
        )
        detail_chart_df, legend_map = _detail_chart_table(
            students_view_df, summary_view_df
        )
        if detail_chart_df.empty:
            st.info("No hay datos suficientes para construir detalle con graficos.")
        else:
            st.markdown(
                _detail_chart_table_html(detail_chart_df, legend_map),
                unsafe_allow_html=True,
            )
