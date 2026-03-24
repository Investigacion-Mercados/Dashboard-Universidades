from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from scipy.cluster.vq import kmeans2
from scipy.optimize import minimize
from scipy.special import expit

from utils.comparacion_helpers import asignar_parroquia, cargar_parroquias
from utils.excel_loader import load_excel_sheet
from utils.quintile_ranges import asignar_quintil_por_rangos, calcular_rangos_quintiles
from utils.student_columns import normalize_university_column

REQUIRED_SHEETS = [
    "Estudiantes",
    "Universo Familiares",
    "Informacion Personal",
    "Empleos",
    "Deudas",
]

RISK_MAP = {
    "A1": "Riesgo estable",
    "A2": "Riesgo moderado",
    "A3": "Riesgo moderado",
    "AL": "Riesgo moderado",
    "B1": "Riesgo moderado",
    "B2": "Riesgo moderado",
    "C1": "Riesgo moderado",
    "C2": "Riesgo moderado",
    "D": "Alto riesgo",
    "E": "Alto riesgo",
}

RISK_SCORE = {
    "Sin deuda": 0,
    "Riesgo estable": 1,
    "Riesgo moderado": 2,
    "Alto riesgo": 3,
}

RISK_LABEL_BY_SCORE = {value: key for key, value in RISK_SCORE.items()}

MODEL_NUMERIC_COLUMNS = [
    "estudiantes_vinculados",
    "edad_estudiante_prom",
    "pct_mujer_estudiantes",
    "salario_hogar_log",
    "deuda_hogar_log",
    "ratio_deuda_ingreso",
    "hijos_hogar",
    "edad_padres_prom",
    "padres_presentes",
    "padres_con_empleo",
    "padres_con_superior",
    "primera_generacion",
    "hogar_huerfano",
    "riesgo_deuda_hogar_score",
]

MODEL_CATEGORICAL_COLUMNS = ["estado_hogar"]


def _norm_id(series: pd.Series) -> pd.Series:
    return (
        pd.Series(series)
        .astype("string")
        .fillna("0")
        .astype(str)
        .str.strip()
        .replace({"": "0", "nan": "0", "None": "0", "<NA>": "0"})
    )


def _to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _parse_valor_deuda(series: pd.Series) -> pd.Series:
    s = pd.Series(series).astype("string").fillna("").astype(str).str.strip()
    has_comma = s.str.contains(",", regex=False)
    s = s.where(
        ~has_comma,
        s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
    )
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def _canonical_sex(value: Any) -> str:
    raw = str(value).strip().upper()
    if raw in {"HOMBRE", "MASCULINO", "M"}:
        return "HOMBRE"
    if raw in {"MUJER", "FEMENINO", "F"}:
        return "MUJER"
    return "DESCONOCIDO"


def _canonical_estado(value: Any) -> str:
    raw = str(value).strip().upper()
    if not raw or raw == "NAN":
        return "Desconocido"
    mapping = {
        "CASADO": "Casado",
        "SOLTERO": "Soltero",
        "DIVORCIADO": "Divorciado",
        "VIUDO": "Viudo",
        "EN UNION DE HEC": "Union libre",
        "EN UNION DE HECHO": "Union libre",
        "UNION LIBRE": "Union libre",
    }
    return mapping.get(raw, raw.title())


def _canonical_education(value: Any) -> str:
    raw = str(value).strip().upper()
    if not raw or raw == "NAN":
        return "SIN DATO"
    return raw


def _compute_age(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce", dayfirst=True)
    today = pd.Timestamp.today().normalize()
    ages = np.floor((today - dates).dt.days / 365.25)
    return pd.to_numeric(ages, errors="coerce")


def _mode_or_default(series: pd.Series, default: str = "Sin dato") -> str:
    clean = (
        pd.Series(series)
        .astype("string")
        .fillna("")
        .astype(str)
        .str.strip()
        .replace({"nan": "", "None": ""})
    )
    clean = clean[clean != ""]
    if clean.empty:
        return default
    return str(clean.mode().iloc[0])


def _stable_seed(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


@st.cache_data(show_spinner=False)
def load_propensity_sources() -> dict[str, dict[str, pd.DataFrame]]:
    sources: dict[str, dict[str, pd.DataFrame]] = {}
    files = {"UDLA": "Udla.xlsx", "Universidades": "Universidades.xlsx"}

    for source_name, filename in files.items():
        source_data: dict[str, pd.DataFrame] = {}
        for sheet_name in REQUIRED_SHEETS:
            source_data[sheet_name] = load_excel_sheet(sheet_name, filename).copy()
        sources[source_name] = source_data

    return sources


def _prepare_students(
    df: pd.DataFrame, default_university: str | None = None
) -> pd.DataFrame:
    students = df.copy()
    if "Cedula" in students.columns:
        students = students.rename(columns={"Cedula": "IDENTIFICACION"})
    elif "CEDULA" in students.columns:
        students = students.rename(columns={"CEDULA": "IDENTIFICACION"})

    students = normalize_university_column(students)
    if "Universidad" not in students.columns:
        students["Universidad"] = default_university or "Sin dato"

    students["IDENTIFICACION"] = _norm_id(students["IDENTIFICACION"])
    students["Universidad"] = (
        students["Universidad"]
        .fillna(default_university or "Sin dato")
        .astype(str)
        .str.strip()
    )
    students = students[students["IDENTIFICACION"] != "0"].copy()
    students = students.drop_duplicates(subset=["IDENTIFICACION"], keep="first")

    if "GENERO" in students.columns:
        students["GENERO_CANON"] = students["GENERO"].map(_canonical_sex)
    else:
        students["GENERO_CANON"] = "DESCONOCIDO"

    students["carrera"] = (
        students["CARRERA"].fillna("").astype(str).str.strip()
        if "CARRERA" in students.columns
        else ""
    )

    unidad_academica = pd.Series(
        [""] * len(students), index=students.index, dtype="object"
    )
    if "FACULTAD" in students.columns:
        unidad_academica = students["FACULTAD"].fillna("").astype(str).str.strip()
    if "AREA" in students.columns:
        area = students["AREA"].fillna("").astype(str).str.strip()
        unidad_academica = unidad_academica.where(unidad_academica != "", area)
    students["unidad_academica"] = unidad_academica.replace("", "Sin dato")

    students["tipo_estudiante"] = (
        students["TIPO"].fillna("").astype(str).str.strip().str.upper()
        if "TIPO" in students.columns
        else ""
    )

    return students[
        [
            "IDENTIFICACION",
            "Universidad",
            "GENERO_CANON",
            "carrera",
            "unidad_academica",
            "tipo_estudiante",
        ]
    ].copy()


def _prepare_familia(df: pd.DataFrame) -> pd.DataFrame:
    familia = df.copy()
    familia["IDENTIFICACION"] = _norm_id(familia["IDENTIFICACION"])
    for column in ["CED_PADRE", "CED_MADRE"]:
        if column in familia.columns:
            familia[column] = _norm_id(familia[column])
        else:
            familia[column] = "0"
    return familia.drop_duplicates(subset=["IDENTIFICACION"], keep="first")[
        ["IDENTIFICACION", "CED_PADRE", "CED_MADRE"]
    ].copy()


def _prepare_info(df: pd.DataFrame) -> pd.DataFrame:
    info = df.copy()
    info["IDENTIFICACION"] = _norm_id(info["IDENTIFICACION"])
    info = info[info["IDENTIFICACION"] != "0"].copy()
    info["FECHA_EXP"] = pd.to_datetime(
        info.get("FECHA EXPEDICION"), errors="coerce", dayfirst=True
    )
    info = info.sort_values(["IDENTIFICACION", "FECHA_EXP"])
    latest = info.groupby("IDENTIFICACION", as_index=False).tail(1).copy()

    latest["SEXO_CANON"] = latest.get("SEXO", pd.Series(dtype="object")).map(
        _canonical_sex
    )
    latest["ESTADO_CANON"] = latest.get(
        "ESTADO_CIVIL", pd.Series(dtype="object")
    ).map(_canonical_estado)
    latest["HIJOS_NUM"] = _to_numeric(latest.get("HIJOS", 0), default=0.0)
    latest["EDAD"] = _compute_age(
        latest.get("FECHA_NACIMIENTO", pd.Series(dtype="object"))
    )
    latest["LATITUD_NUM"] = _to_numeric(latest.get("LATITUD", np.nan), default=np.nan)
    latest["LONGITUD_NUM"] = _to_numeric(
        latest.get("LONGITUD", np.nan), default=np.nan
    )
    latest.loc[latest["LATITUD_NUM"].abs() > 90, "LATITUD_NUM"] = np.nan
    latest.loc[latest["LONGITUD_NUM"].abs() > 180, "LONGITUD_NUM"] = np.nan
    latest["NIVEL_CANON"] = latest.get(
        "NIVEL_ESTUDIO", pd.Series(dtype="object")
    ).map(_canonical_education)

    return latest[
        [
            "IDENTIFICACION",
            "SEXO_CANON",
            "ESTADO_CANON",
            "HIJOS_NUM",
            "EDAD",
            "LATITUD_NUM",
            "LONGITUD_NUM",
            "NIVEL_CANON",
            "FECHA_EXP",
        ]
    ].copy()


def _attach_parroquia_to_info(
    info_df: pd.DataFrame, gdf_parroquias: pd.DataFrame
) -> pd.DataFrame:
    if info_df.empty:
        out = info_df.copy()
        out["parroquia_estudiante"] = pd.Series(dtype="object")
        return out

    out = info_df.copy()
    assigned = asignar_parroquia(
        out[["IDENTIFICACION", "LATITUD_NUM", "LONGITUD_NUM"]].copy(),
        gdf_parroquias,
        "LATITUD_NUM",
        "LONGITUD_NUM",
    )
    if not assigned.empty and "parroquia" in assigned.columns:
        assigned = assigned[["IDENTIFICACION", "parroquia"]].drop_duplicates(
            subset=["IDENTIFICACION"], keep="first"
        )
        out = out.merge(assigned, on="IDENTIFICACION", how="left")
    else:
        out["parroquia"] = np.nan

    out["parroquia_estudiante"] = (
        out["parroquia"]
        .fillna("Sin dato")
        .astype(str)
        .str.strip()
        .replace("", "Sin dato")
    )
    return out.drop(columns=["parroquia"], errors="ignore")


def _prepare_empleo(df: pd.DataFrame) -> pd.DataFrame:
    empleo = df.copy()
    empleo["IDENTIFICACION"] = _norm_id(empleo["IDENTIFICACION"])
    empleo["SALARIO"] = _to_numeric(empleo.get("SALARIO", 0.0), default=0.0)
    empleo["ANIO"] = _to_numeric(empleo.get("ANIO", np.nan), default=np.nan)
    empleo["MES"] = _to_numeric(empleo.get("MES", np.nan), default=np.nan)
    empleo = empleo.dropna(subset=["ANIO", "MES"]).copy()
    if empleo.empty:
        return pd.DataFrame(
            columns=["IDENTIFICACION", "salario", "empleadores", "empleo_formal"]
        )

    empleo["PERIODO"] = empleo["ANIO"].astype(int) * 100 + empleo["MES"].astype(int)
    latest_period = empleo.groupby("IDENTIFICACION")["PERIODO"].transform("max")
    empleo = empleo[empleo["PERIODO"] == latest_period].copy()

    if "RUC_EMPLEADOR" in empleo.columns:
        empleadores = (
            empleo.groupby("IDENTIFICACION")["RUC_EMPLEADOR"]
            .nunique()
            .reset_index(name="empleadores")
        )
    else:
        empleadores = pd.DataFrame(
            {
                "IDENTIFICACION": empleo["IDENTIFICACION"].unique().tolist(),
                "empleadores": 0,
            }
        )

    salary = (
        empleo.groupby("IDENTIFICACION", as_index=False)["SALARIO"]
        .sum()
        .rename(columns={"SALARIO": "salario"})
    )
    salary = salary.merge(empleadores, on="IDENTIFICACION", how="left")
    salary["empleadores"] = salary["empleadores"].fillna(0).astype(int)
    salary["empleo_formal"] = (salary["salario"] > 0).astype(int)
    return salary


def _prepare_deuda(df: pd.DataFrame) -> pd.DataFrame:
    deudas = df.copy()
    deudas["IDENTIFICACION"] = _norm_id(deudas["IDENTIFICACION"])
    deudas["VALOR_NUM"] = _parse_valor_deuda(deudas.get("VALOR", 0.0))
    deudas["ANIO"] = _to_numeric(deudas.get("ANIO", np.nan), default=np.nan)
    deudas["MES"] = _to_numeric(deudas.get("MES", np.nan), default=np.nan)
    deudas = deudas.dropna(subset=["ANIO", "MES"]).copy()

    if deudas.empty:
        return pd.DataFrame(
            columns=[
                "IDENTIFICACION",
                "deuda_total",
                "entidades_deuda",
                "riesgo_deuda",
                "riesgo_deuda_score",
            ]
        )

    deudas["PERIODO"] = deudas["ANIO"].astype(int) * 100 + deudas["MES"].astype(int)
    latest_period = deudas.groupby("IDENTIFICACION")["PERIODO"].transform("max")
    deudas = deudas[deudas["PERIODO"] == latest_period].copy()

    debt_total = (
        deudas.groupby("IDENTIFICACION", as_index=False)["VALOR_NUM"]
        .sum()
        .rename(columns={"VALOR_NUM": "deuda_total"})
    )

    if "COD_ENTIDAD" in deudas.columns:
        entidades = (
            deudas.groupby("IDENTIFICACION")["COD_ENTIDAD"]
            .nunique()
            .reset_index(name="entidades_deuda")
        )
    else:
        entidades = pd.DataFrame(
            {
                "IDENTIFICACION": debt_total["IDENTIFICACION"],
                "entidades_deuda": 0,
            }
        )

    deudas["RISK_LABEL"] = (
        deudas.get("COD_CALIFICACION", pd.Series(dtype="object"))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .map(RISK_MAP)
        .fillna("Sin deuda")
    )
    deudas["RISK_SCORE"] = deudas["RISK_LABEL"].map(RISK_SCORE).fillna(0).astype(int)

    risk_rank = (
        deudas.groupby(["IDENTIFICACION", "RISK_LABEL", "RISK_SCORE"], as_index=False)[
            "VALOR_NUM"
        ]
        .sum()
        .sort_values(
            ["IDENTIFICACION", "VALOR_NUM", "RISK_SCORE"],
            ascending=[True, False, False],
        )
        .drop_duplicates("IDENTIFICACION", keep="first")
        .rename(
            columns={
                "RISK_LABEL": "riesgo_deuda",
                "RISK_SCORE": "riesgo_deuda_score",
            }
        )
    )

    result = debt_total.merge(entidades, on="IDENTIFICACION", how="left").merge(
        risk_rank[["IDENTIFICACION", "riesgo_deuda", "riesgo_deuda_score"]],
        on="IDENTIFICACION",
        how="left",
    )
    result["entidades_deuda"] = result["entidades_deuda"].fillna(0).astype(int)
    result["riesgo_deuda"] = result["riesgo_deuda"].fillna("Sin deuda")
    result["riesgo_deuda_score"] = result["riesgo_deuda_score"].fillna(0).astype(int)
    return result


def _household_id(df: pd.DataFrame) -> pd.Series:
    def _build(row: pd.Series) -> str:
        ids = []
        for column in ["CED_PADRE", "CED_MADRE"]:
            value = str(row.get(column, "0")).strip()
            if value and value != "0":
                ids.append(value)
        if not ids:
            return f"EST-{row['IDENTIFICACION']}"
        return "|".join(sorted(ids))

    return df.apply(_build, axis=1)


def _choose_household_state(df: pd.DataFrame) -> pd.Series:
    padre_estado = df["padre_ESTADO_CANON"].fillna("Desconocido")
    madre_estado = df["madre_ESTADO_CANON"].fillna("Desconocido")
    padre_fecha = df["padre_FECHA_EXP"]
    madre_fecha = df["madre_FECHA_EXP"]

    padre_dt = padre_fecha.fillna(pd.Timestamp.min)
    madre_dt = madre_fecha.fillna(pd.Timestamp.min)

    estado = np.where(
        (padre_estado != "Desconocido") & (padre_dt >= madre_dt),
        padre_estado,
        np.where(
            madre_estado != "Desconocido",
            madre_estado,
            np.where(padre_estado != "Desconocido", padre_estado, "Desconocido"),
        ),
    )
    return pd.Series(estado, index=df.index)


@st.cache_data(show_spinner=False)
def build_student_feature_base() -> tuple[pd.DataFrame, dict[str, dict[int, dict[str, float]]]]:
    sources = load_propensity_sources()
    frames: list[pd.DataFrame] = []
    institution_ranges: dict[str, dict[int, dict[str, float]]] = {}
    gdf_parroquias = cargar_parroquias()

    for source_name, data in sources.items():
        students = _prepare_students(
            data["Estudiantes"],
            default_university="UDLA" if source_name == "UDLA" else None,
        )
        familia = _prepare_familia(data["Universo Familiares"])
        info = _attach_parroquia_to_info(
            _prepare_info(data["Informacion Personal"]), gdf_parroquias
        )
        empleo = _prepare_empleo(data["Empleos"])
        deuda = _prepare_deuda(data["Deudas"])

        base = students.merge(familia, on="IDENTIFICACION", how="left")
        base["CED_PADRE"] = base["CED_PADRE"].fillna("0")
        base["CED_MADRE"] = base["CED_MADRE"].fillna("0")
        base["hogar_id"] = _household_id(base)

        info_student = info.rename(
            columns={
                "SEXO_CANON": "est_SEXO_CANON",
                "ESTADO_CANON": "est_ESTADO_CANON",
                "HIJOS_NUM": "est_HIJOS_NUM",
                "EDAD": "est_EDAD",
                "LATITUD_NUM": "est_LATITUD_NUM",
                "LONGITUD_NUM": "est_LONGITUD_NUM",
                "parroquia_estudiante": "est_parroquia_estudiante",
                "NIVEL_CANON": "est_NIVEL_CANON",
                "FECHA_EXP": "est_FECHA_EXP",
            }
        )
        base = base.merge(info_student, on="IDENTIFICACION", how="left")

        padre_info = info.rename(
            columns={
                "IDENTIFICACION": "CED_PADRE",
                "SEXO_CANON": "padre_SEXO_CANON",
                "ESTADO_CANON": "padre_ESTADO_CANON",
                "HIJOS_NUM": "padre_HIJOS_NUM",
                "EDAD": "padre_EDAD",
                "NIVEL_CANON": "padre_NIVEL_CANON",
                "FECHA_EXP": "padre_FECHA_EXP",
            }
        )
        madre_info = info.rename(
            columns={
                "IDENTIFICACION": "CED_MADRE",
                "SEXO_CANON": "madre_SEXO_CANON",
                "ESTADO_CANON": "madre_ESTADO_CANON",
                "HIJOS_NUM": "madre_HIJOS_NUM",
                "EDAD": "madre_EDAD",
                "NIVEL_CANON": "madre_NIVEL_CANON",
                "FECHA_EXP": "madre_FECHA_EXP",
            }
        )
        base = base.merge(padre_info, on="CED_PADRE", how="left").merge(
            madre_info, on="CED_MADRE", how="left"
        )

        padre_emp = empleo.rename(
            columns={
                "IDENTIFICACION": "CED_PADRE",
                "salario": "salario_padre",
                "empleadores": "empleadores_padre",
                "empleo_formal": "empleo_formal_padre",
            }
        )
        madre_emp = empleo.rename(
            columns={
                "IDENTIFICACION": "CED_MADRE",
                "salario": "salario_madre",
                "empleadores": "empleadores_madre",
                "empleo_formal": "empleo_formal_madre",
            }
        )
        base = base.merge(padre_emp, on="CED_PADRE", how="left").merge(
            madre_emp, on="CED_MADRE", how="left"
        )

        padre_deu = deuda.rename(
            columns={
                "IDENTIFICACION": "CED_PADRE",
                "deuda_total": "deuda_padre",
                "entidades_deuda": "entidades_deuda_padre",
                "riesgo_deuda": "riesgo_deuda_padre",
                "riesgo_deuda_score": "riesgo_deuda_score_padre",
            }
        )
        madre_deu = deuda.rename(
            columns={
                "IDENTIFICACION": "CED_MADRE",
                "deuda_total": "deuda_madre",
                "entidades_deuda": "entidades_deuda_madre",
                "riesgo_deuda": "riesgo_deuda_madre",
                "riesgo_deuda_score": "riesgo_deuda_score_madre",
            }
        )
        base = base.merge(padre_deu, on="CED_PADRE", how="left").merge(
            madre_deu, on="CED_MADRE", how="left"
        )

        numeric_fill_cols = [
            "salario_padre",
            "salario_madre",
            "deuda_padre",
            "deuda_madre",
            "entidades_deuda_padre",
            "entidades_deuda_madre",
            "empleadores_padre",
            "empleadores_madre",
            "empleo_formal_padre",
            "empleo_formal_madre",
            "riesgo_deuda_score_padre",
            "riesgo_deuda_score_madre",
            "padre_HIJOS_NUM",
            "madre_HIJOS_NUM",
        ]
        for column in numeric_fill_cols:
            base[column] = _to_numeric(base.get(column, 0.0), default=0.0)

        base["sexo_estudiante"] = base["est_SEXO_CANON"].where(
            base["est_SEXO_CANON"].notna() & (base["est_SEXO_CANON"] != ""),
            base["GENERO_CANON"],
        )
        base["sexo_estudiante"] = base["sexo_estudiante"].map(_canonical_sex)
        base["edad_estudiante"] = _to_numeric(base["est_EDAD"], default=np.nan)
        base["latitud_estudiante"] = _to_numeric(
            base.get("est_LATITUD_NUM", np.nan), default=np.nan
        )
        base["longitud_estudiante"] = _to_numeric(
            base.get("est_LONGITUD_NUM", np.nan), default=np.nan
        )
        base["parroquia_estudiante"] = (
            base.get("est_parroquia_estudiante", "Sin dato")
            .fillna("Sin dato")
            .astype(str)
            .str.strip()
            .replace("", "Sin dato")
        )
        base["tiene_ubicacion"] = base["parroquia_estudiante"].ne("Sin dato").astype(int)
        base["salario_hogar"] = base["salario_padre"] + base["salario_madre"]
        base["deuda_hogar"] = base["deuda_padre"] + base["deuda_madre"]
        base["salario_hogar_log"] = np.log1p(base["salario_hogar"].clip(lower=0))
        base["deuda_hogar_log"] = np.log1p(base["deuda_hogar"].clip(lower=0))
        base["entidades_deuda_hogar"] = (
            base["entidades_deuda_padre"] + base["entidades_deuda_madre"]
        )
        base["hijos_hogar"] = base[["padre_HIJOS_NUM", "madre_HIJOS_NUM"]].max(axis=1)
        base["edad_padres_prom"] = base[["padre_EDAD", "madre_EDAD"]].mean(axis=1)
        base["padres_presentes"] = (
            (base["CED_PADRE"] != "0").astype(int)
            + (base["CED_MADRE"] != "0").astype(int)
        )
        base["hogar_huerfano"] = (base["padres_presentes"] == 0).astype(int)
        base["padres_con_empleo"] = (
            (base["salario_padre"] > 0).astype(int)
            + (base["salario_madre"] > 0).astype(int)
        )
        base["padres_con_superior"] = (
            (base["padre_NIVEL_CANON"] == "SUPERIOR").astype(int)
            + (base["madre_NIVEL_CANON"] == "SUPERIOR").astype(int)
        )
        base["primera_generacion"] = (
            (base["padres_presentes"] > 0) & (base["padres_con_superior"] == 0)
        ).astype(int)
        base["estado_hogar"] = _choose_household_state(base)
        base["riesgo_deuda_hogar_score"] = (
            base[["riesgo_deuda_score_padre", "riesgo_deuda_score_madre"]]
            .max(axis=1)
            .fillna(0)
        ).astype(int)
        base["riesgo_deuda_hogar"] = base["riesgo_deuda_hogar_score"].map(
            RISK_LABEL_BY_SCORE
        )
        base["sin_empleo_formal"] = (base["salario_hogar"] <= 0).astype(int)
        base["hogar_con_deuda"] = (base["deuda_hogar"] > 0).astype(int)
        base["ratio_deuda_ingreso"] = np.where(
            base["salario_hogar"] > 0,
            base["deuda_hogar"] / (base["salario_hogar"] * 14.0),
            np.where(base["deuda_hogar"] > 0, 5.0, 0.0),
        )
        base["ratio_deuda_ingreso"] = (
            base["ratio_deuda_ingreso"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(0, 10)
        )

        cond_sin_empleo = (
            (
                (base["CED_PADRE"] != "0")
                & (base["CED_MADRE"] != "0")
                & (base["padres_con_empleo"] == 0)
            )
            | (
                (base["CED_PADRE"] != "0")
                & (base["CED_MADRE"] == "0")
                & (base["salario_padre"] <= 0)
            )
            | (
                (base["CED_PADRE"] == "0")
                & (base["CED_MADRE"] != "0")
                & (base["salario_madre"] <= 0)
            )
        )
        cond_deuda_critica = (
            (base["riesgo_deuda_hogar_score"] >= 3)
            & (base["salario_hogar"] > 0)
            & (base["ratio_deuda_ingreso"] >= 2.9)
        )
        vuln_score = cond_sin_empleo.astype(int) + cond_deuda_critica.astype(int)
        base["en_riesgo"] = (vuln_score == 1).astype(int)
        base["fuente_archivo"] = source_name

        frames.append(
            base[
                [
                    "IDENTIFICACION",
                    "Universidad",
                    "fuente_archivo",
                    "hogar_id",
                    "carrera",
                    "unidad_academica",
                    "sexo_estudiante",
                    "edad_estudiante",
                    "latitud_estudiante",
                    "longitud_estudiante",
                    "parroquia_estudiante",
                    "tiene_ubicacion",
                    "salario_hogar",
                    "salario_hogar_log",
                    "deuda_hogar",
                    "deuda_hogar_log",
                    "entidades_deuda_hogar",
                    "hogar_con_deuda",
                    "ratio_deuda_ingreso",
                    "hijos_hogar",
                    "edad_padres_prom",
                    "padres_presentes",
                    "hogar_huerfano",
                    "padres_con_empleo",
                    "padres_con_superior",
                    "primera_generacion",
                    "estado_hogar",
                    "riesgo_deuda_hogar",
                    "riesgo_deuda_hogar_score",
                    "sin_empleo_formal",
                    "en_riesgo",
                    "tipo_estudiante",
                ]
            ].copy()
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["Universidad"] = combined["Universidad"].fillna("Sin dato").astype(str).str.strip()
    combined["quintil_institucion"] = "Sin empleo"
    combined["quintil_num"] = 0

    for universidad, df_uni in combined.groupby("Universidad"):
        rangos = calcular_rangos_quintiles(df_uni["salario_hogar"])
        institution_ranges[universidad] = rangos
        mask = combined["Universidad"] == universidad
        quintiles = combined.loc[mask, "salario_hogar"].apply(
            lambda value: asignar_quintil_por_rangos(
                value, rangos, vacio="Sin empleo"
            )
        )
        combined.loc[mask, "quintil_institucion"] = quintiles
        combined.loc[mask, "quintil_num"] = (
            quintiles.replace({"Sin empleo": "0"}).astype(int)
        )

    return combined, institution_ranges


def _join_unique_values(series: pd.Series, max_items: int = 8) -> str:
    clean = (
        pd.Series(series)
        .fillna("")
        .astype(str)
        .str.strip()
        .replace({"nan": "", "None": ""})
    )
    values = [value for value in clean.unique().tolist() if value != ""]
    if not values:
        return "Sin dato"
    if len(values) > max_items:
        values = values[:max_items] + ["..."]
    return " | ".join(values)


@st.cache_data(show_spinner=False)
def build_household_feature_base(
    tipo_filtro: str = "Todas",
) -> tuple[pd.DataFrame, dict[str, dict[int, dict[str, float]]]]:
    student_base, institution_ranges = build_student_feature_base()

    household = (
        student_base.groupby(["Universidad", "fuente_archivo", "hogar_id"], as_index=False)
        .agg(
            estudiantes_vinculados=("IDENTIFICACION", "nunique"),
            estudiantes_ids=("IDENTIFICACION", lambda s: _join_unique_values(s, max_items=12)),
            carrera=("carrera", _mode_or_default),
            carreras_hogar=("carrera", lambda s: _join_unique_values(s, max_items=8)),
            carreras_distintas=("carrera", lambda s: int(pd.Series(s).fillna("").astype(str).str.strip().replace("", pd.NA).dropna().nunique())),
            unidad_academica=("unidad_academica", _mode_or_default),
            unidades_hogar=("unidad_academica", lambda s: _join_unique_values(s, max_items=8)),
            sexo_estudiante=("sexo_estudiante", _mode_or_default),
            pct_mujer_estudiantes=("sexo_estudiante", lambda s: float((pd.Series(s).fillna("").astype(str).str.upper() == "MUJER").mean())),
            edad_estudiante_prom=("edad_estudiante", "mean"),
            latitud_estudiante=("latitud_estudiante", "mean"),
            longitud_estudiante=("longitud_estudiante", "mean"),
            parroquia_estudiante=("parroquia_estudiante", _mode_or_default),
            parroquias_hogar=("parroquia_estudiante", lambda s: _join_unique_values(s, max_items=8)),
            tiene_ubicacion=("tiene_ubicacion", "max"),
            salario_hogar=("salario_hogar", "max"),
            salario_hogar_log=("salario_hogar_log", "max"),
            deuda_hogar=("deuda_hogar", "max"),
            deuda_hogar_log=("deuda_hogar_log", "max"),
            entidades_deuda_hogar=("entidades_deuda_hogar", "max"),
            hogar_con_deuda=("hogar_con_deuda", "max"),
            ratio_deuda_ingreso=("ratio_deuda_ingreso", "max"),
            hijos_hogar=("hijos_hogar", "max"),
            edad_padres_prom=("edad_padres_prom", "mean"),
            padres_presentes=("padres_presentes", "max"),
            hogar_huerfano=("hogar_huerfano", "max"),
            padres_con_empleo=("padres_con_empleo", "max"),
            padres_con_superior=("padres_con_superior", "max"),
            primera_generacion=("primera_generacion", "max"),
            estado_hogar=("estado_hogar", _mode_or_default),
            riesgo_deuda_hogar=("riesgo_deuda_hogar", _mode_or_default),
            riesgo_deuda_hogar_score=("riesgo_deuda_hogar_score", "max"),
            sin_empleo_formal=("sin_empleo_formal", "max"),
            en_riesgo=("en_riesgo", "max"),
            quintil_institucion=("quintil_institucion", _mode_or_default),
            quintil_num=("quintil_num", "max"),
            tipo_estudiante=("tipo_estudiante", _mode_or_default),
        )
        .copy()
    )

    household["tiene_ubicacion"] = household["tiene_ubicacion"].fillna(0).astype(int)
    household["estudiantes_vinculados"] = (
        household["estudiantes_vinculados"].fillna(0).astype(int)
    )
    household["carreras_distintas"] = household["carreras_distintas"].fillna(0).astype(int)
    household["parroquia_estudiante"] = (
        household["parroquia_estudiante"]
        .fillna("Sin dato")
        .astype(str)
        .str.strip()
        .replace("", "Sin dato")
    )

    if tipo_filtro and tipo_filtro != "Todas":
        household = household[
            (household["Universidad"] != "UDLA")
            | (household["tipo_estudiante"] == tipo_filtro)
        ].copy()

    return household, institution_ranges


def _build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    numeric = frame[MODEL_NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.median()).fillna(0.0)

    categoricals = (
        frame[MODEL_CATEGORICAL_COLUMNS]
        .fillna("Desconocido")
        .astype(str)
        .apply(lambda col: col.str.strip().replace("", "Desconocido"))
    )
    dummies = pd.get_dummies(
        categoricals, prefix=MODEL_CATEGORICAL_COLUMNS, dtype=float
    )
    return pd.concat([numeric, dummies], axis=1)


def _fit_logistic_propensity(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    l2_penalty: float = 1.0,
) -> np.ndarray:
    x_design = np.column_stack([np.ones(X.shape[0]), X])
    init = np.zeros(x_design.shape[1], dtype=float)

    def _loss_and_grad(beta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = x_design @ beta
        probs = expit(logits)
        eps = 1e-9
        loss = -np.sum(
            sample_weight
            * (y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps))
        )
        loss += 0.5 * l2_penalty * np.sum(beta[1:] ** 2)

        grad = x_design.T @ ((probs - y) * sample_weight)
        grad[1:] += l2_penalty * beta[1:]
        return float(loss), grad

    result = minimize(
        fun=lambda beta: _loss_and_grad(beta)[0],
        x0=init,
        jac=lambda beta: _loss_and_grad(beta)[1],
        method="L-BFGS-B",
        options={"maxiter": 300},
    )
    if not result.success:
        return init
    return np.asarray(result.x, dtype=float)


def _scale_cluster_matrix(feature_df: pd.DataFrame) -> np.ndarray:
    if feature_df.empty:
        return np.empty((0, 0), dtype=float)

    matrix = feature_df.to_numpy(dtype=float)
    mean = np.nanmean(matrix, axis=0)
    matrix = np.where(np.isnan(matrix), mean, matrix)
    std = np.nanstd(matrix, axis=0)
    std = np.where(std == 0, 1.0, std)
    scaled = (matrix - mean) / std

    # Down-weight dummy groups so each categorical variable contributes the
    # same total variance as a single numeric feature.  Without this, a
    # one-hot group of N dummies has N× the influence of one numeric column.
    columns = feature_df.columns.tolist()
    for cat_col in MODEL_CATEGORICAL_COLUMNS:
        prefix = f"{cat_col}_"
        dummy_indices = [i for i, c in enumerate(columns) if c.startswith(prefix)]
        n_dummies = len(dummy_indices)
        if n_dummies > 1:
            scaled[:, dummy_indices] /= np.sqrt(n_dummies)

    return scaled


def _run_kmeans_labels(
    scaled: np.ndarray, institution_name: str, requested_k: int
) -> np.ndarray:
    n_rows = scaled.shape[0]
    k = max(1, min(int(requested_k), n_rows))
    if k == 1:
        return np.ones(n_rows, dtype=int)

    seed = _stable_seed(f"{institution_name}-{requested_k}")
    try:
        _centroids, labels = kmeans2(scaled, k, minit="points", iter=50, seed=seed)
        labels = np.asarray(labels, dtype=int)
    except Exception:
        order = np.argsort(np.nan_to_num(scaled[:, 0], nan=0.0))
        labels = np.zeros(n_rows, dtype=int)
        for idx, chunk in enumerate(np.array_split(order, k)):
            labels[chunk] = idx

    return labels + 1


def _minimum_cluster_size(n_rows: int) -> int:
    if n_rows <= 8:
        return 1
    if n_rows <= 20:
        return 2
    if n_rows <= 60:
        return 3
    return max(4, int(np.floor(n_rows * 0.04)))


def _candidate_cluster_counts(
    n_rows: int, min_clusters: int = 2, max_clusters: int = 6
) -> list[int]:
    if n_rows <= 3:
        return [1]

    upper_bound = min(int(max_clusters), n_rows)
    if n_rows < 12:
        upper_bound = min(upper_bound, max(2, n_rows // 2))
    elif n_rows < 30:
        upper_bound = min(upper_bound, 4)

    lower_bound = min(int(min_clusters), upper_bound)
    if upper_bound < 2:
        return [1]
    return list(range(lower_bound, upper_bound + 1))


def _calinski_harabasz_score(matrix: np.ndarray, labels: np.ndarray) -> float:
    if matrix.size == 0 or len(labels) <= 2:
        return float("-inf")

    unique_labels = np.unique(labels)
    k = len(unique_labels)
    n_rows = matrix.shape[0]
    if k <= 1 or n_rows <= k:
        return float("-inf")

    overall_mean = matrix.mean(axis=0)
    within_dispersion = 0.0
    between_dispersion = 0.0

    for label in unique_labels:
        points = matrix[labels == label]
        if len(points) == 0:
            return float("-inf")
        centroid = points.mean(axis=0)
        within_dispersion += float(((points - centroid) ** 2).sum())
        between_dispersion += float(len(points) * ((centroid - overall_mean) ** 2).sum())

    if within_dispersion <= 1e-12:
        return float("inf") if between_dispersion > 0 else float("-inf")

    numerator = between_dispersion / max(k - 1, 1)
    denominator = within_dispersion / max(n_rows - k, 1)
    return float(numerator / max(denominator, 1e-12))


def _assign_clusters(
    feature_df: pd.DataFrame,
    institution_name: str,
    min_clusters: int = 2,
    max_clusters: int = 6,
) -> tuple[np.ndarray, int]:
    if feature_df.empty:
        return np.array([], dtype=int), 0

    scaled = _scale_cluster_matrix(feature_df)
    n_rows = scaled.shape[0]
    candidates = _candidate_cluster_counts(
        n_rows, min_clusters=min_clusters, max_clusters=max_clusters
    )
    if candidates == [1]:
        return np.ones(n_rows, dtype=int), 1

    min_size = _minimum_cluster_size(n_rows)
    evaluations: list[dict[str, Any]] = []
    for k in candidates:
        labels = _run_kmeans_labels(scaled, institution_name, k)
        unique_labels, counts = np.unique(labels, return_counts=True)
        if len(unique_labels) <= 1:
            continue

        score = _calinski_harabasz_score(scaled, labels)
        evaluations.append(
            {
                "k": int(k),
                "labels": labels,
                "score": score,
                "is_valid": len(unique_labels) == k and int(counts.min()) >= min_size,
            }
        )

    if not evaluations:
        labels = _run_kmeans_labels(scaled, institution_name, 1)
        return labels, 1

    valid_evaluations = [
        item for item in evaluations if item["is_valid"] and np.isfinite(item["score"])
    ]
    candidate_pool = valid_evaluations or [
        item for item in evaluations if np.isfinite(item["score"])
    ]
    if not candidate_pool:
        fallback = min(
            max(2, int(round(np.sqrt(max(n_rows, 1) / 2.0)))),
            max(candidates),
        )
        labels = _run_kmeans_labels(scaled, institution_name, fallback)
        return labels, int(np.unique(labels).size)

    best_score = max(float(item["score"]) for item in candidate_pool)
    close_candidates = [
        item for item in candidate_pool if float(item["score"]) >= best_score * 0.97
    ]
    chosen = min(close_candidates, key=lambda item: (int(item["k"]), -float(item["score"])))
    return np.asarray(chosen["labels"], dtype=int), int(chosen["k"])


def _cluster_summary(df: pd.DataFrame, institution_name: str) -> pd.DataFrame:
    subset = df[df["Universidad"] == institution_name].copy()
    if subset.empty:
        return pd.DataFrame()

    grouped = subset.groupby("cluster_id", as_index=False).agg(
        hogares=("hogar_id", "nunique"),
        estudiantes_vinculados=("estudiantes_vinculados", "sum"),
        propensity_promedio=("propensity_udla", "mean"),
        propensity_mediana=("propensity_udla", "median"),
        salario_promedio=("salario_hogar", "mean"),
        deuda_promedio=("deuda_hogar", "mean"),
        hijos_promedio=("hijos_hogar", "mean"),
        edad_padres_promedio=("edad_padres_prom", "mean"),
        primera_generacion_pct=("primera_generacion", "mean"),
        hogar_huerfano_pct=("hogar_huerfano", "mean"),
        riesgo_pct=("en_riesgo", "mean"),
        con_deuda_pct=("hogar_con_deuda", "mean"),
        con_empleo_formal_pct=("padres_con_empleo", lambda s: float((s > 0).mean())),
        con_parroquia_pct=(
            "parroquia_estudiante",
            lambda s: float(
                pd.Series(s)
                .fillna("Sin dato")
                .astype(str)
                .str.strip()
                .replace("", "Sin dato")
                .ne("Sin dato")
                .mean()
            ),
        ),
    )

    modes = subset.groupby("cluster_id").agg(
        estado_modal=("estado_hogar", _mode_or_default),
        quintil_modal=("quintil_institucion", _mode_or_default),
        carrera_modal=("carrera", _mode_or_default),
        parroquia_modal=("parroquia_estudiante", _mode_or_default),
    )
    grouped = grouped.merge(modes, on="cluster_id", how="left")
    orphan_rows = grouped[grouped["cluster_id"] == 0]
    real_rows = grouped[grouped["cluster_id"] != 0].sort_values(
        ["propensity_promedio", "salario_promedio"],
        ascending=[False, False],
    ).reset_index(drop=True)

    real_rows["cluster_id_original"] = real_rows["cluster_id"].astype(int)
    cluster_labels = [f"Cluster {idx}" for idx in range(1, len(real_rows) + 1)]
    cluster_map = dict(zip(real_rows["cluster_id_original"], cluster_labels))
    real_rows["cluster_id"] = real_rows["cluster_id"].map(cluster_map)

    if not orphan_rows.empty:
        orphan_rows = orphan_rows.copy()
        orphan_rows["cluster_id_original"] = 0
        orphan_rows["cluster_id"] = "Sin dato familiar"
        grouped = pd.concat([real_rows, orphan_rows], ignore_index=True)
    else:
        grouped = real_rows

    grouped = grouped.rename(columns={"cluster_id": "cluster"})

    pct_cols = [
        "propensity_promedio",
        "propensity_mediana",
        "primera_generacion_pct",
        "hogar_huerfano_pct",
        "riesgo_pct",
        "con_deuda_pct",
        "con_empleo_formal_pct",
        "con_parroquia_pct",
    ]
    for column in pct_cols:
        grouped[column] = grouped[column] * 100.0

    return grouped


def _attach_pretty_cluster_labels(
    df: pd.DataFrame, summary: pd.DataFrame, institution_name: str
) -> pd.DataFrame:
    if summary.empty:
        return df

    if "cluster_id_original" not in summary.columns:
        return df

    label_map = dict(zip(summary["cluster_id_original"], summary["cluster"]))
    mask = df["Universidad"] == institution_name
    df.loc[mask, "cluster"] = df.loc[mask, "cluster_id"].map(label_map)
    return df


def _cluster_matches(
    full_df: pd.DataFrame,
    feature_scaled: pd.DataFrame,
    university_name: str,
    udla_summary: pd.DataFrame,
    uni_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if udla_summary.empty or uni_summary.empty:
        return uni_summary, udla_summary

    combined = full_df[["Universidad", "cluster"]].join(feature_scaled)
    udla_centroids = (
        combined[combined["Universidad"] == "UDLA"]
        .groupby("cluster", as_index=True)[feature_scaled.columns]
        .mean()
    )
    uni_centroids = (
        combined[combined["Universidad"] == university_name]
        .groupby("cluster", as_index=True)[feature_scaled.columns]
        .mean()
    )

    if udla_centroids.empty or uni_centroids.empty:
        return uni_summary, udla_summary

    nearest_cluster: list[str] = []
    similarity_score: list[float] = []
    for cluster_name, row in uni_centroids.iterrows():
        diffs = udla_centroids.subtract(row, axis=1)
        distances = np.sqrt((diffs**2).sum(axis=1))
        target = distances.sort_values().index[0]
        distance = float(distances.loc[target])
        nearest_cluster.append(str(target))
        similarity_score.append(100.0 / (1.0 + distance))

    match_df = pd.DataFrame(
        {
            "cluster": uni_centroids.index.astype(str),
            "cluster_udla_cercano": nearest_cluster,
            "similitud_centroidal": similarity_score,
        }
    )
    uni_summary = uni_summary.merge(match_df, on="cluster", how="left")
    return uni_summary, udla_summary


def _support_interval(series: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return 0.0, 0.0
    return float(clean.quantile(0.05)), float(clean.quantile(0.95))


@st.cache_data(show_spinner=False)
def run_propensity_analysis(
    min_clusters: int = 2, max_clusters: int = 6, tipo_filtro: str = "Todas"
) -> dict[str, Any]:
    base_df, ranges = build_household_feature_base(tipo_filtro=tipo_filtro)
    universities = sorted(
        [value for value in base_df["Universidad"].unique().tolist() if value != "UDLA"]
    )

    overview_rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}

    for university_name in universities:
        comparison = base_df[
            base_df["Universidad"].isin(["UDLA", university_name])
        ].copy()
        comparison["es_udla"] = (comparison["Universidad"] == "UDLA").astype(int)
        cluster_counts: dict[str, int] = {}

        feature_df = _build_feature_frame(comparison)
        feature_cols = feature_df.columns.tolist()
        matrix = feature_df.to_numpy(dtype=float)
        mean = np.nanmean(matrix, axis=0)
        matrix = np.where(np.isnan(matrix), mean, matrix)
        std = np.nanstd(matrix, axis=0)
        std = np.where(std == 0, 1.0, std)
        scaled = (matrix - mean) / std

        for cat_col in MODEL_CATEGORICAL_COLUMNS:
            prefix = f"{cat_col}_"
            dummy_indices = [i for i, c in enumerate(feature_cols) if c.startswith(prefix)]
            n_dummies = len(dummy_indices)
            if n_dummies > 1:
                scaled[:, dummy_indices] /= np.sqrt(n_dummies)

        scaled_df = pd.DataFrame(scaled, columns=feature_cols, index=comparison.index)

        y = comparison["es_udla"].to_numpy(dtype=float)
        y_sum = max(float(y.sum()), 1.0)
        y_zero = max(float(len(y) - y.sum()), 1.0)
        weights = np.where(y == 1.0, len(y) / (2.0 * y_sum), len(y) / (2.0 * y_zero))

        beta = _fit_logistic_propensity(scaled, y, weights)
        probs = expit(np.column_stack([np.ones(len(scaled)), scaled]) @ beta)
        comparison["propensity_udla"] = probs

        for institution_name in ["UDLA", university_name]:
            mask = comparison["Universidad"] == institution_name
            orphan_mask = mask & (comparison.get("hogar_huerfano", 0) == 1)
            cluster_mask = mask & ~orphan_mask

            if int(cluster_mask.sum()) > 0:
                labels, chosen_k = _assign_clusters(
                    feature_df.loc[cluster_mask],
                    institution_name=institution_name,
                    min_clusters=min_clusters,
                    max_clusters=max_clusters,
                )
                comparison.loc[cluster_mask, "cluster_id"] = labels
            else:
                chosen_k = 0

            if int(orphan_mask.sum()) > 0:
                comparison.loc[orphan_mask, "cluster_id"] = 0

            cluster_counts[institution_name] = int(chosen_k)

        comparison["cluster_id"] = comparison["cluster_id"].astype(int)
        udla_summary = _cluster_summary(comparison, "UDLA")
        uni_summary = _cluster_summary(comparison, university_name)

        comparison = _attach_pretty_cluster_labels(comparison, udla_summary, "UDLA")
        comparison = _attach_pretty_cluster_labels(comparison, uni_summary, university_name)

        uni_summary, udla_summary = _cluster_matches(
            comparison, scaled_df, university_name, udla_summary, uni_summary
        )

        udla_scores = comparison.loc[
            comparison["Universidad"] == "UDLA", "propensity_udla"
        ]
        uni_scores = comparison.loc[
            comparison["Universidad"] == university_name, "propensity_udla"
        ]

        low_support, high_support = _support_interval(udla_scores)
        overlap_pct = float(
            ((uni_scores >= low_support) & (uni_scores <= high_support)).mean()
            * 100.0
        )

        coverage = comparison[comparison["Universidad"] == university_name]
        household_count = int(coverage["hogar_id"].nunique())
        linked_students = int(coverage["estudiantes_vinculados"].sum())
        sample_status = "Exploratoria" if household_count < 30 else "Estable"
        overview_rows.append(
            {
                "Universidad": university_name,
                "Hogares": household_count,
                "Estudiantes vinculados": linked_students,
                "Clusters": int(cluster_counts.get(university_name, coverage["cluster"].nunique())),
                "Lectura": sample_status,
                "Propensity promedio hacia UDLA": float(uni_scores.mean() * 100.0),
                "Propensity mediana": float(uni_scores.median() * 100.0),
                "Solapamiento con UDLA": overlap_pct,
                "Primera generacion": float(
                    coverage["primera_generacion"].mean() * 100.0
                ),
                "Hogares huerfanos": float(
                    coverage["hogar_huerfano"].mean() * 100.0
                ),
                "Con deuda": float(coverage["hogar_con_deuda"].mean() * 100.0),
            }
        )

        details[university_name] = {
            "comparison": comparison,
            "feature_scaled": scaled_df,
            "university_clusters": uni_summary,
            "udla_clusters": udla_summary,
            "support_interval": (low_support, high_support),
            "feature_columns": feature_cols,
            "sample_status": sample_status,
            "cluster_counts": cluster_counts,
            "quintile_ranges": {
                university_name: ranges.get(university_name, {}),
                "UDLA": ranges.get("UDLA", {}),
            },
        }

    # ------------------------------------------------------------------
    # Standalone UDLA clustering (not pairwise — for the UDLA detail tab)
    # ------------------------------------------------------------------
    udla_all = base_df[base_df["Universidad"] == "UDLA"].copy()
    udla_all["propensity_udla"] = np.nan  # no propensity for UDLA vs itself

    udla_feature_df = _build_feature_frame(udla_all)
    udla_feature_cols = udla_feature_df.columns.tolist()
    udla_matrix = udla_feature_df.to_numpy(dtype=float)
    udla_mean = np.nanmean(udla_matrix, axis=0)
    udla_matrix = np.where(np.isnan(udla_matrix), udla_mean, udla_matrix)
    udla_std = np.nanstd(udla_matrix, axis=0)
    udla_std = np.where(udla_std == 0, 1.0, udla_std)
    udla_scaled = (udla_matrix - udla_mean) / udla_std
    udla_scaled_df = pd.DataFrame(
        udla_scaled, columns=udla_feature_cols, index=udla_all.index
    )

    orphan_mask_udla = udla_all["hogar_huerfano"] == 1
    cluster_mask_udla = ~orphan_mask_udla
    udla_cluster_counts: dict[str, int] = {}

    if int(cluster_mask_udla.sum()) > 0:
        udla_labels, udla_chosen_k = _assign_clusters(
            udla_feature_df.loc[cluster_mask_udla],
            institution_name="UDLA",
            min_clusters=min_clusters,
            max_clusters=max_clusters,
        )
        udla_all.loc[cluster_mask_udla, "cluster_id"] = udla_labels
    else:
        udla_chosen_k = 0

    if int(orphan_mask_udla.sum()) > 0:
        udla_all.loc[orphan_mask_udla, "cluster_id"] = 0

    udla_cluster_counts["UDLA"] = int(udla_chosen_k)
    udla_all["cluster_id"] = udla_all["cluster_id"].astype(int)
    udla_standalone_summary = _cluster_summary(udla_all, "UDLA")
    udla_all = _attach_pretty_cluster_labels(udla_all, udla_standalone_summary, "UDLA")

    n_udla_hogares = int(udla_all["hogar_id"].nunique())
    n_udla_estudiantes = int(udla_all["estudiantes_vinculados"].sum())

    details["UDLA"] = {
        "comparison": udla_all,
        "feature_scaled": udla_scaled_df,
        "university_clusters": udla_standalone_summary,
        "udla_clusters": pd.DataFrame(),
        "support_interval": (0.0, 0.0),
        "feature_columns": udla_feature_cols,
        "sample_status": "Estable",
        "cluster_counts": udla_cluster_counts,
        "quintile_ranges": {"UDLA": ranges.get("UDLA", {})},
        "is_udla_standalone": True,
    }

    overview_df = pd.DataFrame(overview_rows)
    if not overview_df.empty:
        overview_df["_orden_estabilidad"] = (
            overview_df["Lectura"].eq("Estable").astype(int)
        )
        overview_df = overview_df.sort_values(
            ["_orden_estabilidad", "Propensity promedio hacia UDLA", "Hogares"],
            ascending=[False, False, False],
        ).drop(columns="_orden_estabilidad").reset_index(drop=True)

    return {
        "base_df": base_df,
        "overview": overview_df,
        "details": details,
        "quintile_ranges": ranges,
    }
