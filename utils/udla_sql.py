from __future__ import annotations

import os
import urllib.parse

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine


SQL_SERVER = os.getenv("UDLA_SQL_SERVER", "SGCN05")
SQL_DATABASE = os.getenv("UDLA_SQL_DATABASE", "BDD_Proyectos")
SQL_SCHEMA = os.getenv("UDLA_SQL_SCHEMA", "Mercados")
TABLES = [
    "Personas",
    "Familiares",
    "Ingresos",
    "Deudas",
]


def _to_decimal(series: pd.Series) -> pd.Series:
    """
    Convierte una serie a float interpretando coma como separador decimal.
    Soporta mezcla de enteros ("36829") y decimales ("566,53").
    Si coexisten coma y punto, asume coma decimal y punto como miles -> quita puntos.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.strip()
    s = s.replace({"": pd.NA, "None": pd.NA, "nan": pd.NA})
    s = s.str.replace(r"[^\d,.\-+]", "", regex=True)
    both = s.str.contains(",") & s.str.contains(r"\.")
    s = s.mask(both, s.str.replace(".", "", regex=False))
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _build_odbc_connect(driver: str, encrypt: str, trust_cert: str) -> str:
    return (
        f"DRIVER={{{driver}}};"
        f"SERVER={SQL_SERVER};"
        f"DATABASE={SQL_DATABASE};"
        "Trusted_Connection=yes;"
        f"Encrypt={encrypt};"
        f"TrustServerCertificate={trust_cert};"
    )


def _get_sql_engine():
    env_driver = os.getenv("SQLSERVER_DRIVER", "").strip()
    env_encrypt = os.getenv("SQL_ENCRYPT", "").strip().lower()
    env_trust = os.getenv("SQL_TRUST_CERT", "").strip().lower()

    drivers = (
        [env_driver]
        if env_driver
        else ["ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server"]
    )
    encrypt_opts = [env_encrypt] if env_encrypt in {"yes", "no"} else ["yes", "no"]
    trust_cert = env_trust if env_trust in {"yes", "no"} else "yes"

    last_err = None
    for driver in drivers:
        for encrypt in encrypt_opts:
            engine = None
            try:
                odbc_str = _build_odbc_connect(
                    driver, encrypt=encrypt, trust_cert=trust_cert
                )
                params = urllib.parse.quote_plus(odbc_str)
                engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
                with engine.connect() as _:
                    pass
                return engine
            except Exception as exc:
                last_err = exc
                try:
                    if engine is not None:
                        engine.dispose()
                except Exception:
                    pass
                continue
    raise last_err


@st.cache_data(show_spinner=False, ttl=3600)
def cargar_datos_udla() -> dict[str, pd.DataFrame]:
    """
    Carga tablas clave del SQL UDLA y normaliza nombres de columnas a minúsculas.
    """
    try:
        engine = _get_sql_engine()
    except Exception as exc:
        st.error(f"No se pudo conectar a SQL Server ({SQL_SERVER}/{SQL_DATABASE}).")
        st.error(str(exc))
        st.stop()

    data: dict[str, pd.DataFrame] = {}
    try:
        with engine.connect() as conn:
            for tabla in TABLES:
                query = f"SELECT * FROM [{SQL_SCHEMA}].[{tabla}]"
                df_tabla = pd.read_sql(query, conn)
                df_tabla.columns = df_tabla.columns.map(lambda c: str(c).strip().lower())

                for col in ["identificacion", "ced_padre", "ced_madre", "ruc_empleador"]:
                    if col in df_tabla.columns:
                        df_tabla[col] = df_tabla[col].astype(str).str.strip()

                if tabla.lower() == "ingresos" and "salario" in df_tabla.columns:
                    df_tabla["salario"] = _to_decimal(df_tabla["salario"])

                if tabla.lower() == "deudas" and "valor" in df_tabla.columns:
                    df_tabla["valor"] = _to_decimal(df_tabla["valor"])

                data[tabla] = df_tabla
    finally:
        try:
            engine.dispose()
        except Exception:
            pass

    return data
