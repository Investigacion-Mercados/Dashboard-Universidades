from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

_USE_STREAMLIT_CACHE = bool(
    st is not None and getattr(st, "runtime", None) is not None and st.runtime.exists()
)

_ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EXCEL_FILENAME = "data.xlsx"
EXCEL_SESSION_KEY = "selected_excel_filename"
_VALID_EXCEL_SUFFIXES = {".xlsx", ".xlsm"}


def list_excel_files() -> list[str]:
    db_dir = _ROOT_DIR / "db"
    if not db_dir.exists():
        return []
    return sorted(
        path.name
        for path in db_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _VALID_EXCEL_SUFFIXES
    )


def _default_excel_filename(files: list[str] | None = None) -> str:
    disponibles = files if files is not None else list_excel_files()
    if not disponibles:
        raise FileNotFoundError("No se encontraron archivos Excel en la carpeta db.")
    if DEFAULT_EXCEL_FILENAME in disponibles:
        return DEFAULT_EXCEL_FILENAME
    return disponibles[0]


def get_active_excel_filename(filename: str | None = None) -> str:
    if filename and filename != DEFAULT_EXCEL_FILENAME:
        return filename

    disponibles = list_excel_files()
    seleccionado = _default_excel_filename(disponibles)

    if st is None:
        return seleccionado

    try:
        valor_session = str(st.session_state.get(EXCEL_SESSION_KEY, seleccionado)).strip()
    except Exception:
        return seleccionado

    if valor_session in disponibles:
        return valor_session

    try:
        st.session_state[EXCEL_SESSION_KEY] = seleccionado
    except Exception:
        pass
    return seleccionado


def set_active_excel_filename(filename: str) -> str:
    disponibles = list_excel_files()
    if filename not in disponibles:
        raise FileNotFoundError(f"No existe el archivo Excel seleccionado: {filename}")
    if st is not None:
        try:
            st.session_state[EXCEL_SESSION_KEY] = filename
        except Exception:
            pass
    return filename


def _excel_path(filename: str | None = None) -> Path:
    return _ROOT_DIR / "db" / get_active_excel_filename(filename)


def load_excel(filename: str | None = None) -> dict[str, pd.DataFrame]:
    """
    Carga el Excel completo (todas las hojas) desde `db/<filename>` con cache.

    Devuelve un dict {nombre_hoja: DataFrame}.
    """
    path = _excel_path(filename)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo Excel: {path}")

    mtime_ns = path.stat().st_mtime_ns
    return _load_excel_all_cached(str(path), mtime_ns)


def load_excel_sheet(sheet_name: str, filename: str | None = None) -> pd.DataFrame:
    """
    Carga una hoja específica desde `db/<filename>` con cache.
    """
    path = _excel_path(filename)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo Excel: {path}")

    mtime_ns = path.stat().st_mtime_ns
    return _load_excel_sheet_cached(str(path), sheet_name, mtime_ns)


def _read_all(path: str) -> dict[str, pd.DataFrame]:
    # sheet_name=None => devuelve dict de hojas
    data = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    return {str(k): v for k, v in data.items()}


def _read_sheet(path: str, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")


if _USE_STREAMLIT_CACHE:

    @st.cache_data(show_spinner=False)
    def _load_excel_all_cached(path: str, mtime_ns: int) -> dict[str, pd.DataFrame]:
        _ = mtime_ns  # parte de la key del cache
        return _read_all(path)

    @st.cache_data(show_spinner=False)
    def _load_excel_sheet_cached(
        path: str, sheet_name: str, mtime_ns: int
    ) -> pd.DataFrame:
        _ = mtime_ns  # parte de la key del cache
        return _read_sheet(path, sheet_name)

else:

    @lru_cache(maxsize=4)
    def _load_excel_all_cached(path: str, mtime_ns: int) -> dict[str, pd.DataFrame]:
        _ = mtime_ns  # parte de la key del cache
        return _read_all(path)

    @lru_cache(maxsize=64)
    def _load_excel_sheet_cached(
        path: str, sheet_name: str, mtime_ns: int
    ) -> pd.DataFrame:
        _ = mtime_ns  # parte de la key del cache
        return _read_sheet(path, sheet_name)
