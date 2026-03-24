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


def _excel_path(filename: str = "data.xlsx") -> Path:
    return _ROOT_DIR / "db" / filename


def load_excel(filename: str = "data.xlsx") -> dict[str, pd.DataFrame]:
    """
    Carga el Excel completo (todas las hojas) desde `db/<filename>` con cache.

    Devuelve un dict {nombre_hoja: DataFrame}.
    """
    path = _excel_path(filename)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo Excel: {path}")

    mtime_ns = path.stat().st_mtime_ns
    return _load_excel_all_cached(str(path), mtime_ns)


def load_excel_sheet(sheet_name: str, filename: str = "data.xlsx") -> pd.DataFrame:
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
