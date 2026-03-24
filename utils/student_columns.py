from __future__ import annotations

import re

import pandas as pd


def _normalize_column_name(name: str) -> str:
    return re.sub(r"\s+", "", str(name)).casefold()


def find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    normalized = {_normalize_column_name(col): str(col) for col in df.columns}
    for alias in aliases:
        match = normalized.get(_normalize_column_name(alias))
        if match is not None:
            return match
    return None


def normalize_university_column(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza la columna de universidad a `Universidad`."""
    university_col = find_column(df, ["Universidad"])
    if university_col is None:
        return df

    university = df[university_col].copy()
    if university.dtype == "O":
        university = university.fillna("").astype(str).str.strip()
        university = university.replace("", "Sin dato")
    else:
        university = university.fillna("Sin dato")

    if university_col != "Universidad":
        df = df.rename(columns={university_col: "Universidad"})

    return df.assign(Universidad=university)
