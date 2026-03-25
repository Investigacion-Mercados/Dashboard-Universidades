from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    from utils.student_columns import find_column
except ImportError:  # pragma: no cover
    from student_columns import find_column

FILTER_SPECS = [
    {
        "key": "universidad",
        "label": "Universidad",
        "aliases": ["Universidad"],
        "all_label": "Todas las universidades",
    },
    {
        "key": "tipo",
        "label": "Tipo",
        "aliases": ["Tipo"],
        "all_label": "Todos los tipos",
    },
    {
        "key": "facultad",
        "label": "Facultad",
        "aliases": ["Facultad"],
        "all_label": "Todas las facultades",
    },
    {
        "key": "carrera",
        "label": "Carrera",
        "aliases": ["Carrera"],
        "all_label": "Todas las carreras",
    },
]


def _series_filtro(df: pd.DataFrame, column_name: str) -> pd.Series:
    return df[column_name].fillna("").astype(str).str.strip()


def apply_student_academic_filters(
    df: pd.DataFrame, selections: dict[str, str | None]
) -> pd.DataFrame:
    filtered = df
    for spec in FILTER_SPECS:
        selected_value = selections.get(spec["key"])
        if not selected_value:
            continue
        column_name = find_column(filtered, spec["aliases"])
        if column_name is None:
            continue
        filtered = filtered[_series_filtro(filtered, column_name) == selected_value].copy()
    return filtered


def render_student_academic_filters(
    df: pd.DataFrame,
    *,
    key_prefix: str,
    warn_missing_university: bool = True,
    lock_single_option_keys: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    selections: dict[str, str | None] = {spec["key"]: None for spec in FILTER_SPECS}
    available_specs = []
    has_university = False
    locked_keys = lock_single_option_keys or set()

    for spec in FILTER_SPECS:
        column_name = find_column(df, spec["aliases"])
        if column_name is None:
            continue
        if spec["key"] == "universidad":
            has_university = True
        available_specs.append((spec, column_name))

    if not has_university and warn_missing_university:
        st.warning("La hoja Estudiantes no contiene la columna 'Universidad'.")

    if not available_specs:
        return df, selections

    filtered = df
    columns = st.columns(len(available_specs))
    for ui_col, (spec, column_name) in zip(columns, available_specs):
        current_values = _series_filtro(filtered, column_name)
        options = sorted(current_values[current_values != ""].unique().tolist())
        widget_key = f"{key_prefix}_{spec['key']}"
        select_options = options
        if not (spec["key"] in locked_keys and len(options) == 1):
            select_options = [spec["all_label"]] + options

        try:
            current_widget_value = st.session_state.get(widget_key)
            if current_widget_value not in select_options:
                st.session_state.pop(widget_key, None)
        except Exception:
            pass

        with ui_col:
            if spec["key"] in locked_keys and len(options) == 1:
                selected_value = options[0]
                st.selectbox(
                    spec["label"],
                    options=options,
                    index=0,
                    key=widget_key,
                    disabled=True,
                )
            else:
                selected_option = st.selectbox(
                    spec["label"],
                    options=select_options,
                    index=0,
                    key=widget_key,
                )
                selected_value = (
                    None if selected_option == spec["all_label"] else selected_option
                )
        selections[spec["key"]] = selected_value
        if selected_value is not None:
            filtered = filtered[current_values == selected_value].copy()

    return filtered, selections
