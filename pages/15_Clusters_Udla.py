from __future__ import annotations

import io
import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.udla_clusters import (
    build_udla_cluster_base,
    build_university_cluster_base,
    run_udla_cluster_analysis,
    run_university_cluster_projection,
)

st.set_page_config(page_title="Clusters Udla", page_icon="C", layout="wide")

COLOR_PRIMARY = "#1d4ed8"
COLOR_ACCENT = "#0f766e"
CHART_TEMPLATE = "plotly_white"
CHART_MARGIN = dict(l=24, r=24, t=40, b=24)
INGRESO_TICKS = [0, 1, 2, 3, 4, 5]
INGRESO_LABELS = ["Sin empleo", "Q1", "Q2", "Q3", "Q4", "Q5"]
DEUDA_TICKS = [0, 1, 2, 3, 4, 5]
DEUDA_LABELS = ["Sin deuda", "Q1", "Q2", "Q3", "Q4", "Q5"]


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


def _render_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    specs = [
        ("tipo_estudiante", "Tipo", "Todos los tipos"),
        ("facultad", "Facultad", "Todas las facultades"),
        ("carrera", "Carrera", "Todas las carreras"),
    ]
    filtered = df.copy()
    selections: dict[str, str | None] = {key: None for key, _, _ in specs}
    cols = st.columns(len(specs))

    for ui_col, (key, label, all_label) in zip(cols, specs):
        current_values = _clean_series(filtered[key])
        options = sorted(current_values[current_values != "Sin dato"].unique().tolist())
        select_options = [all_label] + options
        widget_key = f"clusters_udla_{key}"

        current_widget_value = st.session_state.get(widget_key)
        if current_widget_value not in select_options:
            st.session_state.pop(widget_key, None)

        with ui_col:
            selected = st.selectbox(
                label,
                options=select_options,
                index=0,
                key=widget_key,
            )

        if selected != all_label:
            selections[key] = selected
            filtered = filtered[current_values == selected].copy()

    return filtered, selections


def _ranges_frame(label: str, ranges: dict[int, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for quintil in range(1, 6):
        item = ranges.get(quintil, {"min": 0.0, "max": 0.0})
        rows.append(
            {
                "Tipo": label,
                "Quintil": f"Q{quintil}",
                "Minimo": float(item.get("min", 0.0)),
                "Maximo": float(item.get("max", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _cluster_summary_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    columns = [
        "cluster",
        "estudiantes",
        "hogares",
        "edad_estudiante",
        "pct_mujeres",
        "pct_quito",
        "pct_primera_generacion",
        "quintil_ingreso_promedio",
        "quintil_deuda_promedio",
        "hijos_promedio",
        "sexo_modal",
        "tipo_modal",
        "facultad_modal",
        "carrera_modal",
        "quintil_ingreso_modal",
        "quintil_deuda_modal",
    ]
    rename_map = {
        "cluster": "Cluster",
        "estudiantes": "Estudiantes",
        "hogares": "Hogares",
        "edad_estudiante": "Edad prom.",
        "pct_mujeres": "% mujeres",
        "pct_quito": "% Quito",
        "pct_primera_generacion": "% 1ra gen.",
        "quintil_ingreso_promedio": "Q ingreso prom.",
        "quintil_deuda_promedio": "Q deuda prom.",
        "hijos_promedio": "Hijos prom.",
        "sexo_modal": "Genero dom.",
        "tipo_modal": "Tipo dom.",
        "facultad_modal": "Facultad dom.",
        "carrera_modal": "Carrera dom.",
        "quintil_ingreso_modal": "Q ingreso dom.",
        "quintil_deuda_modal": "Q deuda dom.",
    }
    available = [col for col in columns if col in df.columns]
    return df[available].rename(columns=rename_map)


def _cluster_members_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["estudiante_quito"] = out["estudiante_quito"].map({1: "Si", 0: "No"}).fillna("No")
    out["primera_generacion"] = (
        out["primera_generacion"].map({1: "Si", 0: "No"}).fillna("No")
    )
    columns = [
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
        "cluster",
    ]
    rename_map = {
        "IDENTIFICACION": "Identificacion",
        "hogar_id": "Hogar",
        "tipo_estudiante": "Tipo",
        "facultad": "Facultad",
        "carrera": "Carrera",
        "sexo_estudiante": "Genero",
        "edad_estudiante": "Edad",
        "estudiante_quito": "Quito",
        "primera_generacion": "1ra generacion",
        "quintil_ingreso_hogar": "Quintil ingreso",
        "quintil_deuda_hogar": "Quintil deuda",
        "hijos_hogar_promedio": "Hijos hogar prom.",
        "cluster": "Cluster",
    }
    return out[columns].rename(columns=rename_map)


def _detail_cluster_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
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
    return out[columns].rename(columns=rename_map)


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


def _stacked_bar_chart(
    df: pd.DataFrame,
    category_series: pd.Series,
    title: str,
    *,
    category_order: list[str] | None = None,
    unique_id_col: str | None = None,
    y_label: str = "Participacion (%)",
    normalize: bool = True,
    height: int = 320,
) -> go.Figure:
    if df.empty:
        return go.Figure()

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
    grouped["valor"] = (
        grouped["conteo"] / grouped["total"] * 100.0 if normalize else grouped["conteo"]
    )

    if category_order is None:
        category_order = (
            grouped.groupby("categoria", as_index=False)["conteo"]
            .sum()
            .sort_values("conteo", ascending=False)["categoria"]
            .tolist()
        )

    cluster_order = pd.Index(df["cluster"].dropna().astype(str)).unique().tolist()
    fig = px.bar(
        grouped,
        x="cluster",
        y="valor",
        color="categoria",
        barmode="stack",
        template=CHART_TEMPLATE,
        category_orders={"cluster": cluster_order, "categoria": category_order},
        hover_data={"conteo": True, "total": True, "valor": ":.1f"},
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        margin=CHART_MARGIN,
        xaxis_title="",
        yaxis_title=y_label,
        legend_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
    )
    if normalize:
        fig.update_yaxes(range=[0, 100])
    return fig


def _overview_table(
    cluster_tables: dict[str, pd.DataFrame], cluster_order: list[str]
) -> pd.DataFrame:
    rows = []
    for cluster_name in cluster_order:
        table = cluster_tables.get(cluster_name, pd.DataFrame()).copy()
        if table.empty:
            continue
        rows.append(
            {
                "Cluster": cluster_name,
                "Cantidad de personas": int(table["Cantidad de personas"].sum()),
                "Universidades con personas": int(
                    (
                        pd.to_numeric(
                            table["Cantidad de personas"], errors="coerce"
                        ).fillna(0)
                        > 0
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _cluster_university_pivot(
    cluster_tables: dict[str, pd.DataFrame], cluster_order: list[str]
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for cluster_name in cluster_order:
        table = cluster_tables.get(cluster_name, pd.DataFrame()).copy()
        if table.empty:
            continue
        current = table[["Universidad", "Cantidad de personas"]].copy()
        current["Cluster"] = cluster_name
        rows.append(current[["Cluster", "Universidad", "Cantidad de personas"]])

    if not rows:
        return pd.DataFrame()

    stacked = pd.concat(rows, ignore_index=True)
    stacked["Cantidad de personas"] = (
        pd.to_numeric(stacked["Cantidad de personas"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    pivot = (
        stacked.pivot_table(
            index="Cluster",
            columns="Universidad",
            values="Cantidad de personas",
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index(axis=1)
        .reset_index()
    )

    ordered_rows = [name for name in cluster_order if name in pivot["Cluster"].tolist()]
    if ordered_rows:
        pivot["Cluster"] = pd.Categorical(
            pivot["Cluster"], categories=ordered_rows, ordered=True
        )
        pivot = pivot.sort_values("Cluster").reset_index(drop=True)
        pivot["Cluster"] = pivot["Cluster"].astype(str)

    return pivot


def _udla_reference_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    columns = [
        "cluster",
        "estudiantes",
        "hogares",
        "quintil_ingreso_modal",
        "quintil_deuda_modal",
        "tipo_estudiantes_pg",
    ]
    rename_map = {
        "cluster": "Cluster",
        "estudiantes": "Estudiantes UDLA",
        "hogares": "Hogares UDLA",
        "quintil_ingreso_modal": "Q ingreso UDLA",
        "quintil_deuda_modal": "Q deuda UDLA",
        "tipo_estudiantes_pg": "Tipo estudiantes",
    }
    return summary_df[columns].rename(columns=rename_map)


def _distribution_chart(
    df: pd.DataFrame,
    column: str,
    title: str,
    top_n: int | None = None,
    category_order: list[str] | None = None,
) -> go.Figure:
    values = _clean_series(df[column])
    counts = values.value_counts()
    if top_n is not None:
        counts = counts.head(top_n)
    chart_df = counts.reset_index()
    chart_df.columns = ["categoria", "conteo"]
    total = max(int(chart_df["conteo"].sum()), 1)
    chart_df["pct"] = chart_df["conteo"] / total * 100.0

    fig = px.bar(
        chart_df,
        x="categoria",
        y="pct",
        text=chart_df["pct"].map(lambda value: f"{value:.1f}%"),
        template=CHART_TEMPLATE,
        category_orders={"categoria": category_order or chart_df["categoria"].tolist()},
    )
    fig.update_traces(marker_color=COLOR_PRIMARY, textposition="outside", cliponaxis=False)
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        margin=CHART_MARGIN,
        xaxis_title="",
        yaxis_title="Participacion (%)",
        showlegend=False,
        height=300,
    )
    return fig


def _profile_heatmap(profile_df: pd.DataFrame) -> go.Figure:
    if profile_df.empty:
        return go.Figure()

    heat = profile_df.set_index("cluster").T
    fig = go.Figure(
        data=go.Heatmap(
            z=heat.to_numpy(dtype=float),
            x=heat.columns.tolist(),
            y=heat.index.tolist(),
            colorscale="Blues",
            text=[[f"{value:.1f}" for value in row] for row in heat.to_numpy(dtype=float)],
            texttemplate="%{text}",
            colorbar=dict(title="Valor"),
        )
    )
    fig.update_layout(
        title=dict(text="Perfil medio por cluster", font=dict(size=13)),
        margin=CHART_MARGIN,
        height=380,
    )
    return fig


def _to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Detalle") -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()


st.title("Clusters Udla")
st.caption(
    "Universo exclusivo de UDLA. Los clusters se recalculan con los filtros actuales "
    "usando edad, genero, condicion Quito (cedula inicia con 17), primera generacion, "
    "quintil de ingreso del hogar, quintil de deuda del hogar y promedio de hijos del hogar."
)

with st.spinner("Preparando universo UDLA..."):
    data_bundle = build_udla_cluster_base()
    university_bundle = build_university_cluster_base()

base_df = data_bundle["students"].copy()
income_ranges = data_bundle.get("income_ranges", {})
debt_ranges = data_bundle.get("debt_ranges", {})
all_students_df = university_bundle["students"].copy()

if base_df.empty:
    st.info("No hay datos UDLA disponibles para construir clusters.")
    st.stop()

st.markdown("### Filtros")
filtered_df, selections = _render_filters(base_df)

if filtered_df.empty:
    st.info("No hay estudiantes para los filtros seleccionados.")
    st.stop()

with st.spinner("Calculando clusters de estudiantes UDLA..."):
    analysis = run_udla_cluster_analysis(filtered_df)
    external_universities_df = all_students_df[
        all_students_df["fuente_archivo"] == "Universidades"
    ].copy()
    university_analysis = run_university_cluster_projection(
        filtered_df, external_universities_df
    )

students_df = analysis["students"].copy()
summary_df = analysis["summary"].copy()
profile_df = analysis["profile"].copy()
cluster_count = int(analysis.get("k", 0))
university_cluster_tables = university_analysis.get("cluster_tables", {})
university_cluster_order = university_analysis.get("cluster_order", [])
projected_students_df = university_analysis.get("students", pd.DataFrame()).copy()
udla_projection_summary = university_analysis.get("udla_summary", pd.DataFrame()).copy()
other_cluster_count = int(university_analysis.get("other_count", 0))

if students_df.empty or summary_df.empty:
    st.info("No fue posible construir clusters con los filtros seleccionados.")
    st.stop()

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Estudiantes", f"{int(students_df['IDENTIFICACION'].nunique()):,}")
metric_2.metric("Hogares", f"{int(students_df['hogar_id'].nunique()):,}")
metric_3.metric("Clusters", cluster_count)
metric_4.metric(
    "% Quito",
    f"{float(students_df['estudiante_quito'].mean() * 100.0):.1f}%",
)

tab_mapa, tab_perfiles, tab_detalle, tab_detalle_graficos, tab_universidades, tab_metodo = st.tabs(
    [
        "Mapa de clusters",
        "Perfiles",
        "Detalle",
        "Detalle Graficos",
        "Cluster Universidades",
        "Metodologia",
    ]
)

with tab_mapa:
    col_left, col_right = st.columns(2)

    with col_left:
        scatter_students = px.scatter(
            students_df,
            x="edad_estudiante",
            y="quintil_ingreso_num",
            color="cluster",
            hover_name="carrera",
            hover_data={
                "facultad": True,
                "tipo_estudiante": True,
                "sexo_estudiante": True,
                "quintil_deuda_hogar": True,
                "hijos_hogar_promedio": ":.1f",
                "edad_estudiante": ":.1f",
                "quintil_ingreso_num": False,
            },
            template=CHART_TEMPLATE,
            opacity=0.72,
        )
        scatter_students.update_layout(
            title=dict(text="Edad vs quintil de ingreso", font=dict(size=13)),
            margin=CHART_MARGIN,
            xaxis_title="Edad estudiante",
            yaxis_title="Quintil ingreso hogar",
            legend_title="",
            height=400,
        )
        scatter_students.update_yaxes(
            tickvals=INGRESO_TICKS,
            ticktext=INGRESO_LABELS,
        )
        st.plotly_chart(scatter_students, use_container_width=True)

    with col_right:
        centroid_fig = px.scatter(
            summary_df,
            x="quintil_ingreso_promedio",
            y="quintil_deuda_promedio",
            size="estudiantes",
            color="cluster",
            text="cluster",
            hover_data={
                "hogares": True,
                "pct_quito": ":.1f",
                "pct_primera_generacion": ":.1f",
                "hijos_promedio": ":.1f",
            },
            template=CHART_TEMPLATE,
        )
        centroid_fig.update_traces(textposition="top center")
        centroid_fig.update_layout(
            title=dict(text="Centroides por cluster", font=dict(size=13)),
            margin=CHART_MARGIN,
            xaxis_title="Quintil ingreso promedio",
            yaxis_title="Quintil deuda promedio",
            legend_title="",
            height=400,
        )
        centroid_fig.update_xaxes(tickvals=INGRESO_TICKS, ticktext=INGRESO_LABELS)
        centroid_fig.update_yaxes(tickvals=DEUDA_TICKS, ticktext=DEUDA_LABELS)
        st.plotly_chart(centroid_fig, use_container_width=True)

    st.caption(
        "Izquierda: estudiantes individuales. Derecha: cada burbuja resume un cluster "
        "y su tamano refleja la cantidad de estudiantes."
    )

with tab_perfiles:
    counts_fig = px.bar(
        summary_df,
        x="cluster",
        y="estudiantes",
        text="estudiantes",
        color="cluster",
        template=CHART_TEMPLATE,
    )
    counts_fig.update_layout(
        title=dict(text="Tamano de los clusters", font=dict(size=13)),
        margin=CHART_MARGIN,
        xaxis_title="",
        yaxis_title="Estudiantes",
        legend_title="",
        height=320,
    )
    st.plotly_chart(counts_fig, use_container_width=True)

    st.plotly_chart(_profile_heatmap(profile_df), use_container_width=True)

    st.dataframe(
        _cluster_summary_display(summary_df),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Edad prom.": st.column_config.NumberColumn("Edad prom.", format="%.1f"),
            "% mujeres": st.column_config.NumberColumn("% mujeres", format="%.1f%%"),
            "% Quito": st.column_config.NumberColumn("% Quito", format="%.1f%%"),
            "% 1ra gen.": st.column_config.NumberColumn("% 1ra gen.", format="%.1f%%"),
            "Q ingreso prom.": st.column_config.NumberColumn(
                "Q ingreso prom.", format="%.2f"
            ),
            "Q deuda prom.": st.column_config.NumberColumn(
                "Q deuda prom.", format="%.2f"
            ),
            "Hijos prom.": st.column_config.NumberColumn("Hijos prom.", format="%.1f"),
        },
    )

with tab_detalle:
    detail_df = _detail_cluster_table(summary_df)
    st.dataframe(
        detail_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Cantidad estudiantes": st.column_config.NumberColumn(
                "Cantidad estudiantes", format="%d"
            ),
            "15-19 anos": st.column_config.NumberColumn(
                "15-19 anos", format="%.1f%%"
            ),
            "20-22 anos": st.column_config.NumberColumn(
                "20-22 anos", format="%.1f%%"
            ),
            "23-25 anos": st.column_config.NumberColumn(
                "23-25 anos", format="%.1f%%"
            ),
            "Mas de 25 anos": st.column_config.NumberColumn(
                "Mas de 25 anos", format="%.1f%%"
            ),
            "Es de Quito (%)": st.column_config.NumberColumn(
                "Es de Quito (%)", format="%.1f%%"
            ),
            "Promedio hijos": st.column_config.NumberColumn(
                "Promedio hijos", format="%d"
            ),
        },
    )
    st.download_button(
        label="Descargar detalle en Excel (.xlsx)",
        data=_to_excel_bytes(detail_df, sheet_name="Detalle UDLA"),
        file_name="detalle_clusters_udla.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="clusters_udla_detalle_xlsx",
    )

with tab_detalle_graficos:
    st.caption(
        "Distribucion apilada por cluster para las mismas variables resumidas en la tabla de detalle."
    )

    age_col, gender_col = st.columns(2)
    with age_col:
        st.plotly_chart(
            _stacked_bar_chart(
                students_df,
                _age_bucket(students_df["edad_estudiante"]),
                "Edad estudiante",
                category_order=["15-19 anos", "20-22 anos", "23-25 anos", "Mas de 25 anos", "Sin dato"],
            ),
            use_container_width=True,
        )
    with gender_col:
        st.plotly_chart(
            _stacked_bar_chart(
                students_df,
                students_df["sexo_estudiante"],
                "Genero estudiante",
                category_order=["MUJER", "HOMBRE", "DESCONOCIDO"],
            ),
            use_container_width=True,
        )

    quito_col, ingreso_col = st.columns(2)
    with quito_col:
        st.plotly_chart(
            _stacked_bar_chart(
                students_df,
                _yes_no_series(students_df["estudiante_quito"]),
                "Es de Quito",
                category_order=["Si", "No"],
            ),
            use_container_width=True,
        )
    with ingreso_col:
        st.plotly_chart(
            _stacked_bar_chart(
                students_df,
                students_df["quintil_ingreso_hogar"],
                "Quintil de ingresos",
                category_order=["Sin empleo", "1", "2", "3", "4", "5"],
            ),
            use_container_width=True,
        )

    hijos_col, deuda_col = st.columns(2)
    with hijos_col:
        st.plotly_chart(
            _stacked_bar_chart(
                students_df,
                _hijos_bucket(students_df["hijos_hogar_promedio"]),
                "Promedio hijos en el hogar",
                category_order=["0", "1", "2", "3", "4", "5+"],
            ),
            use_container_width=True,
        )
    with deuda_col:
        st.plotly_chart(
            _stacked_bar_chart(
                students_df,
                students_df["quintil_deuda_hogar"],
                "Hogares con deuda / quintil deuda",
                category_order=["Sin deuda", "1", "2", "3", "4", "5"],
                unique_id_col="hogar_id",
            ),
            use_container_width=True,
        )

    estado_col, tipo_col = st.columns(2)
    with estado_col:
        st.plotly_chart(
            _stacked_bar_chart(
                students_df,
                students_df["estado_hogar"],
                "Estado del hogar",
            ),
            use_container_width=True,
        )
    with tipo_col:
        st.plotly_chart(
            _stacked_bar_chart(
                students_df,
                _yes_no_series(students_df["primera_generacion"]).replace(
                    {"Si": "Primera generacion", "No": "No primera generacion"}
                ),
                "Tipo de estudiantes",
                category_order=["Primera generacion", "No primera generacion"],
            ),
            use_container_width=True,
        )

with tab_universidades:
    st.caption(
        "Esta vista usa directamente `db/Udla.xlsx` y `db/Universidades.xlsx`. "
        "No depende del archivo activo seleccionado en el dashboard."
    )

    if projected_students_df.empty or not university_cluster_order:
        st.info(
            "No hay suficientes datos para proyectar universidades sobre los clusters UDLA."
        )
    else:
        u1, u2, u3, u4 = st.columns(4)
        u1.metric(
            "Personas evaluadas",
            f"{int(projected_students_df['IDENTIFICACION'].nunique()):,}",
        )
        u2.metric(
            "Universidades",
            f"{int(projected_students_df['Universidad'].fillna('').astype(str).str.strip().nunique()):,}",
        )
        u3.metric("Clusters UDLA", int(university_analysis.get("k", 0)))
        u4.metric("Otro cluster", f"{other_cluster_count:,}")

        st.dataframe(
            _overview_table(university_cluster_tables, university_cluster_order),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Cantidad de personas": st.column_config.NumberColumn(
                    "Cantidad de personas", format="%d"
                ),
                "Universidades con personas": st.column_config.NumberColumn(
                    "Universidades con personas", format="%d"
                ),
            },
        )

        st.markdown("#### Distribucion por cluster y universidad")
        pivot_df = _cluster_university_pivot(
            university_cluster_tables, university_cluster_order
        )
        if pivot_df.empty:
            st.info("No hay datos para mostrar la matriz cluster x universidad.")
        else:
            pivot_config = {
                col: st.column_config.NumberColumn(col, format="%d")
                for col in pivot_df.columns
                if col != "Cluster"
            }
            st.dataframe(
                pivot_df,
                use_container_width=True,
                hide_index=True,
                column_config=pivot_config,
            )

        with st.expander("Ver referencia de clusters UDLA", expanded=False):
            st.dataframe(
                _udla_reference_table(udla_projection_summary),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Estudiantes UDLA": st.column_config.NumberColumn(
                        "Estudiantes UDLA", format="%d"
                    ),
                    "Hogares UDLA": st.column_config.NumberColumn(
                        "Hogares UDLA", format="%d"
                    ),
                },
            )

with tab_metodo:
    st.markdown(
        "### Como se construyen los clusters\n"
        "La unidad de analisis es el estudiante UDLA. Cada cluster se calcula solo con "
        "las variables solicitadas y se vuelve a estimar segun los filtros de tipo, "
        "facultad y carrera."
    )

    methodology = pd.DataFrame(
        [
            {"Variable": "Edad estudiante", "Tratamiento": "Numerica"},
            {"Variable": "Genero estudiante", "Tratamiento": "Categorica"},
            {"Variable": "Estudiante de Quito", "Tratamiento": "Binaria (cedula inicia con 17)"},
            {"Variable": "Primera generacion", "Tratamiento": "Binaria"},
            {"Variable": "Quintil de ingreso del hogar", "Tratamiento": "Ordinal (0-5)"},
            {"Variable": "Quintil de deuda del hogar", "Tratamiento": "Ordinal (0-5)"},
            {
                "Variable": "Promedio de hijos en el hogar",
                "Tratamiento": "Numerica derivada desde max(padre, madre), sin sumar",
            },
        ]
    )
    st.dataframe(methodology, use_container_width=True, hide_index=True)

    st.markdown(
        "El numero de clusters se elige automaticamente con el indice de "
        "Calinski-Harabasz, manteniendo grupos con tamano minimo razonable."
    )
    st.markdown(
        "En la pestaña `Cluster Universidades`, los clusters se entrenan con UDLA y "
        "luego se proyectan estudiantes de `db/Universidades.xlsx` al cluster UDLA mas "
        "cercano. Los filtros de la pagina se usan para definir el grupo de referencia "
        "UDLA, pero la proyeccion toma directamente todas las universidades de "
        "`db/Universidades.xlsx`. Si una observacion cae fuera del radio maximo "
        "observado en UDLA para ese cluster, se clasifica como `Otro cluster`."
    )

    range_left, range_right = st.columns(2)
    with range_left:
        st.markdown("#### Rangos de quintiles de ingreso UDLA")
        st.dataframe(
            _ranges_frame("Ingreso", income_ranges),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Minimo": st.column_config.NumberColumn("Minimo", format="$%.2f"),
                "Maximo": st.column_config.NumberColumn("Maximo", format="$%.2f"),
            },
        )
    with range_right:
        st.markdown("#### Rangos de quintiles de deuda UDLA")
        st.dataframe(
            _ranges_frame("Deuda", debt_ranges),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Minimo": st.column_config.NumberColumn("Minimo", format="$%.2f"),
                "Maximo": st.column_config.NumberColumn("Maximo", format="$%.2f"),
            },
        )
