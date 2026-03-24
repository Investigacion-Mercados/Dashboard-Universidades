from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.propensity_helpers import run_propensity_analysis

st.set_page_config(
    page_title="Propensity Score vs UDLA",
    page_icon="P",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COLOR_UNI = "#2563eb"
COLOR_UDLA = "#dc2626"
COLOR_ACCENT = "#059669"
COLOR_WARN = "#d97706"
COLOR_MUTED = "#94a3b8"
CHART_TEMPLATE = "plotly_white"
CHART_MARGIN = dict(l=24, r=24, t=40, b=24)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _ranges_frame(label: str, ranges: dict[int, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for quintil in range(1, 6):
        item = ranges.get(quintil, {"min": 0.0, "max": 0.0})
        rows.append(
            {
                "Institucion": label,
                "Quintil": f"Q{quintil}",
                "Ingreso minimo": float(item.get("min", 0.0)),
                "Ingreso maximo": float(item.get("max", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _cluster_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    columns = [
        "cluster",
        "hogares",
        "estudiantes_vinculados",
        "propensity_promedio",
        "propensity_mediana",
        "salario_promedio",
        "deuda_promedio",
        "hijos_promedio",
        "edad_padres_promedio",
        "quintil_modal",
        "estado_modal",
        "carrera_modal",
        "parroquia_modal",
        "primera_generacion_pct",
        "hogar_huerfano_pct",
        "riesgo_pct",
        "con_deuda_pct",
        "con_empleo_formal_pct",
        "con_parroquia_pct",
    ]
    if "cluster_udla_cercano" in df.columns:
        columns.extend(["cluster_udla_cercano", "similitud_centroidal"])

    available = [c for c in columns if c in df.columns]

    rename_map = {
        "cluster": "Cluster",
        "hogares": "Hogares",
        "estudiantes_vinculados": "Estudiantes",
        "propensity_promedio": "Propensity prom.",
        "propensity_mediana": "Propensity med.",
        "salario_promedio": "Salario prom.",
        "deuda_promedio": "Deuda prom.",
        "hijos_promedio": "Hijos prom.",
        "edad_padres_promedio": "Edad padres prom.",
        "quintil_modal": "Quintil dom.",
        "estado_modal": "Estado dom.",
        "carrera_modal": "Carrera dom.",
        "parroquia_modal": "Parroquia dom.",
        "primera_generacion_pct": "1ra gen. %",
        "hogar_huerfano_pct": "Huerfanos %",
        "riesgo_pct": "En riesgo %",
        "con_deuda_pct": "Con deuda %",
        "con_empleo_formal_pct": "Empleo formal %",
        "con_parroquia_pct": "Con parroquia %",
        "cluster_udla_cercano": "Cluster UDLA cercano",
        "similitud_centroidal": "Similitud centroidal",
    }

    return df[available].rename(columns=rename_map)


def _series_distribution(
    series: pd.Series, label_group: str, label_col: str, top_n: int | None = None
) -> pd.DataFrame:
    values = (
        pd.Series(series)
        .fillna("Sin dato")
        .astype(str)
        .str.strip()
        .replace("", "Sin dato")
    )
    counts = values.value_counts()
    if top_n is not None:
        counts = counts.head(top_n)
    total = float(counts.sum()) if float(counts.sum()) > 0 else 1.0
    df = counts.reset_index()
    df.columns = [label_col, "conteo"]
    df["participacion"] = df["conteo"] / total * 100.0
    df["grupo"] = label_group
    return df


def _compare_distribution_chart(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    column: str,
    left_label: str,
    right_label: str,
    title: str,
    top_n: int | None = None,
    height: int = 320,
) -> go.Figure:
    left = _series_distribution(left_df[column], left_label, "categoria", top_n=top_n)
    frames = [left]
    if not right_df.empty:
        right = _series_distribution(
            right_df[column], right_label, "categoria", top_n=top_n
        )
        frames.append(right)

    chart_df = pd.concat(frames, ignore_index=True)
    totals = (
        chart_df.groupby("categoria", as_index=False)["conteo"]
        .sum()
        .sort_values("conteo", ascending=False)
    )
    order = totals["categoria"].tolist()

    fig = px.bar(
        chart_df,
        x="categoria",
        y="participacion",
        color="grupo",
        barmode="group",
        category_orders={"categoria": order},
        text=chart_df["participacion"].map(lambda v: f"{v:.1f}%"),
        color_discrete_sequence=[COLOR_UNI, COLOR_UDLA],
        template=CHART_TEMPLATE,
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=height,
        margin=CHART_MARGIN,
        xaxis_title="",
        yaxis_title="Participacion (%)",
        legend_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig


def _top_category_chart(
    df: pd.DataFrame, column: str, title: str, top_n: int = 10, color: str = COLOR_UNI
) -> go.Figure:
    chart_df = _series_distribution(df[column], "Cluster", "categoria", top_n=top_n)
    fig = px.bar(
        chart_df,
        x="participacion",
        y="categoria",
        orientation="h",
        text=chart_df["conteo"].astype(int),
        template=CHART_TEMPLATE,
    )
    fig.update_traces(marker_color=color, textposition="outside")
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=max(200, 36 * len(chart_df) + 80),
        margin=CHART_MARGIN,
        xaxis_title="Participacion (%)",
        yaxis_title="",
        showlegend=False,
    )
    return fig


def _cluster_members_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    cols = [
        "hogar_id",
        "estudiantes_vinculados",
        "estudiantes_ids",
        "carrera",
        "carreras_hogar",
        "parroquia_estudiante",
        "parroquias_hogar",
        "sexo_estudiante",
        "estado_hogar",
        "quintil_institucion",
        "propensity_udla",
        "salario_hogar",
        "deuda_hogar",
        "ratio_deuda_ingreso",
        "hijos_hogar",
        "edad_estudiante_prom",
        "edad_padres_prom",
        "padres_presentes",
        "padres_con_empleo",
        "padres_con_superior",
        "primera_generacion",
        "hogar_huerfano",
        "riesgo_deuda_hogar",
    ]
    available = [c for c in cols if c in df.columns]
    out = df[available].copy()

    out["propensity_udla"] = out["propensity_udla"] * 100.0
    yes_no_cols = ["primera_generacion", "hogar_huerfano"]
    for column in yes_no_cols:
        if column in out.columns:
            out[column] = out[column].map({1: "Si", 0: "No"}).fillna("No")

    rename_map = {
        "hogar_id": "Hogar",
        "estudiantes_vinculados": "Estudiantes",
        "estudiantes_ids": "Ids estudiantes",
        "carrera": "Carrera",
        "carreras_hogar": "Carreras hogar",
        "parroquia_estudiante": "Parroquia",
        "parroquias_hogar": "Parroquias hogar",
        "sexo_estudiante": "Sexo dom.",
        "estado_hogar": "Estado hogar",
        "quintil_institucion": "Quintil",
        "propensity_udla": "Propensity (%)",
        "salario_hogar": "Salario",
        "deuda_hogar": "Deuda",
        "ratio_deuda_ingreso": "Ratio deuda/ingreso",
        "hijos_hogar": "Hijos",
        "edad_estudiante_prom": "Edad est.",
        "edad_padres_prom": "Edad padres",
        "padres_presentes": "Padres presentes",
        "padres_con_empleo": "Con empleo",
        "padres_con_superior": "Con superior",
        "primera_generacion": "1ra generacion",
        "hogar_huerfano": "Hogar huerfano",
        "riesgo_deuda_hogar": "Riesgo deuda",
    }
    return out.rename(columns=rename_map).sort_values(
        ["Propensity (%)", "Salario"], ascending=[False, False]
    )


def _radar_chart(
    uni_row: pd.Series,
    udla_row: pd.Series | None,
    uni_label: str,
) -> go.Figure:
    """Radar comparing a university cluster profile vs its nearest UDLA cluster."""
    indicators = {
        "Propensity prom.": "propensity_promedio",
        "Salario prom.": "salario_promedio",
        "1ra generacion %": "primera_generacion_pct",
        "Con deuda %": "con_deuda_pct",
        "Empleo formal %": "con_empleo_formal_pct",
        "Hijos prom.": "hijos_promedio",
    }

    def _safe(row: pd.Series, key: str) -> float:
        val = row.get(key, 0.0)
        return float(val) if pd.notna(val) else 0.0

    uni_vals = [_safe(uni_row, v) for v in indicators.values()]
    categories = list(indicators.keys())

    # Normalize to 0-100 for visual balance
    max_vals = [max(abs(v), 1.0) for v in uni_vals]
    if udla_row is not None and not udla_row.empty:
        udla_vals = [_safe(udla_row, v) for v in indicators.values()]
        max_vals = [max(abs(u), abs(d), 1.0) for u, d in zip(uni_vals, udla_vals)]
    else:
        udla_vals = None

    uni_norm = [v / m * 100.0 for v, m in zip(uni_vals, max_vals)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=uni_norm + [uni_norm[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name=uni_label,
        fillcolor=f"rgba(37, 99, 235, 0.15)",
        line=dict(color=COLOR_UNI, width=2),
    ))

    if udla_vals is not None:
        udla_norm = [v / m * 100.0 for v, m in zip(udla_vals, max_vals)]
        fig.add_trace(go.Scatterpolar(
            r=udla_norm + [udla_norm[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="UDLA",
            fillcolor=f"rgba(220, 38, 38, 0.10)",
            line=dict(color=COLOR_UDLA, width=2),
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 110], showticklabels=False)),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        height=340,
        margin=dict(l=60, r=60, t=30, b=40),
        template=CHART_TEMPLATE,
    )
    return fig


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown(
    "<h2 style='margin-bottom:0'>Propensity Score y Clusters por Universidad</h2>",
    unsafe_allow_html=True,
)
st.caption(
    "Estima la probabilidad de que un hogar pertenezca a UDLA dado su perfil "
    "socioeconomico. Los clusters se calculan independientemente por institucion y "
    "la unidad de analisis es el **hogar**."
)

# ---------------------------------------------------------------------------
# Enrollment type filter
# ---------------------------------------------------------------------------
TIPO_OPTIONS = ["Todas", "ENROLLMENT", "NEW ENROLLMENT"]
tipo_filtro = st.selectbox(
    "Tipo de alumno (UDLA)",
    options=TIPO_OPTIONS,
    index=0,
    key="tipo_alumno_filter",
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with st.spinner("Calculando propensity score y clusters..."):
    analysis = run_propensity_analysis(tipo_filtro=tipo_filtro)

overview = analysis["overview"]
details = analysis["details"]
base_df = analysis["base_df"]

if overview.empty:
    st.info("No hay universidades suficientes para ejecutar el analisis.")
    st.stop()

stable_options = overview.loc[overview["Lectura"] == "Estable", "Universidad"].tolist()
default_university = stable_options[0] if stable_options else overview["Universidad"].iloc[0]


# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tab_overview, tab_detail, tab_method = st.tabs([
    "Panorama general",
    "Detalle por universidad",
    "Metodologia",
])


# ───────────────────────────────────────────────────────────────────────────
# TAB 1 — Panorama general
# ───────────────────────────────────────────────────────────────────────────
with tab_overview:
    n_universities = int(len(overview))
    n_hogares_ext = int(base_df.loc[base_df["Universidad"] != "UDLA", "hogar_id"].nunique())
    n_hogares_udla = int(base_df.loc[base_df["Universidad"] == "UDLA", "hogar_id"].nunique())
    n_features = int(len(details[default_university]["feature_columns"]))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Universidades analizadas", n_universities)
    k2.metric("Hogares externos", f"{n_hogares_ext:,}")
    k3.metric("Hogares UDLA (ref.)", f"{n_hogares_udla:,}")
    k4.metric("Variables del modelo", n_features)

    st.markdown("")

    # ── Ranking chart ──
    sorted_overview = overview.sort_values("Propensity promedio hacia UDLA", ascending=True)
    bar_fig = px.bar(
        sorted_overview,
        x="Propensity promedio hacia UDLA",
        y="Universidad",
        orientation="h",
        color="Lectura",
        color_discrete_map={"Estable": COLOR_UNI, "Exploratoria": COLOR_WARN},
        hover_data={
            "Hogares": True,
            "Estudiantes vinculados": True,
            "Clusters": True,
            "Solapamiento con UDLA": ":.1f",
            "Propensity mediana": ":.1f",
        },
        template=CHART_TEMPLATE,
    )
    bar_fig.update_layout(
        title=dict(text="Propensity promedio hacia UDLA por universidad", font=dict(size=14)),
        height=max(300, 38 * n_universities + 80),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Propensity promedio (%)",
        yaxis_title="",
        legend_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(bar_fig, use_container_width=True)
    st.caption(
        ":blue[Azul] = muestra estable (>= 30 hogares). "
        ":orange[Naranja] = exploratoria (< 30 hogares). "
        "Mayor valor = perfil socioeconomico mas parecido al de UDLA."
    )

    # ── Overview table ──
    st.dataframe(
        overview,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Propensity promedio hacia UDLA": st.column_config.NumberColumn(
                "Propensity prom. (%)", format="%.2f%%"
            ),
            "Propensity mediana": st.column_config.NumberColumn(
                "Propensity med. (%)", format="%.2f%%"
            ),
            "Solapamiento con UDLA": st.column_config.NumberColumn(
                "Solapamiento (%)", format="%.2f%%"
            ),
            "Primera generacion": st.column_config.NumberColumn(
                "1ra generacion (%)", format="%.1f%%"
            ),
            "Hogares huerfanos": st.column_config.NumberColumn(
                "Hogares huerfanos (%)", format="%.1f%%"
            ),
            "Con deuda": st.column_config.NumberColumn(
                "Con deuda (%)", format="%.1f%%"
            ),
            "Lectura": st.column_config.TextColumn(
                "Confiabilidad",
                help="Estable: >= 30 hogares. Exploratoria: < 30 hogares.",
            ),
        },
    )


# ───────────────────────────────────────────────────────────────────────────
# TAB 2 — Detalle por universidad
# ───────────────────────────────────────────────────────────────────────────
with tab_detail:
    uni_options = ["UDLA"] + overview["Universidad"].tolist()
    selected_university = st.selectbox(
        "Selecciona una universidad",
        options=uni_options,
        index=0,
    )

    is_udla_standalone = selected_university == "UDLA"
    detail = details[selected_university]
    comparison = detail["comparison"]
    uni_clusters = detail["university_clusters"].copy()
    udla_clusters = detail["udla_clusters"].copy()
    cluster_counts = detail.get("cluster_counts", {})

    if is_udla_standalone:
        uni_df = comparison.copy()
        udla_df = pd.DataFrame()
        support_low, support_high = 0.0, 0.0

        n_hogares = int(uni_df["hogar_id"].nunique())
        n_estudiantes = int(uni_df["estudiantes_vinculados"].sum())
        n_huerfanos = int(uni_df["hogar_huerfano"].sum()) if "hogar_huerfano" in uni_df.columns else 0
        k_clusters = int(cluster_counts.get("UDLA", len(uni_clusters)))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Hogares", f"{n_hogares:,}")
        m2.metric("Estudiantes", f"{n_estudiantes:,}")
        m3.metric("Hogares huerfanos", f"{n_huerfanos:,}", help="Sin padre ni madre conocidos — agrupados aparte")
        m4.metric("Clusters", k_clusters, help="Excluyendo el grupo 'Sin dato familiar'")
    else:
        selected_row = overview[overview["Universidad"] == selected_university].iloc[0]
        uni_df = comparison[comparison["Universidad"] == selected_university].copy()
        udla_df = comparison[comparison["Universidad"] == "UDLA"].copy()
        support_low, support_high = detail["support_interval"]

        if detail["sample_status"] == "Exploratoria":
            st.warning(
                "Muestra pequena — los resultados son exploratorios y deben leerse con cautela."
            )

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Hogares", int(selected_row["Hogares"]))
        m2.metric("Estudiantes", int(selected_row["Estudiantes vinculados"]))
        m3.metric("Propensity prom.", f"{float(selected_row['Propensity promedio hacia UDLA']):.1f}%")
        m4.metric("Propensity med.", f"{float(selected_row['Propensity mediana']):.1f}%")
        m5.metric("Solapamiento", f"{float(selected_row['Solapamiento con UDLA']):.1f}%")
        m6.metric(
            "Clusters",
            f"{int(cluster_counts.get(selected_university, selected_row['Clusters']))} / "
            f"{int(cluster_counts.get('UDLA', len(udla_clusters)))}",
            help="Universidad / UDLA",
        )

    st.markdown("")

    # ── Build sub-tabs depending on mode ──
    if is_udla_standalone:
        sub_clusters, sub_deep = st.tabs([
            "Mapa de clusters",
            "Radiografia por cluster",
        ])
    else:
        sub_scores, sub_clusters, sub_deep = st.tabs([
            "Distribucion del score",
            "Mapa de clusters",
            "Radiografia por cluster",
        ])

        # ·· Sub-tab: score distribution (external universities only) ··
        with sub_scores:
            col_hist, col_box = st.columns([3, 2])

            with col_hist:
                dist_fig = go.Figure()
                dist_fig.add_trace(go.Histogram(
                    x=uni_df["propensity_udla"] * 100.0,
                    name=selected_university,
                    opacity=0.7,
                    histnorm="percent",
                    nbinsx=30,
                    marker_color=COLOR_UNI,
                ))
                dist_fig.add_trace(go.Histogram(
                    x=udla_df["propensity_udla"] * 100.0,
                    name="UDLA",
                    opacity=0.5,
                    histnorm="percent",
                    nbinsx=30,
                    marker_color=COLOR_UDLA,
                ))
                dist_fig.add_vrect(
                    x0=support_low * 100.0,
                    x1=support_high * 100.0,
                    fillcolor="#fbbf24",
                    opacity=0.10,
                    line_width=0,
                    annotation_text="Soporte UDLA 5%-95%",
                    annotation_position="top left",
                    annotation_font_size=10,
                )
                dist_fig.update_layout(
                    barmode="overlay",
                    title=dict(text="Distribucion del propensity score", font=dict(size=13)),
                    height=400,
                    margin=CHART_MARGIN,
                    xaxis_title="Propensity score (%)",
                    yaxis_title="Participacion (%)",
                    template=CHART_TEMPLATE,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(dist_fig, use_container_width=True)

            with col_box:
                box_data = pd.concat([
                    uni_df[["propensity_udla"]].assign(Grupo=selected_university),
                    udla_df[["propensity_udla"]].assign(Grupo="UDLA"),
                ])
                box_data["propensity_udla"] = box_data["propensity_udla"] * 100.0
                box_fig = px.box(
                    box_data,
                    x="Grupo",
                    y="propensity_udla",
                    color="Grupo",
                    color_discrete_map={selected_university: COLOR_UNI, "UDLA": COLOR_UDLA},
                    template=CHART_TEMPLATE,
                    points="outliers",
                )
                box_fig.update_layout(
                    title=dict(text="Comparacion de distribuciones", font=dict(size=13)),
                    height=400,
                    margin=CHART_MARGIN,
                    showlegend=False,
                    yaxis_title="Propensity score (%)",
                    xaxis_title="",
                )
                st.plotly_chart(box_fig, use_container_width=True)

            st.caption(
                "La zona amarilla marca el rango 5%-95% de UDLA. "
                "Mayor solapamiento entre las curvas indica mayor similitud socioeconomica."
            )

    # ·· Sub-tab: cluster map ··
    with sub_clusters:
        if is_udla_standalone:
            # UDLA standalone: scatter salario vs deuda, colored by cluster
            real_clusters = uni_clusters[uni_clusters["cluster"] != "Sin dato familiar"]
            if not real_clusters.empty:
                col_scatter_udla, col_profile = st.columns([3, 2])
                with col_scatter_udla:
                    scatter_fig = px.scatter(
                        real_clusters,
                        x="salario_promedio",
                        y="deuda_promedio",
                        size="hogares",
                        color="cluster",
                        hover_name="cluster",
                        text="cluster",
                        template=CHART_TEMPLATE,
                        size_max=50,
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    scatter_fig.update_traces(textposition="top center", textfont_size=10)
                    scatter_fig.update_layout(
                        title=dict(text="Clusters UDLA: salario vs deuda", font=dict(size=13)),
                        height=420,
                        margin=CHART_MARGIN,
                        xaxis_title="Salario promedio ($)",
                        yaxis_title="Deuda promedio ($)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(scatter_fig, use_container_width=True)

                with col_profile:
                    bar_profile = real_clusters.melt(
                        id_vars=["cluster"],
                        value_vars=["primera_generacion_pct", "con_deuda_pct", "con_empleo_formal_pct", "hogar_huerfano_pct"],
                        var_name="indicador",
                        value_name="valor",
                    )
                    label_map = {
                        "primera_generacion_pct": "1ra generacion",
                        "con_deuda_pct": "Con deuda",
                        "con_empleo_formal_pct": "Empleo formal",
                        "hogar_huerfano_pct": "Huerfanos",
                    }
                    bar_profile["indicador"] = bar_profile["indicador"].map(label_map)
                    profile_fig = px.bar(
                        bar_profile,
                        x="indicador",
                        y="valor",
                        color="cluster",
                        barmode="group",
                        text=bar_profile["valor"].map(lambda v: f"{v:.0f}%"),
                        template=CHART_TEMPLATE,
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    profile_fig.update_layout(
                        title=dict(text="Perfil social por cluster (%)", font=dict(size=13)),
                        height=420,
                        margin=CHART_MARGIN,
                        xaxis_title="",
                        yaxis_title="%",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    profile_fig.update_traces(textposition="outside", cliponaxis=False)
                    st.plotly_chart(profile_fig, use_container_width=True)

                st.caption(
                    "Se excluye el grupo 'Sin dato familiar' (hogares sin padres conocidos). "
                    "Tamano de burbuja = cantidad de hogares."
                )

            # Cluster table
            st.markdown("")
            st.markdown(
                f"**Clusters de UDLA** "
                f"(k={int(cluster_counts.get('UDLA', len(real_clusters)))} + grupo sin dato familiar)"
            )
            st.dataframe(
                _cluster_display(uni_clusters),
                use_container_width=True,
                hide_index=True,
            )
        else:
            # External university: comparison mode
            col_scatter, col_sim = st.columns([3, 2])

            with col_scatter:
                cluster_plot = pd.concat(
                    [
                        uni_clusters.assign(Institucion=selected_university),
                        udla_clusters.assign(Institucion="UDLA"),
                    ],
                    ignore_index=True,
                )
                cluster_fig = px.scatter(
                    cluster_plot,
                    x="salario_promedio",
                    y="propensity_promedio",
                    size="hogares",
                    color="Institucion",
                    hover_name="cluster",
                    text="cluster",
                    color_discrete_map={selected_university: COLOR_UNI, "UDLA": COLOR_UDLA},
                    template=CHART_TEMPLATE,
                    size_max=45,
                )
                cluster_fig.update_traces(textposition="top center", textfont_size=10)
                cluster_fig.update_layout(
                    title=dict(text="Clusters: salario vs propensity", font=dict(size=13)),
                    height=420,
                    margin=CHART_MARGIN,
                    xaxis_title="Salario promedio ($)",
                    yaxis_title="Propensity promedio (%)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(cluster_fig, use_container_width=True)

            with col_sim:
                if not uni_clusters.empty and "similitud_centroidal" in uni_clusters.columns:
                    sim_data = uni_clusters[uni_clusters["cluster"] != "Sin dato familiar"].copy()
                    if not sim_data.empty:
                        sim_fig = px.bar(
                            sim_data.sort_values("similitud_centroidal", ascending=True),
                            x="similitud_centroidal",
                            y="cluster",
                            orientation="h",
                            text=sim_data.sort_values("similitud_centroidal", ascending=True)[
                                "cluster_udla_cercano"
                            ].fillna("—"),
                            color="similitud_centroidal",
                            color_continuous_scale="Blues",
                            template=CHART_TEMPLATE,
                        )
                        sim_fig.update_traces(textposition="outside")
                        sim_fig.update_layout(
                            title=dict(text="Similitud con cluster UDLA cercano", font=dict(size=13)),
                            height=420,
                            margin=CHART_MARGIN,
                            xaxis_title="Similitud (0-100)",
                            yaxis_title="",
                            coloraxis_showscale=False,
                        )
                        st.plotly_chart(sim_fig, use_container_width=True)
                    else:
                        st.info("No hay datos de similitud disponibles.")
                else:
                    st.info("No hay datos de similitud disponibles.")

            st.caption(
                "Izquierda: cada burbuja es un cluster, tamano = hogares, posicion arriba-derecha = "
                "mayor salario y afinidad con UDLA. Derecha: similitud centroidal vs cluster UDLA "
                "mas cercano (0-100, donde 100 = identicos)."
            )

            # ── Cluster tables side by side ──
            st.markdown("")
            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown(
                    f"**Clusters de {selected_university}** "
                    f"(k={int(cluster_counts.get(selected_university, len(uni_clusters)))})"
                )
                st.dataframe(
                    _cluster_display(uni_clusters),
                    use_container_width=True,
                    hide_index=True,
                )
            with tc2:
                st.markdown(
                    f"**Clusters de UDLA** "
                    f"(k={int(cluster_counts.get('UDLA', len(udla_clusters)))})"
                )
                st.dataframe(
                    _cluster_display(udla_clusters),
                    use_container_width=True,
                    hide_index=True,
                )

    # ·· Sub-tab: deep dive per cluster ··
    with sub_deep:
        if uni_clusters.empty:
            st.info("No hay clusters disponibles para esta universidad.")
        else:
            cluster_names = uni_clusters["cluster"].tolist()
            cluster_tabs = st.tabs(cluster_names)

            for cluster_tab, (_, cluster_row) in zip(cluster_tabs, uni_clusters.iterrows()):
                cluster_name = str(cluster_row["cluster"])
                cluster_uni = uni_df[uni_df["cluster"] == cluster_name].copy()
                has_udla_match = not is_udla_standalone
                matched_udla_cluster = ""
                cluster_udla = uni_df.iloc[0:0].copy()

                if has_udla_match:
                    matched_udla_cluster = str(
                        cluster_row.get("cluster_udla_cercano", "") or ""
                    ).strip()
                    cluster_udla = (
                        udla_df[udla_df["cluster"] == matched_udla_cluster].copy()
                        if matched_udla_cluster
                        else udla_df.iloc[0:0].copy()
                    )

                with cluster_tab:
                    # ── Header with match info ──
                    if has_udla_match and matched_udla_cluster:
                        sim_val = cluster_row.get("similitud_centroidal", None)
                        sim_text = (
                            f" — similitud **{float(sim_val):.1f}**/100"
                            if pd.notna(sim_val)
                            else ""
                        )
                        st.markdown(
                            f"Cluster UDLA mas cercano: **{matched_udla_cluster}**{sim_text}"
                        )
                    elif is_udla_standalone and cluster_name == "Sin dato familiar":
                        st.info(
                            "Este grupo contiene hogares sin padre ni madre identificados. "
                            "Sus variables economicas y sociales son cero por falta de datos, "
                            "no por condicion real."
                        )

                    # ── KPI grid ──
                    salary_median = (
                        float(cluster_uni["salario_hogar"].median())
                        if not cluster_uni.empty
                        else 0.0
                    )
                    debt_median = (
                        float(cluster_uni["deuda_hogar"].median())
                        if not cluster_uni.empty
                        else 0.0
                    )
                    first_gen_pct = (
                        float(cluster_uni["primera_generacion"].mean() * 100.0)
                        if not cluster_uni.empty
                        else 0.0
                    )
                    orphan_pct = (
                        float(cluster_uni["hogar_huerfano"].mean() * 100.0)
                        if not cluster_uni.empty and "hogar_huerfano" in cluster_uni.columns
                        else 0.0
                    )
                    debt_pct = (
                        float(cluster_uni["hogar_con_deuda"].mean() * 100.0)
                        if not cluster_uni.empty
                        else 0.0
                    )

                    g1, g2, g3, g4, g5, g6 = st.columns(6)
                    g1.metric("Hogares", int(cluster_row["hogares"]))
                    g2.metric("Estudiantes", int(cluster_row["estudiantes_vinculados"]))
                    if not is_udla_standalone:
                        g3.metric(
                            "Propensity prom.",
                            f"{float(cluster_row['propensity_promedio']):.1f}%",
                        )
                    else:
                        g3.metric(
                            "Carrera dom.",
                            str(cluster_row.get("carrera_modal", "—")),
                        )
                    g4.metric("Salario mediano", f"${salary_median:,.0f}")
                    g5.metric("Deuda mediana", f"${debt_median:,.0f}")
                    g6.metric(
                        "Quintil dom.",
                        str(cluster_row.get("quintil_modal", "—")),
                    )

                    h1, h2, h3, h4 = st.columns(4)
                    h1.metric("1ra generacion", f"{first_gen_pct:.1f}%")
                    h2.metric("Hogares huerfanos", f"{orphan_pct:.1f}%")
                    h3.metric("Con deuda", f"{debt_pct:.1f}%")
                    h4.metric(
                        "Parroquia dom.",
                        str(cluster_row.get("parroquia_modal", "Sin dato")),
                    )

                    st.markdown("")

                    # ── Charts row ──
                    if has_udla_match:
                        col_radar, col_quintil, col_estado = st.columns([1, 1, 1])

                        with col_radar:
                            matched_udla_row = None
                            if matched_udla_cluster and not udla_clusters.empty:
                                match_mask = udla_clusters["cluster"] == matched_udla_cluster
                                if match_mask.any():
                                    matched_udla_row = udla_clusters[match_mask].iloc[0]
                            st.plotly_chart(
                                _radar_chart(cluster_row, matched_udla_row, cluster_name),
                                use_container_width=True,
                            )

                        with col_quintil:
                            st.plotly_chart(
                                _compare_distribution_chart(
                                    cluster_uni,
                                    cluster_udla,
                                    column="quintil_institucion",
                                    left_label=cluster_name,
                                    right_label=matched_udla_cluster or "UDLA",
                                    title="Quintiles",
                                    height=340,
                                ),
                                use_container_width=True,
                            )

                        with col_estado:
                            st.plotly_chart(
                                _compare_distribution_chart(
                                    cluster_uni,
                                    cluster_udla,
                                    column="estado_hogar",
                                    left_label=cluster_name,
                                    right_label=matched_udla_cluster or "UDLA",
                                    title="Estado del hogar",
                                    top_n=6,
                                    height=340,
                                ),
                                use_container_width=True,
                            )
                    else:
                        # UDLA standalone: no comparison, show solo distributions
                        col_quintil, col_estado = st.columns(2)
                        with col_quintil:
                            st.plotly_chart(
                                _top_category_chart(
                                    cluster_uni,
                                    column="quintil_institucion",
                                    title="Distribucion por quintil",
                                    top_n=6,
                                    color=COLOR_UDLA,
                                ),
                                use_container_width=True,
                            )
                        with col_estado:
                            st.plotly_chart(
                                _top_category_chart(
                                    cluster_uni,
                                    column="estado_hogar",
                                    title="Estado del hogar",
                                    top_n=6,
                                    color=COLOR_UDLA,
                                ),
                                use_container_width=True,
                            )

                    # ── Top categories ──
                    cat1, cat2 = st.columns(2)
                    with cat1:
                        st.plotly_chart(
                            _top_category_chart(
                                cluster_uni,
                                column="carrera",
                                title="Top carreras",
                                top_n=8,
                                color=COLOR_UNI if not is_udla_standalone else COLOR_UDLA,
                            ),
                            use_container_width=True,
                        )
                    with cat2:
                        st.plotly_chart(
                            _top_category_chart(
                                cluster_uni,
                                column="parroquia_estudiante",
                                title="Top parroquias",
                                top_n=8,
                                color=COLOR_ACCENT,
                            ),
                            use_container_width=True,
                        )

                    # ── Member table ──
                    with st.expander("Ver registros del cluster", expanded=False):
                        st.dataframe(
                            _cluster_members_table(cluster_uni),
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Propensity (%)": st.column_config.NumberColumn(
                                    "Propensity (%)", format="%.1f%%"
                                ),
                                "Salario": st.column_config.NumberColumn(
                                    "Salario", format="$ %,.0f"
                                ),
                                "Deuda": st.column_config.NumberColumn(
                                    "Deuda", format="$ %,.0f"
                                ),
                                "Ratio deuda/ingreso": st.column_config.NumberColumn(
                                    "Ratio D/I", format="%.2f"
                                ),
                            },
                        )


# ───────────────────────────────────────────────────────────────────────────
# TAB 3 — Metodologia
# ───────────────────────────────────────────────────────────────────────────
with tab_method:
    st.markdown("### Como se construye el analisis")
    st.markdown(
        "Se usan simultaneamente `db/Udla.xlsx` y `db/Universidades.xlsx`. "
        "Primero se construye un perfil enriquecido por estudiante y luego se agrega "
        "al nivel de **hogar** (`hogar_id`). El propensity score y los clusters se "
        "calculan sobre los hogares."
    )

    st.markdown("#### Fuentes de datos")
    methodology = pd.DataFrame(
        [
            {
                "Fuente": "Estudiantes",
                "Variables derivadas": "universidad, carrera, sexo estudiante",
            },
            {
                "Fuente": "Universo Familiares",
                "Variables derivadas": "hogar_id, padres presentes, hogar huerfano, puente estudiante-hogar",
            },
            {
                "Fuente": "Informacion Personal",
                "Variables derivadas": "edad estudiante, edad padres, estado hogar, hijos, nivel educativo, parroquia",
            },
            {
                "Fuente": "Empleos",
                "Variables derivadas": "salario hogar, empleo formal, quintiles por institucion",
            },
            {
                "Fuente": "Deudas",
                "Variables derivadas": "deuda hogar, riesgo de deuda, entidades, ratio deuda/ingreso",
            },
        ]
    )
    st.dataframe(methodology, use_container_width=True, hide_index=True)

    st.markdown("#### Variables del modelo")
    st.markdown(
        "El propensity score usa regresion logistica regularizada (L2) con las "
        "siguientes variables a nivel de hogar:"
    )
    var_cols = st.columns(3)
    var_groups = {
        "Economicas": [
            "Salario del hogar (log)",
            "Deuda del hogar (log)",
            "Ratio deuda/ingreso",
            "Riesgo de deuda (score)",
        ],
        "Demograficas": [
            "Estudiantes vinculados",
            "Edad promedio estudiantes",
            "% mujeres estudiantes",
            "Hijos del hogar",
            "Edad promedio padres",
            "Estado civil del hogar",
        ],
        "Sociales": [
            "Padres presentes",
            "Padres con empleo",
            "Padres con educacion superior",
            "Primera generacion",
            "Hogar huerfano (sin padres conocidos)",
        ],
    }
    for col, (group_name, variables) in zip(var_cols, var_groups.items()):
        with col:
            st.markdown(f"**{group_name}**")
            for v in variables:
                st.markdown(f"- {v}")

    st.markdown("#### Clustering")
    st.markdown(
        "Los clusters se calculan con K-Means **por institucion** (sin mezclar). "
        "El numero de clusters (k) se selecciona automaticamente usando el indice de "
        "Calinski-Harabasz, buscando buena separacion entre grupos sin fragmentar "
        "muestras pequenas."
    )
    st.info(
        "Los clusters de UDLA cambian segun la universidad seleccionada, ya que la "
        "estandarizacion se recalcula para cada par de comparacion. Esto es correcto "
        "metodologicamente para comparaciones pairwise."
    )

    # ── Quintile ranges ──
    if details:
        first_detail = details.get(selected_university if selected_university in details else default_university)
        if first_detail:
            st.markdown("#### Rangos de quintiles")
            sel_uni = selected_university if selected_university in details else default_university
            range_frame = pd.concat(
                [
                    _ranges_frame(sel_uni, first_detail["quintile_ranges"].get(sel_uni, {})),
                    _ranges_frame("UDLA", first_detail["quintile_ranges"].get("UDLA", {})),
                ],
                ignore_index=True,
            )
            st.dataframe(
                range_frame,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Ingreso minimo": st.column_config.NumberColumn(
                        "Ingreso minimo", format="$ %.2f"
                    ),
                    "Ingreso maximo": st.column_config.NumberColumn(
                        "Ingreso maximo", format="$ %.2f"
                    ),
                },
            )
