"""Visualization page for the Learning Analytics Platform."""

from dash import html, dcc, Input, Output, State, no_update
import pandas as pd
import numpy as np
import json
import io
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLORS = ["#2563eb", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"]
CHART_CONFIG = {"displayModeBar": True, "responsive": True}


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout():
    """Return the visualization page layout."""

    return html.Div(
        [
            dcc.Tabs(
                id="viz-tabs",
                value="distribution",
                children=[
                    # ---- Tab 1: Distribution ----
                    dcc.Tab(
                        label="Distribution",
                        value="distribution",
                        children=[
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                "Distribution Analysis",
                                                className="card-title",
                                            ),
                                        ],
                                        className="card-header",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Column",
                                                        className="form-label",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="dist-column",
                                                        placeholder="Select column...",
                                                    ),
                                                ],
                                                className="form-group",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Bins",
                                                        className="form-label",
                                                    ),
                                                    dcc.Slider(
                                                        id="dist-bins",
                                                        min=10,
                                                        max=100,
                                                        step=1,
                                                        value=30,
                                                        marks={
                                                            10: "10",
                                                            30: "30",
                                                            50: "50",
                                                            75: "75",
                                                            100: "100",
                                                        },
                                                        tooltip={
                                                            "placement": "bottom",
                                                            "always_visible": False,
                                                        },
                                                    ),
                                                ],
                                                className="form-group",
                                            ),
                                        ],
                                        className="grid-2",
                                    ),
                                    dcc.Graph(
                                        id="dist-chart",
                                        config=CHART_CONFIG,
                                    ),
                                ],
                                className="card",
                            ),
                        ],
                    ),
                    # ---- Tab 2: Correlation ----
                    dcc.Tab(
                        label="Correlation",
                        value="correlation",
                        children=[
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                "Correlation Analysis",
                                                className="card-title",
                                            ),
                                        ],
                                        className="card-header",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Method",
                                                        className="form-label",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="corr-method",
                                                        options=[
                                                            {
                                                                "label": "Pearson",
                                                                "value": "pearson",
                                                            },
                                                            {
                                                                "label": "Spearman",
                                                                "value": "spearman",
                                                            },
                                                            {
                                                                "label": "Kendall",
                                                                "value": "kendall",
                                                            },
                                                        ],
                                                        value="pearson",
                                                        clearable=False,
                                                    ),
                                                ],
                                                className="form-group",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Columns",
                                                        className="form-label",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="corr-columns",
                                                        multi=True,
                                                        placeholder="Select columns...",
                                                    ),
                                                ],
                                                className="form-group",
                                            ),
                                        ],
                                        className="grid-2",
                                    ),
                                    dcc.Graph(
                                        id="corr-chart",
                                        config=CHART_CONFIG,
                                    ),
                                ],
                                className="card",
                            ),
                        ],
                    ),
                    # ---- Tab 3: Clustering ----
                    dcc.Tab(
                        label="Clustering",
                        value="clustering",
                        children=[
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                "Clustering Visualization",
                                                className="card-title",
                                            ),
                                        ],
                                        className="card-header",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "X Axis",
                                                        className="form-label",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="cluster-x",
                                                        placeholder="Select X axis...",
                                                    ),
                                                ],
                                                className="form-group",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Y Axis",
                                                        className="form-label",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="cluster-y",
                                                        placeholder="Select Y axis...",
                                                    ),
                                                ],
                                                className="form-group",
                                            ),
                                        ],
                                        className="grid-2",
                                    ),
                                    html.Div(
                                        id="cluster-info",
                                        style={
                                            "padding": "8px 0",
                                            "fontSize": "14px",
                                            "color": "#64748b",
                                        },
                                    ),
                                    dcc.Graph(
                                        id="cluster-chart",
                                        config=CHART_CONFIG,
                                    ),
                                ],
                                className="card",
                            ),
                        ],
                    ),
                    # ---- Tab 4: Time Series ----
                    dcc.Tab(
                        label="Time Series",
                        value="timeseries",
                        children=[
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                "Time Series Analysis",
                                                className="card-title",
                                            ),
                                        ],
                                        className="card-header",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Time Column",
                                                        className="form-label",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="ts-time-col",
                                                        placeholder="Select time column...",
                                                    ),
                                                ],
                                                className="form-group",
                                            ),
                                            html.Div(
                                                [
                                                    html.Label(
                                                        "Value Columns",
                                                        className="form-label",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="ts-value-cols",
                                                        multi=True,
                                                        placeholder="Select value columns...",
                                                    ),
                                                ],
                                                className="form-group",
                                            ),
                                        ],
                                        className="grid-2",
                                    ),
                                    dcc.Graph(
                                        id="ts-chart",
                                        config=CHART_CONFIG,
                                    ),
                                ],
                                className="card",
                            ),
                        ],
                    ),
                ],
            ),
        ],
        className="page-content",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_figure(message="No data available"):
    """Return a blank figure with a centered annotation."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 16, "color": "#94a3b8"},
            }
        ],
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def _read_data(data):
    """Parse JSON data from a dcc.Store into a DataFrame."""
    if not data:
        return None
    try:
        if isinstance(data, str):
            return pd.read_json(io.StringIO(data), orient="split")
        if isinstance(data, dict):
            return pd.read_json(io.StringIO(json.dumps(data)), orient="split")
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def register_callbacks(app):
    """Register Dash callbacks for the visualization page."""

    # ------------------------------------------------------------------
    # Populate dropdowns when data store changes
    # ------------------------------------------------------------------
    @app.callback(
        [
            Output("dist-column", "options"),
            Output("corr-columns", "options"),
            Output("cluster-x", "options"),
            Output("cluster-y", "options"),
            Output("ts-time-col", "options"),
            Output("ts-value-cols", "options"),
        ],
        [Input("featured-data-store", "data")],
    )
    def populate_dropdowns(data):
        df = _read_data(data)
        if df is None:
            empty = []
            return empty, empty, empty, empty, empty, empty

        all_options = [
            {"label": col, "value": col} for col in df.columns
        ]
        numeric_options = [
            {"label": col, "value": col}
            for col in df.select_dtypes(include="number").columns
        ]

        return (
            numeric_options,   # dist-column
            numeric_options,   # corr-columns
            numeric_options,   # cluster-x
            numeric_options,   # cluster-y
            all_options,       # ts-time-col
            numeric_options,   # ts-value-cols
        )

    # ------------------------------------------------------------------
    # Tab 1 - Distribution histogram
    # ------------------------------------------------------------------
    @app.callback(
        Output("dist-chart", "figure"),
        [
            Input("dist-column", "value"),
            Input("dist-bins", "value"),
        ],
        [State("featured-data-store", "data")],
    )
    def update_distribution(column, bins, data):
        df = _read_data(data)
        if df is None or not column:
            return _empty_figure("Select a column to view its distribution")

        fig = go.Figure(
            data=[
                go.Histogram(
                    x=df[column],
                    nbinsx=bins,
                    marker_color=COLORS[0],
                    opacity=0.85,
                )
            ]
        )
        fig.update_layout(
            template="plotly_white",
            title=f"Distribution of {column}",
            xaxis_title=column,
            yaxis_title="Count",
            bargap=0.05,
        )
        return fig

    # ------------------------------------------------------------------
    # Tab 2 - Correlation heatmap
    # ------------------------------------------------------------------
    @app.callback(
        Output("corr-chart", "figure"),
        [
            Input("corr-method", "value"),
            Input("corr-columns", "value"),
        ],
        [State("featured-data-store", "data")],
    )
    def update_correlation(method, columns, data):
        df = _read_data(data)
        if df is None or not columns or len(columns) < 2:
            return _empty_figure(
                "Select at least two columns to view correlation"
            )

        corr_matrix = df[columns].corr(method=method)

        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns.tolist(),
                    y=corr_matrix.index.tolist(),
                    colorscale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate="%{text}",
                    hovertemplate=(
                        "%{x} vs %{y}<br>Correlation: %{z:.3f}"
                        "<extra></extra>"
                    ),
                )
            ]
        )
        fig.update_layout(
            template="plotly_white",
            title=f"Correlation Matrix ({method.title()})",
            xaxis_title="",
            yaxis_title="",
        )
        return fig

    # ------------------------------------------------------------------
    # Tab 3 - Clustering scatter plot
    # ------------------------------------------------------------------
    @app.callback(
        [
            Output("cluster-chart", "figure"),
            Output("cluster-info", "children"),
        ],
        [
            Input("cluster-x", "value"),
            Input("cluster-y", "value"),
        ],
        [
            State("featured-data-store", "data"),
            State("analytics-results-store", "data"),
        ],
    )
    def update_clustering(x_col, y_col, data, analytics_data):
        df = _read_data(data)
        if df is None or not x_col or not y_col:
            return (
                _empty_figure("Select X and Y axes to view clusters"),
                "",
            )

        # Retrieve cluster labels from analytics results store
        labels = None
        if analytics_data:
            try:
                results = (
                    json.loads(analytics_data)
                    if isinstance(analytics_data, str)
                    else analytics_data
                )
                labels = results.get("cluster_labels")
            except Exception:
                pass

        if labels is not None:
            labels = np.array(labels)
        else:
            labels = np.zeros(len(df), dtype=int)

        unique_labels = sorted(set(labels))
        n_clusters = len(unique_labels)
        info_text = f"{n_clusters} cluster(s) detected"

        fig = go.Figure()
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            color = COLORS[idx % len(COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=df.loc[mask, x_col],
                    y=df.loc[mask, y_col],
                    mode="markers",
                    name=f"Cluster {label}",
                    marker=dict(
                        color=color,
                        size=8,
                        opacity=0.7,
                    ),
                    hovertemplate=(
                        f"{x_col}: %{{x}}<br>"
                        f"{y_col}: %{{y}}<br>"
                        f"Cluster: {label}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            template="plotly_white",
            title=f"Clustering: {x_col} vs {y_col}",
            xaxis_title=x_col,
            yaxis_title=y_col,
            legend_title="Cluster",
        )
        return fig, info_text

    # ------------------------------------------------------------------
    # Tab 4 - Time series line plot
    # ------------------------------------------------------------------
    @app.callback(
        Output("ts-chart", "figure"),
        [
            Input("ts-time-col", "value"),
            Input("ts-value-cols", "value"),
        ],
        [State("featured-data-store", "data")],
    )
    def update_timeseries(time_col, value_cols, data):
        df = _read_data(data)
        if df is None or not time_col or not value_cols:
            return _empty_figure(
                "Select a time column and value columns to view trends"
            )

        # Sort by the time column
        df_sorted = df.sort_values(time_col)

        fig = go.Figure()
        for idx, col in enumerate(value_cols):
            color = COLORS[idx % len(COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=df_sorted[time_col],
                    y=df_sorted[col],
                    mode="lines",
                    name=col,
                    line=dict(color=color, width=2),
                    hovertemplate=(
                        f"{time_col}: %{{x}}<br>"
                        f"{col}: %{{y}}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            template="plotly_white",
            title="Time Series",
            xaxis_title=time_col,
            yaxis_title="Value",
            legend_title="Columns",
            hovermode="x unified",
        )
        return fig
