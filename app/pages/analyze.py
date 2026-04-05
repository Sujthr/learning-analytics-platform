"""Analysis page – EDA, hypothesis testing, and machine learning."""

import io
import json
import traceback
from pathlib import Path

from dash import dash_table
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from dash import Input, Output, State, callback_context, dcc, html, no_update

from src.analytics.analyzer import AnalyticsEngine

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_CFG_PATH = Path(__file__).parent.parent.parent / "config" / "pipeline_config.yaml"
with open(_CFG_PATH) as _f:
    CONFIG = yaml.safe_load(_f)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REGRESSION_MODELS = [
    {"label": "Linear Regression", "value": "linear"},
    {"label": "Ridge Regression", "value": "ridge"},
    {"label": "Lasso Regression", "value": "lasso"},
    {"label": "Random Forest", "value": "random_forest"},
    {"label": "Gradient Boosting", "value": "gradient_boosting"},
]

_CLASSIFICATION_MODELS = [
    {"label": "Logistic Regression", "value": "logistic"},
    {"label": "Random Forest", "value": "random_forest"},
    {"label": "Gradient Boosting", "value": "gradient_boosting"},
    {"label": "SVM", "value": "svm"},
]

_CLUSTERING_MODELS = [
    {"label": "K-Means", "value": "kmeans"},
    {"label": "DBSCAN", "value": "dbscan"},
    {"label": "Hierarchical", "value": "hierarchical"},
]


def _read_store(data):
    """Deserialise a dcc.Store JSON payload to a DataFrame."""
    if data is None:
        return None
    if isinstance(data, str):
        return pd.read_json(io.StringIO(data), orient="split")
    if isinstance(data, dict):
        return pd.read_json(io.StringIO(json.dumps(data)), orient="split")
    return None


def _col_options(df):
    """Return dropdown options from DataFrame columns."""
    if df is None:
        return []
    return [{"label": c, "value": c} for c in df.columns]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout():
    """Return the analysis page layout."""
    return html.Div(
        className="grid-sidebar",
        children=[
            # ---- Left panel (controls) ----
            html.Div(
                style={"width": "320px", "flexShrink": 0},
                children=[
                    # -- EDA Section --
                    html.Div(
                        className="card",
                        children=[
                            html.Div("Exploratory Data Analysis", className="card-header card-title"),
                            html.Div([
                                html.Div([
                                    html.Label("Correlation Method", className="form-label"),
                                    dcc.Dropdown(
                                        id="eda-corr-method",
                                        options=[
                                            {"label": "Pearson", "value": "pearson"},
                                            {"label": "Spearman", "value": "spearman"},
                                            {"label": "Kendall", "value": "kendall"},
                                        ],
                                        value="pearson",
                                        clearable=False,
                                    ),
                                ], className="form-group"),
                                html.Button("Run EDA", id="btn-eda", className="btn btn-primary", title="Compute distributions & correlation matrix"),
                            ]),
                        ],
                    ),

                    # -- Hypothesis Testing Section --
                    html.Div(
                        className="card",
                        style={"marginTop": "16px"},
                        children=[
                            html.Div("Hypothesis Testing", className="card-header card-title"),
                            html.Div([
                                html.Div([
                                    html.Label("Test Type", className="form-label"),
                                    dcc.Dropdown(
                                        id="test-type",
                                        options=[
                                            {"label": "T-Test", "value": "t_test"},
                                            {"label": "ANOVA", "value": "anova"},
                                        ],
                                        value="t_test",
                                        clearable=False,
                                    ),
                                ], className="form-group"),
                                html.Div([
                                    html.Label("Target Column", className="form-label"),
                                    dcc.Dropdown(id="test-target-col", placeholder="Select target column"),
                                ], className="form-group"),
                                html.Div([
                                    html.Label("Group Column", className="form-label"),
                                    dcc.Dropdown(id="test-group-col", placeholder="Select group column"),
                                ], className="form-group"),
                                html.Div([
                                    html.Label("Significance Level", className="form-label"),
                                    dcc.Input(
                                        id="test-alpha",
                                        type="number",
                                        value=0.05,
                                        step=0.01,
                                        min=0.001,
                                        max=0.5,
                                        style={"width": "100%"},
                                    ),
                                ], className="form-group"),
                                html.Button("Run Test", id="btn-test", className="btn btn-primary", title="Run t-test or ANOVA for significance testing"),
                            ]),
                        ],
                    ),

                    # -- Machine Learning Section --
                    html.Div(
                        className="card",
                        style={"marginTop": "16px"},
                        children=[
                            html.Div("Machine Learning", className="card-header card-title"),
                            html.Div([
                                html.Div([
                                    html.Label("Task", className="form-label"),
                                    dcc.Dropdown(
                                        id="ml-task",
                                        options=[
                                            {"label": "Regression", "value": "regression"},
                                            {"label": "Classification", "value": "classification"},
                                            {"label": "Clustering", "value": "clustering"},
                                        ],
                                        value="regression",
                                        clearable=False,
                                    ),
                                ], className="form-group"),
                                html.Div([
                                    html.Label("Target Column", className="form-label"),
                                    dcc.Dropdown(id="ml-target", placeholder="Select target column"),
                                ], className="form-group"),
                                html.Div([
                                    html.Label("Model Type", className="form-label"),
                                    dcc.Dropdown(id="ml-model", clearable=False),
                                ], className="form-group"),
                                html.Div(
                                    id="ml-clusters-wrapper",
                                    children=[
                                        html.Label("Number of Clusters", className="form-label"),
                                        dcc.Input(
                                            id="ml-clusters",
                                            type="number",
                                            value=4,
                                            min=2,
                                            max=20,
                                            step=1,
                                            style={"width": "100%"},
                                        ),
                                    ],
                                    className="form-group",
                                    style={"display": "none"},
                                ),
                                html.Div([
                                    html.Label("Test Size (%)", className="form-label"),
                                    dcc.Slider(
                                        id="ml-test-size",
                                        min=10,
                                        max=50,
                                        step=5,
                                        value=20,
                                        marks={v: f"{v}%" for v in range(10, 55, 10)},
                                    ),
                                ], className="form-group"),
                                html.Button("Train Model", id="btn-ml", className="btn btn-success", title="Train regression, classification or clustering model"),
                            ]),
                        ],
                    ),
                ],
            ),

            # ---- Right panel (results) ----
            html.Div(
                id="analysis-results",
                style={"flex": 1, "minWidth": 0, "paddingLeft": "24px"},
                children=[
                    html.Div(
                        "Select an analysis type and click Run to see results.",
                        className="alert-info",
                    )
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def register_callbacks(app):
    """Register all Dash callbacks for the analysis page."""

    # ---- Dynamic dropdown: populate test columns ----
    @app.callback(
        [Output("test-target-col", "options"),
         Output("test-group-col", "options")],
        [Input("featured-data-store", "data"),
         Input("integrated-data-store", "data")],
    )
    def _update_test_columns(feat_data, integ_data):
        data = feat_data or integ_data
        df = _read_store(data)
        if df is None:
            return [], []
        numeric_opts = [{"label": c, "value": c}
                        for c in df.select_dtypes(include=["number"]).columns]
        all_opts = _col_options(df)
        return numeric_opts, all_opts

    # ---- Dynamic dropdown: populate ML target column ----
    @app.callback(
        Output("ml-target", "options"),
        [Input("featured-data-store", "data"),
         Input("integrated-data-store", "data")],
    )
    def _update_ml_target(feat_data, integ_data):
        data = feat_data or integ_data
        df = _read_store(data)
        if df is None:
            return []
        return _col_options(df)

    # ---- Dynamic dropdown: update model choices based on task ----
    @app.callback(
        [Output("ml-model", "options"),
         Output("ml-model", "value"),
         Output("ml-clusters-wrapper", "style"),
         Output("ml-target", "style")],
        Input("ml-task", "value"),
    )
    def _update_ml_model_options(task):
        hide = {"display": "none"}
        show = {"display": "block"}
        show_full = {"width": "100%"}
        if task == "regression":
            return _REGRESSION_MODELS, "random_forest", hide, show_full
        elif task == "classification":
            return _CLASSIFICATION_MODELS, "random_forest", hide, show_full
        else:  # clustering
            return _CLUSTERING_MODELS, "kmeans", show, hide

    # ---- EDA callback ----
    @app.callback(
        [Output("analysis-results", "children", allow_duplicate=True),
         Output("analytics-results-store", "data", allow_duplicate=True)],
        Input("btn-eda", "n_clicks"),
        [State("eda-corr-method", "value"),
         State("featured-data-store", "data"),
         State("integrated-data-store", "data"),
         State("analytics-results-store", "data")],
        prevent_initial_call=True,
    )
    def _run_eda(n_clicks, corr_method, feat_data, integ_data, existing_results):
        if not n_clicks:
            return no_update, no_update

        data = feat_data or integ_data
        if data is None:
            return html.Div("No data available. Please load and process data first.",
                            className="alert-danger"), no_update

        try:
            df = _read_store(data)
            engine = AnalyticsEngine(CONFIG.get("analytics", {}))

            # Distribution statistics
            dist_stats = engine.compute_distributions(df)
            dist_rows = []
            for col, stats_dict in dist_stats.items():
                row = {"Column": col}
                row.update({k: round(v, 4) if isinstance(v, float) else v
                            for k, v in stats_dict.items()})
                dist_rows.append(row)

            dist_table = dash_table.DataTable(
                data=dist_rows,
                columns=[{"name": c, "id": c} for c in dist_rows[0].keys()] if dist_rows else [],
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "center", "padding": "8px"},
                style_header={"fontWeight": "bold"},
                page_size=10,
            ) if dist_rows else html.Div("No numeric columns found.", className="alert-info")

            # Correlation matrix heatmap
            corr_matrix = engine.compute_correlations(df, method=corr_method)
            heatmap_fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.index.tolist(),
                colorscale="RdBu_r",
                zmin=-1,
                zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                hovertemplate="(%{x}, %{y}): %{z:.3f}<extra></extra>",
            ))
            heatmap_fig.update_layout(
                title=f"{corr_method.capitalize()} Correlation Matrix",
                height=500,
                template="plotly_white",
            )

            # Store results for reports/visualize pages
            store = existing_results if isinstance(existing_results, dict) else {}
            store["distributions"] = dist_stats
            store["correlations"] = corr_matrix.to_dict()

            return html.Div([
                html.Div([
                    html.Div("Distribution Statistics", className="card-header card-title"),
                    dist_table,
                ], className="card"),
                html.Div([
                    html.Div("Correlation Matrix", className="card-header card-title"),
                    dcc.Graph(figure=heatmap_fig),
                ], className="card", style={"marginTop": "16px"}),
            ]), store

        except Exception as exc:
            return html.Div(f"EDA Error: {exc}", className="alert-danger"), no_update

    # ---- Hypothesis testing callback ----
    @app.callback(
        [Output("analysis-results", "children", allow_duplicate=True),
         Output("analytics-results-store", "data", allow_duplicate=True)],
        Input("btn-test", "n_clicks"),
        [State("test-type", "value"),
         State("test-target-col", "value"),
         State("test-group-col", "value"),
         State("test-alpha", "value"),
         State("featured-data-store", "data"),
         State("integrated-data-store", "data"),
         State("analytics-results-store", "data")],
        prevent_initial_call=True,
    )
    def _run_test(n_clicks, test_type, target_col, group_col, alpha,
                  feat_data, integ_data, existing_results):
        if not n_clicks:
            return no_update, no_update

        data = feat_data or integ_data
        if data is None:
            return html.Div("No data available.", className="alert-danger"), no_update
        if not target_col or not group_col:
            return html.Div("Please select both a target and group column.",
                            className="alert-danger"), no_update

        try:
            df = _read_store(data)
            cfg = CONFIG.get("analytics", {})
            cfg.setdefault("hypothesis_testing", {})["significance_level"] = alpha or 0.05
            engine = AnalyticsEngine(cfg)

            if test_type == "t_test":
                # Drop NaN in both target and group columns for clean comparison
                df_clean = df[[target_col, group_col]].dropna()
                groups = df_clean[group_col].unique()
                if len(groups) < 2:
                    return html.Div("Need at least 2 groups for a T-Test.",
                                    className="alert-danger"), no_update
                result = engine.t_test(df_clean, target_col, group_col,
                                       groups[0], groups[1])
                test_name = "Independent Samples T-Test"
                stat_label = "t-statistic"
                stat_value = result["t_statistic"]
            else:  # anova
                df_clean = df[[target_col, group_col]].dropna()
                result = engine.anova(df_clean, target_col, group_col)
                if "error" in result:
                    return html.Div(f"ANOVA Error: {result['error']}",
                                    className="alert-danger"), no_update
                test_name = "One-Way ANOVA"
                stat_label = "F-statistic"
                stat_value = result["f_statistic"]

            sig_class = "alert-success" if result["significant"] else "alert-info"
            sig_text = ("Statistically significant"
                        if result["significant"]
                        else "Not statistically significant")

            # Store results for reports
            store = existing_results if isinstance(existing_results, dict) else {}
            # Remove non-serializable items
            clean_result = {k: v for k, v in result.items() if not callable(v)}
            store_key = "t_test_engagement" if test_type == "t_test" else "anova_engagement"
            store[store_key] = clean_result

            return html.Div([
                html.Div([
                    html.Div(test_name, className="card-header card-title"),
                    html.Div([
                        html.Div([
                            html.Div(stat_label, className="result-metric-label"),
                            html.Div(f"{stat_value:.4f}", className="result-metric-value"),
                        ], className="result-metric"),
                        html.Div([
                            html.Div("p-value", className="result-metric-label"),
                            html.Div(f"{result['p_value']:.6f}", className="result-metric-value"),
                        ], className="result-metric"),
                        html.Div([
                            html.Div("Alpha", className="result-metric-label"),
                            html.Div(f"{result['alpha']}", className="result-metric-value"),
                        ], className="result-metric"),
                    ], style={"display": "flex", "gap": "24px", "padding": "16px"}),
                    html.Div(sig_text, className=sig_class,
                             style={"margin": "16px"}),
                ], className="card"),
            ]), store

        except Exception as exc:
            return html.Div(f"Hypothesis Test Error: {exc}",
                            className="alert-danger"), no_update

    # ---- Machine Learning callback ----
    @app.callback(
        [Output("analysis-results", "children", allow_duplicate=True),
         Output("analytics-results-store", "data", allow_duplicate=True)],
        Input("btn-ml", "n_clicks"),
        [State("ml-task", "value"),
         State("ml-target", "value"),
         State("ml-model", "value"),
         State("ml-clusters", "value"),
         State("ml-test-size", "value"),
         State("featured-data-store", "data"),
         State("integrated-data-store", "data"),
         State("analytics-results-store", "data")],
        prevent_initial_call=True,
    )
    def _run_ml(n_clicks, task, target, model_type, n_clusters, test_size_pct,
                feat_data, integ_data, existing_results):
        if not n_clicks:
            return no_update, no_update

        data = feat_data or integ_data
        if data is None:
            return html.Div("No data available. Please load and process data first.",
                            className="alert-danger"), no_update

        try:
            df = _read_store(data)
            engine = AnalyticsEngine(CONFIG.get("analytics", {}))
            test_size = (test_size_pct or 20) / 100.0
            children = []
            store = existing_results if isinstance(existing_results, dict) else {}

            # ---- Regression ----
            if task == "regression":
                if not target:
                    return html.Div("Please select a target column.", className="alert-danger"), no_update
                result = engine.regression(df, target, model_type=model_type,
                                           test_size=test_size)
                metrics_card = html.Div([
                    html.Div("Regression Metrics", className="card-header card-title"),
                    html.Div([
                        html.Div([
                            html.Div("R\u00b2 Score", className="result-metric-label"),
                            html.Div(f"{result['r2']:.4f}", className="result-metric-value"),
                        ], className="result-metric"),
                        html.Div([
                            html.Div("RMSE", className="result-metric-label"),
                            html.Div(f"{result['rmse']:.4f}", className="result-metric-value"),
                        ], className="result-metric"),
                        html.Div([
                            html.Div("MAE", className="result-metric-label"),
                            html.Div(f"{result['mae']:.4f}", className="result-metric-value"),
                        ], className="result-metric"),
                        html.Div([
                            html.Div("CV Mean R\u00b2", className="result-metric-label"),
                            html.Div(f"{result['cv_mean']:.4f}", className="result-metric-value"),
                        ], className="result-metric"),
                    ], style={"display": "flex", "gap": "24px", "padding": "16px",
                              "flexWrap": "wrap"}),
                ], className="card")
                children.append(metrics_card)

                if "feature_importances" in result:
                    children.append(_build_feature_importance_card(result["feature_importances"]))

                # Store regression results (exclude non-serializable model)
                store["regression"] = {k: v for k, v in result.items()
                                       if k != "model"}

            # ---- Classification ----
            elif task == "classification":
                if not target:
                    return html.Div("Please select a target column.", className="alert-danger"), no_update
                result = engine.classification(df, target, model_type=model_type,
                                               test_size=test_size)
                metrics_card = html.Div([
                    html.Div("Classification Metrics", className="card-header card-title"),
                    html.Div([
                        html.Div([
                            html.Div("Accuracy", className="result-metric-label"),
                            html.Div(f"{result['accuracy']:.4f}", className="result-metric-value"),
                        ], className="result-metric"),
                        html.Div([
                            html.Div("F1 Score", className="result-metric-label"),
                            html.Div(f"{result['f1']:.4f}", className="result-metric-value"),
                        ], className="result-metric"),
                        html.Div([
                            html.Div("Precision", className="result-metric-label"),
                            html.Div(f"{result['precision']:.4f}", className="result-metric-value"),
                        ], className="result-metric"),
                        html.Div([
                            html.Div("Recall", className="result-metric-label"),
                            html.Div(f"{result['recall']:.4f}", className="result-metric-value"),
                        ], className="result-metric"),
                    ], style={"display": "flex", "gap": "24px", "padding": "16px",
                              "flexWrap": "wrap"}),
                ], className="card")
                children.append(metrics_card)

                # Confusion matrix
                cm = np.array(result["confusion_matrix"])
                cm_fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    text=cm,
                    texttemplate="%{text}",
                    colorscale="Blues",
                    hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>",
                ))
                cm_fig.update_layout(
                    title="Confusion Matrix",
                    xaxis_title="Predicted",
                    yaxis_title="Actual",
                    height=400,
                    template="plotly_white",
                )
                children.append(html.Div([
                    html.Div("Confusion Matrix", className="card-header card-title"),
                    dcc.Graph(figure=cm_fig),
                ], className="card", style={"marginTop": "16px"}))

                if "feature_importances" in result:
                    children.append(_build_feature_importance_card(result["feature_importances"]))

                # Store classification results (exclude non-serializable model)
                store["classification"] = {k: v for k, v in result.items()
                                           if k != "model"}

            # ---- Clustering ----
            else:
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                if not numeric_cols:
                    return html.Div("No numeric columns available for clustering.",
                                    className="alert-danger"), no_update
                result = engine.clustering(df, features=numeric_cols,
                                           method=model_type,
                                           n_clusters=int(n_clusters or 4))
                metrics_card = html.Div([
                    html.Div("Clustering Metrics", className="card-header card-title"),
                    html.Div([
                        html.Div([
                            html.Div("Silhouette Score", className="result-metric-label"),
                            html.Div(f"{result['silhouette_score']:.4f}",
                                     className="result-metric-value"),
                        ], className="result-metric"),
                        html.Div([
                            html.Div("Number of Clusters", className="result-metric-label"),
                            html.Div(str(result["n_clusters"]),
                                     className="result-metric-value"),
                        ], className="result-metric"),
                    ], style={"display": "flex", "gap": "24px", "padding": "16px"}),
                    html.Div([
                        html.Div("Cluster Sizes", className="card-header card-title"),
                        html.Div([
                            html.Span(f"Cluster {k}: {v}  ")
                            for k, v in sorted(result["cluster_sizes"].items())
                        ], style={"padding": "8px 16px"}),
                    ]),
                ], className="card")
                children.append(metrics_card)

                # Store clustering results (exclude non-serializable items)
                store["clustering"] = {k: v for k, v in result.items()
                                       if k not in ("model",)}
                store["cluster_labels"] = result.get("labels", [])

            return html.Div(children), store

        except Exception as exc:
            return html.Div(f"ML Error: {exc}", className="alert-danger"), no_update


def _build_feature_importance_card(importances: dict):
    """Build a feature importance bar chart card."""
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig = go.Figure(data=go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker_color="#1f77b4",
    ))
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed"),
        height=max(300, len(features) * 28),
        template="plotly_white",
        margin=dict(l=150),
    )
    return html.Div([
        html.Div("Feature Importance", className="card-header card-title"),
        dcc.Graph(figure=fig),
    ], className="card", style={"marginTop": "16px"})
