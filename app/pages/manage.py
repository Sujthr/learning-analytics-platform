"""Data Management page for the Learning Analytics Platform.

Provides data cleaning, integration, and feature engineering controls
with a live data preview table and quality KPIs.
"""

import io
import json
import logging
from pathlib import Path

import pandas as pd
import yaml
from dash import html, dcc, Input, Output, State, no_update, callback_context
from dash import dash_table

from src.cleaning.cleaner import DataCleaner
from src.integration.integrator import DataIntegrator
from src.features.engineer import FeatureEngineer

logger = logging.getLogger(__name__)

_config_path = Path(__file__).parent.parent.parent / "config" / "pipeline_config.yaml"
with open(_config_path) as _f:
    CONFIG = yaml.safe_load(_f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_to_store(df: pd.DataFrame) -> str:
    """Serialize a DataFrame for dcc.Store."""
    return df.to_json(date_format="iso", orient="split")


def _store_to_df(data) -> pd.DataFrame:
    """Deserialize a DataFrame from dcc.Store (handles str or dict)."""
    if isinstance(data, str):
        return pd.read_json(io.StringIO(data), orient="split")
    if isinstance(data, dict):
        return pd.read_json(io.StringIO(json.dumps(data)), orient="split")
    raise TypeError(f"Unexpected store data type: {type(data)}")


def _parse_uploaded_store(store_data: dict) -> dict[str, pd.DataFrame]:
    """Parse the uploaded-data-store dict into {source_name: DataFrame}.

    The upload page stores data as:
        {"source::filename": '{"source":..., "data": "<df json>"}', ...}
    """
    datasets = {}
    if not isinstance(store_data, dict):
        return datasets
    for key, value in store_data.items():
        try:
            if isinstance(value, str):
                meta = json.loads(value)
            else:
                meta = value
            source = meta.get("source", key.split("::")[0] if "::" in key else key)
            df_json = meta.get("data", meta)
            if isinstance(df_json, str):
                df = pd.read_json(io.StringIO(df_json), orient="split")
            else:
                df = pd.DataFrame(df_json)
            datasets[source] = df
        except Exception as e:
            logger.warning("Failed to parse uploaded store key '%s': %s", key, e)
    return datasets


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout():
    """Return the Data Management page layout."""
    return html.Div(className="grid-sidebar", children=[

        # ---- Left panel: Control Panel ----
        html.Div(className="card", children=[
            html.Div(className="card-header", children=[
                html.H3("Control Panel", className="card-title"),
            ]),

            # -- Data Cleaning --
            html.Div(style={"padding": "1rem"}, children=[
                html.H4("Data Cleaning"),

                html.Div(className="form-group", children=[
                    html.Label("Missing Values Strategy", className="form-label"),
                    dcc.Dropdown(
                        id="clean-missing-strategy",
                        options=[
                            {"label": "Mean", "value": "mean"},
                            {"label": "Median", "value": "median"},
                            {"label": "Drop", "value": "drop"},
                            {"label": "Zero", "value": "zero"},
                        ],
                        value="median",
                        clearable=False,
                    ),
                ]),

                html.Div(className="form-group", children=[
                    html.Label("Missing Threshold (%)", className="form-label"),
                    dcc.Slider(
                        id="clean-threshold",
                        min=0,
                        max=100,
                        step=5,
                        value=50,
                        marks={i: f"{i}%" for i in range(0, 101, 25)},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ]),

                html.Div(className="form-group", children=[
                    html.Label("Duplicate Handling", className="form-label"),
                    dcc.Dropdown(
                        id="clean-duplicates",
                        options=[
                            {"label": "Keep First", "value": "first"},
                            {"label": "Keep Last", "value": "last"},
                            {"label": "Remove All", "value": "false"},
                        ],
                        value="first",
                        clearable=False,
                    ),
                ]),

                html.Button(
                    "Run Cleaning",
                    id="btn-clean",
                    className="btn btn-primary",
                    title="Remove duplicates, fill missing values & normalize timestamps",
                ),

                html.Hr(),

                # -- Data Integration --
                html.H4("Data Integration"),

                html.Div(className="form-group", children=[
                    html.Label("Join Key", className="form-label"),
                    dcc.Dropdown(
                        id="integration-join-key",
                        options=[
                            {"label": "student_id", "value": "student_id"},
                            {"label": "course_id", "value": "course_id"},
                        ],
                        value="student_id",
                        clearable=False,
                    ),
                ]),

                html.Div(className="form-group", children=[
                    html.Label("Conflict Resolution", className="form-label"),
                    dcc.Dropdown(
                        id="integration-conflict",
                        options=[
                            {"label": "Latest", "value": "latest"},
                            {"label": "Source Priority", "value": "source_priority"},
                        ],
                        value="latest",
                        clearable=False,
                    ),
                ]),

                html.Button(
                    "Run Integration",
                    id="btn-integrate",
                    className="btn btn-success",
                    title="Merge all sources using student ID mapping table",
                ),

                html.Hr(),

                # -- Feature Engineering --
                html.H4("Feature Engineering"),

                html.Div(className="form-group", children=[
                    html.Label("Features to Compute", className="form-label"),
                    dcc.Checklist(
                        id="feature-checklist",
                        options=[
                            {"label": "Engagement Score", "value": "engagement_score"},
                            {"label": "Session Frequency", "value": "session_frequency"},
                            {"label": "Video Completion", "value": "video_completion"},
                            {"label": "Assessment Improvement", "value": "assessment_improvement"},
                        ],
                        value=[
                            "engagement_score",
                            "session_frequency",
                            "video_completion",
                            "assessment_improvement",
                        ],
                    ),
                ]),

                html.Button(
                    "Generate Features",
                    id="btn-features",
                    className="btn btn-primary",
                    title="Compute engagement score, session frequency & more",
                ),
            ]),
        ]),

        # ---- Right panel: Data Preview ----
        html.Div(children=[

            # Tab navigation
            html.Div(className="tab-nav", children=[
                html.Div(
                    "Raw Data",
                    id="tab-raw",
                    className="tab-item",
                    n_clicks=0,
                ),
                html.Div(
                    "Cleaned Data",
                    id="tab-cleaned",
                    className="tab-item",
                    n_clicks=0,
                ),
                html.Div(
                    "Integrated Data",
                    id="tab-integrated",
                    className="tab-item",
                    n_clicks=0,
                ),
            ]),

            # Hidden store to track active tab
            dcc.Store(id="active-tab-store", data="raw"),

            # Status alert area
            html.Div(id="manage-alert", style={"marginTop": "0.5rem"}),

            # Quality KPIs
            html.Div(className="card", style={"marginTop": "1rem"}, children=[
                html.Div(className="card-header", children=[
                    html.H3("Data Quality Summary", className="card-title"),
                ]),
                html.Div(className="kpi-grid", children=[
                    html.Div(className="kpi-card", children=[
                        html.Span("Total Rows"),
                        html.H3(id="kpi-total-rows", children="--"),
                    ]),
                    html.Div(className="kpi-card", children=[
                        html.Span("Missing Values"),
                        html.H3(id="kpi-missing-values", children="--"),
                    ]),
                    html.Div(className="kpi-card", children=[
                        html.Span("Duplicates"),
                        html.H3(id="kpi-duplicates", children="--"),
                    ]),
                    html.Div(className="kpi-card", children=[
                        html.Span("Columns"),
                        html.H3(id="kpi-columns", children="--"),
                    ]),
                ]),
            ]),

            # Data table
            html.Div(className="card", style={"marginTop": "1rem"}, children=[
                dash_table.DataTable(
                    id="data-preview-table",
                    page_size=15,
                    sort_action="native",
                    filter_action="native",
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "textAlign": "left",
                        "padding": "8px",
                        "minWidth": "100px",
                    },
                    style_header={
                        "fontWeight": "bold",
                    },
                ),
            ]),
        ]),

    ])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def register_callbacks(app):
    """Register all Dash callbacks for the Data Management page."""

    # ---- Tab switching ----
    @app.callback(
        Output("active-tab-store", "data"),
        Input("tab-raw", "n_clicks"),
        Input("tab-cleaned", "n_clicks"),
        Input("tab-integrated", "n_clicks"),
        prevent_initial_call=True,
    )
    def switch_tab(raw_clicks, cleaned_clicks, integrated_clicks):
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        tab_map = {
            "tab-raw": "raw",
            "tab-cleaned": "cleaned",
            "tab-integrated": "integrated",
        }
        return tab_map.get(triggered_id, "raw")

    # ---- Data Cleaning ----
    @app.callback(
        Output("cleaned-data-store", "data"),
        Output("manage-alert", "children", allow_duplicate=True),
        Input("btn-clean", "n_clicks"),
        State("uploaded-data-store", "data"),
        State("clean-missing-strategy", "value"),
        State("clean-threshold", "value"),
        State("clean-duplicates", "value"),
        prevent_initial_call=True,
    )
    def run_cleaning(n_clicks, raw_data, strategy, threshold, dup_handling):
        if not n_clicks or not raw_data:
            return no_update, html.Div(
                "No data available. Please upload data first.",
                className="alert alert-danger",
            )

        try:
            datasets = _parse_uploaded_store(raw_data)
            if not datasets:
                return no_update, html.Div(
                    "Could not parse uploaded data.", className="alert alert-danger",
                )

            cleaning_config = {
                "missing_values": {
                    "numeric_strategy": strategy,
                    "categorical_strategy": "mode",
                    "threshold": threshold / 100.0,
                },
                "duplicates": {
                    "subset": None,
                    "keep": dup_handling,
                },
                "timestamp": CONFIG.get("cleaning", {}).get("timestamp", {}),
            }

            cleaner = DataCleaner(cleaning_config)

            cleaned_store = {}
            total_removed = 0
            total_filled = 0
            for source, df_raw in datasets.items():
                df_cleaned = cleaner.clean(df_raw)
                report = cleaner.get_cleaning_report(df_raw, df_cleaned)
                total_removed += report["rows_removed"]
                total_filled += report["missing_filled"]
                cleaned_store[source] = _df_to_store(df_cleaned)

            alert = html.Div(
                f"Cleaning complete across {len(datasets)} source(s): "
                f"{total_removed} rows removed, "
                f"{total_filled} missing values handled.",
                className="alert alert-success",
            )

            return cleaned_store, alert

        except Exception as exc:
            logger.exception("Cleaning failed")
            return no_update, html.Div(
                f"Cleaning failed: {exc}",
                className="alert alert-danger",
            )

    # ---- Data Integration ----
    @app.callback(
        Output("integrated-data-store", "data"),
        Output("manage-alert", "children", allow_duplicate=True),
        Input("btn-integrate", "n_clicks"),
        State("cleaned-data-store", "data"),
        State("integration-join-key", "value"),
        State("integration-conflict", "value"),
        prevent_initial_call=True,
    )
    def run_integration(n_clicks, cleaned_data, join_key, conflict_res):
        if not n_clicks or not cleaned_data:
            return no_update, html.Div(
                "No cleaned data available. Please run cleaning first.",
                className="alert alert-danger",
            )

        try:
            # cleaned_data is a dict {source: json_str}
            datasets = {}
            if isinstance(cleaned_data, dict):
                for source, json_str in cleaned_data.items():
                    datasets[source] = _store_to_df(json_str)
            else:
                datasets["primary"] = _store_to_df(cleaned_data)

            integration_config = {
                "join_key": join_key,
                "conflict_resolution": conflict_res,
                "source_priority": CONFIG.get("integration", {}).get(
                    "source_priority", []
                ),
            }

            integrator = DataIntegrator(integration_config)

            # Load mapping table if available
            mapping_path = Path(__file__).parent.parent.parent / "data" / "sample" / "student_mapping.csv"
            if mapping_path.exists():
                integrator.load_mapping_table(str(mapping_path))

            df_integrated = integrator.integrate(datasets)
            report = integrator.get_integration_report()

            alert = html.Div(
                f"Integration complete: {report['total_sources']} source(s), "
                f"{report['matched_records']} matched records, "
                f"{report['conflicts_resolved']} conflicts resolved.",
                className="alert alert-success",
            )

            return _df_to_store(df_integrated), alert

        except Exception as exc:
            logger.exception("Integration failed")
            return no_update, html.Div(
                f"Integration failed: {exc}",
                className="alert alert-danger",
            )

    # ---- Feature Engineering ----
    @app.callback(
        Output("featured-data-store", "data"),
        Output("manage-alert", "children", allow_duplicate=True),
        Input("btn-features", "n_clicks"),
        State("integrated-data-store", "data"),
        State("feature-checklist", "value"),
        prevent_initial_call=True,
    )
    def run_feature_engineering(n_clicks, integrated_data, selected_features):
        if not n_clicks or not integrated_data:
            return no_update, html.Div(
                "No integrated data available. Please run integration first.",
                className="alert-danger",
            )

        try:
            df_integrated = _store_to_df(integrated_data)

            feature_config = CONFIG.get("features", {})
            engineer = FeatureEngineer(feature_config)

            df_featured = df_integrated.copy()

            if "session_frequency" in selected_features:
                df_featured = engineer.compute_session_frequency(
                    df_featured,
                    feature_config.get("session_frequency", {}).get("window_days", 30),
                )
            if "video_completion" in selected_features:
                df_featured = engineer.compute_video_completion(
                    df_featured,
                    feature_config.get("video_completion", {}).get("threshold", 0.8),
                )
            if "assessment_improvement" in selected_features:
                df_featured = engineer.compute_assessment_improvement(
                    df_featured,
                    feature_config.get("assessment_improvement", {}).get("method", "slope"),
                )
            if "engagement_score" in selected_features:
                df_featured = engineer.compute_engagement_score(df_featured)

            new_cols = set(df_featured.columns) - set(df_integrated.columns)
            alert = html.Div(
                f"Feature engineering complete: {len(new_cols)} new feature(s) added "
                f"({', '.join(sorted(new_cols)) if new_cols else 'none'}).",
                className="alert-success",
            )

            return _df_to_store(df_featured), alert

        except Exception as exc:
            logger.exception("Feature engineering failed")
            return no_update, html.Div(
                f"Feature engineering failed: {exc}",
                className="alert-danger",
            )

    # ---- Preview & KPIs ----
    @app.callback(
        Output("data-preview-table", "data"),
        Output("data-preview-table", "columns"),
        Output("kpi-total-rows", "children"),
        Output("kpi-missing-values", "children"),
        Output("kpi-duplicates", "children"),
        Output("kpi-columns", "children"),
        Input("active-tab-store", "data"),
        Input("uploaded-data-store", "data"),
        Input("cleaned-data-store", "data"),
        Input("integrated-data-store", "data"),
        Input("featured-data-store", "data"),
    )
    def update_preview(active_tab, raw_data, cleaned_data, integrated_data, featured_data):
        # Fall back: if integrated tab selected but featured data exists, show featured
        if active_tab == "integrated" and featured_data:
            data = featured_data
        elif active_tab == "integrated" and integrated_data:
            data = integrated_data
        elif active_tab == "cleaned" and cleaned_data:
            data = cleaned_data
        elif active_tab == "raw" and raw_data:
            data = raw_data
        else:
            data = None

        if not data:
            return [], [], "--", "--", "--", "--"

        try:
            # Handle dict-of-sources (uploaded or cleaned stores)
            if isinstance(data, dict):
                frames = []
                for key, val in data.items():
                    try:
                        if isinstance(val, str):
                            # Could be a json-encoded meta object from upload
                            try:
                                meta = json.loads(val)
                                if "data" in meta:
                                    sub_df = pd.read_json(io.StringIO(meta["data"]), orient="split")
                                else:
                                    sub_df = pd.read_json(io.StringIO(val), orient="split")
                            except (json.JSONDecodeError, ValueError):
                                sub_df = pd.read_json(io.StringIO(val), orient="split")
                        else:
                            sub_df = pd.DataFrame(val)
                        sub_df["_source"] = key.split("::")[0] if "::" in key else key
                        frames.append(sub_df)
                    except Exception:
                        continue
                if not frames:
                    return [], [], "--", "--", "--", "--"
                df = pd.concat(frames, ignore_index=True)
            else:
                df = _store_to_df(data)
        except Exception:
            return [], [], "--", "--", "--", "--"

        # Limit preview to first 100 rows
        preview = df.head(100)

        columns = [{"name": col, "id": col} for col in preview.columns]
        records = preview.to_dict("records")

        # KPIs
        total_rows = f"{len(df):,}"
        missing_vals = f"{int(df.isnull().sum().sum()):,}"
        duplicates = f"{int(df.duplicated().sum()):,}"
        num_columns = str(len(df.columns))

        return records, columns, total_rows, missing_vals, duplicates, num_columns
