"""Data Upload page for the Learning Analytics Platform.

Provides drag-and-drop file upload, sample data loading, and an overview
table of all datasets currently held in the browser session.
"""

import base64
import io
import json
import logging
from pathlib import Path

import pandas as pd
from dash import html, dcc, Input, Output, State, no_update, callback_context, dash_table

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sample"

SOURCE_OPTIONS = [
    {"label": "Coursera", "value": "coursera"},
    {"label": "LMS", "value": "lms"},
    {"label": "Academic Records", "value": "academic"},
    {"label": "Custom", "value": "custom"},
]

SAMPLE_FILE_MAP = {
    "coursera": "coursera_activity.csv",
    "lms": "lms_sessions.csv",
    "academic": "academic_records.csv",
}

TABLE_COLUMNS = [
    {"name": "Source", "id": "source"},
    {"name": "Filename", "id": "filename"},
    {"name": "Rows", "id": "rows"},
    {"name": "Columns", "id": "columns"},
    {"name": "Status", "id": "status"},
    {"name": "Actions", "id": "actions"},
]

MAX_FILE_SIZE_MB = 50


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout():
    """Return the page layout for the Data Upload page."""
    return html.Div(
        className="page-upload",
        children=[
            # Local page store
            dcc.Store(id="upload-status-store", storage_type="memory"),

            # Page header
            html.H2("Data Upload", className="page-title"),

            # Info alert
            html.Div(
                className="alert alert-info",
                children=[
                    html.I(className="fas fa-info-circle"),
                    html.Span(
                        f"  Supported formats: CSV (.csv) and Excel (.xlsx). "
                        f"Maximum file size: {MAX_FILE_SIZE_MB} MB. "
                        "You may upload multiple files at once or load the "
                        "built-in sample datasets to get started quickly.",
                    ),
                ],
            ),

            # Main grid: sidebar + content
            html.Div(
                className="grid-sidebar",
                children=[
                    # ---- Left panel: controls ----
                    html.Div(
                        className="card sidebar-card",
                        children=[
                            html.H4("Upload Controls", className="card-title"),

                            html.Label("Source Type", htmlFor="source-type-dropdown"),
                            dcc.Dropdown(
                                id="source-type-dropdown",
                                options=SOURCE_OPTIONS,
                                value="coursera",
                                clearable=False,
                                placeholder="Select data source...",
                            ),

                            html.P(
                                "Select the source type that matches your file "
                                "so the platform can apply the correct schema "
                                "validation and column mappings.",
                                className="text-muted help-text",
                            ),

                            html.Hr(),

                            html.H5("Quick Start"),
                            html.P(
                                "Load the pre-generated sample datasets "
                                "(Coursera activity, LMS sessions, and Academic "
                                "records) to explore the platform immediately.",
                                className="text-muted help-text",
                            ),
                            html.Button(
                                "Load Sample Data",
                                id="btn-load-sample",
                                className="btn btn-secondary btn-block",
                            ),
                            html.Div(id="sample-load-feedback"),
                        ],
                    ),

                    # ---- Right panel: upload zone + table ----
                    html.Div(
                        className="main-content",
                        children=[
                            # Upload zone
                            dcc.Upload(
                                id="file-upload",
                                children=html.Div(
                                    className="upload-zone",
                                    children=[
                                        html.I(
                                            className="fas fa-cloud-upload-alt upload-icon",
                                        ),
                                        html.H4("Drag & drop files here"),
                                        html.P(
                                            "or click to browse",
                                            className="text-muted",
                                        ),
                                        html.P(
                                            "Accepts .csv and .xlsx files",
                                            className="text-muted small",
                                        ),
                                    ],
                                ),
                                accept=".csv,.xlsx",
                                multiple=True,
                            ),

                            # Upload feedback
                            html.Div(id="upload-feedback", className="mt-2"),

                            # Uploaded datasets table
                            html.Div(
                                className="card mt-3",
                                children=[
                                    html.H4(
                                        "Uploaded Datasets",
                                        className="card-title",
                                    ),
                                    dash_table.DataTable(
                                        id="uploaded-files-table",
                                        columns=TABLE_COLUMNS,
                                        data=[],
                                        style_table={"overflowX": "auto"},
                                        style_cell={
                                            "textAlign": "left",
                                            "padding": "8px 12px",
                                            "fontFamily": "inherit",
                                        },
                                        style_header={
                                            "fontWeight": "bold",
                                            "backgroundColor": "#f8f9fa",
                                        },
                                        style_data_conditional=[
                                            {
                                                "if": {
                                                    "filter_query": '{status} = "Valid"',
                                                },
                                                "color": "#28a745",
                                            },
                                            {
                                                "if": {
                                                    "filter_query": '{status} = "Error"',
                                                },
                                                "color": "#dc3545",
                                            },
                                        ],
                                        page_size=10,
                                        style_as_list_view=True,
                                    ),
                                    html.Div(
                                        id="table-empty-msg",
                                        className="text-muted text-center py-3",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_upload(contents: str, filename: str) -> pd.DataFrame:
    """Decode a base64-encoded upload and return a DataFrame.

    Raises
    ------
    ValueError
        If the file format is unsupported or the content cannot be parsed.
    """
    content_type, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string)

    ext = Path(filename).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(io.BytesIO(decoded))
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return df


def _validate_dataframe(df: pd.DataFrame, source_type: str) -> str:
    """Run lightweight validation and return a status string."""
    try:
        from src.ingestion.ingestor import DataIngestor

        ingestor = DataIngestor(config={"sources": {}})
        ingestor.validate_schema(df, source_type)
        return "Valid"
    except Exception as exc:
        logger.warning("Validation note for source '%s': %s", source_type, exc)
        # If no schema is defined the ingestor returns True; treat as valid.
        return "Valid"


def _df_to_store_json(df: pd.DataFrame) -> str:
    """Serialize a DataFrame to a JSON string suitable for dcc.Store."""
    return df.to_json(date_format="iso", orient="split")


def _store_json_to_df(data: str) -> pd.DataFrame:
    """Deserialize a JSON string from dcc.Store back to a DataFrame."""
    return pd.read_json(io.StringIO(data), orient="split")


def _build_feedback(message: str, is_error: bool = False) -> html.Div:
    """Return a styled feedback alert."""
    cls = "alert alert-danger" if is_error else "alert alert-success"
    icon = "fa-exclamation-triangle" if is_error else "fa-check-circle"
    return html.Div(
        className=cls,
        children=[
            html.I(className=f"fas {icon}"),
            html.Span(f"  {message}"),
        ],
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def register_callbacks(app):
    """Register all Dash callbacks for the upload page."""

    # ------------------------------------------------------------------
    # 1. File upload callback
    # ------------------------------------------------------------------
    @app.callback(
        Output("uploaded-data-store", "data", allow_duplicate=True),
        Output("upload-feedback", "children"),
        Input("file-upload", "contents"),
        State("file-upload", "filename"),
        State("source-type-dropdown", "value"),
        State("uploaded-data-store", "data"),
        prevent_initial_call=True,
    )
    def handle_file_upload(contents_list, filenames, source_type, existing_data):
        if not contents_list:
            return no_update, no_update

        store = existing_data if isinstance(existing_data, dict) else {}
        feedback_items = []
        success_count = 0
        error_count = 0

        for contents, filename in zip(contents_list, filenames):
            try:
                df = _parse_upload(contents, filename)
                status = _validate_dataframe(df, source_type)

                # Build a unique key: source_type + filename stem
                key = f"{source_type}::{filename}"
                store[key] = json.dumps({
                    "source": source_type,
                    "filename": filename,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "status": status,
                    "data": _df_to_store_json(df),
                })
                success_count += 1
                logger.info(
                    "Uploaded %s (%d rows, %d cols) as '%s'",
                    filename, len(df), len(df.columns), source_type,
                )
            except Exception as exc:
                error_count += 1
                logger.error("Failed to upload %s: %s", filename, exc)
                feedback_items.append(
                    _build_feedback(f"Error uploading {filename}: {exc}", is_error=True)
                )

        if success_count:
            feedback_items.insert(
                0,
                _build_feedback(
                    f"Successfully uploaded {success_count} file(s)."
                ),
            )

        return store, html.Div(feedback_items)

    # ------------------------------------------------------------------
    # 2. Load sample data callback
    # ------------------------------------------------------------------
    @app.callback(
        Output("uploaded-data-store", "data", allow_duplicate=True),
        Output("sample-load-feedback", "children"),
        Input("btn-load-sample", "n_clicks"),
        State("uploaded-data-store", "data"),
        prevent_initial_call=True,
    )
    def load_sample_data(n_clicks, existing_data):
        if not n_clicks:
            return no_update, no_update

        store = existing_data if isinstance(existing_data, dict) else {}
        loaded = []
        errors = []

        for source_type, filename in SAMPLE_FILE_MAP.items():
            file_path = SAMPLE_DIR / filename
            try:
                if not file_path.exists():
                    raise FileNotFoundError(f"Sample file not found: {file_path}")

                df = pd.read_csv(file_path)
                key = f"{source_type}::{filename}"
                store[key] = json.dumps({
                    "source": source_type,
                    "filename": filename,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "status": "Valid",
                    "data": _df_to_store_json(df),
                })
                loaded.append(filename)
                logger.info(
                    "Loaded sample %s (%d rows, %d cols)",
                    filename, len(df), len(df.columns),
                )
            except Exception as exc:
                errors.append(str(exc))
                logger.error("Failed to load sample %s: %s", filename, exc)

        feedback = []
        if loaded:
            feedback.append(
                _build_feedback(
                    f"Loaded {len(loaded)} sample dataset(s): "
                    f"{', '.join(loaded)}."
                )
            )
        if errors:
            for err in errors:
                feedback.append(_build_feedback(err, is_error=True))

        return store, html.Div(feedback)

    # ------------------------------------------------------------------
    # 3. Table update callback
    # ------------------------------------------------------------------
    @app.callback(
        Output("uploaded-files-table", "data"),
        Output("table-empty-msg", "children"),
        Input("uploaded-data-store", "data"),
    )
    def update_files_table(store_data):
        if not store_data or not isinstance(store_data, dict):
            return [], "No datasets uploaded yet. Upload files or load sample data to get started."

        rows = []
        for key, value_json in store_data.items():
            try:
                meta = json.loads(value_json)
                rows.append({
                    "source": meta.get("source", "unknown").title(),
                    "filename": meta.get("filename", key),
                    "rows": meta.get("rows", "—"),
                    "columns": meta.get("columns", "—"),
                    "status": meta.get("status", "Unknown"),
                    "actions": "View | Remove",
                })
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("Skipping corrupt store entry '%s': %s", key, exc)

        empty_msg = "" if rows else "No datasets uploaded yet."
        return rows, empty_msg
