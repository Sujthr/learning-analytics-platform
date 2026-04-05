"""Home / Dashboard page for the Learning Analytics Platform."""

from dash import html, dcc, callback_context, Input, Output, State
import dash


# ---------------------------------------------------------------------------
# Pipeline step definitions
# ---------------------------------------------------------------------------
PIPELINE_STEPS = [
    {"key": "upload", "label": "Upload"},
    {"key": "clean", "label": "Clean"},
    {"key": "integrate", "label": "Integrate"},
    {"key": "features", "label": "Features"},
    {"key": "analytics", "label": "Analytics"},
]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout():
    """Return the home/dashboard page layout."""

    return html.Div(
        [
            # Interval timer for polling
            dcc.Interval(
                id="pipeline-poll-interval",
                interval=5000,  # 5 seconds
                n_intervals=0,
            ),

            # ---- KPI Grid ----
            html.Div(
                [
                    _kpi_card(
                        icon_class="kpi-icon-blue",
                        label="Total Students",
                        value_id="kpi-total-students",
                        default_value="--",
                    ),
                    _kpi_card(
                        icon_class="kpi-icon-green",
                        label="Datasets Loaded",
                        value_id="kpi-datasets-loaded",
                        default_value="0",
                    ),
                    _kpi_card(
                        icon_class="kpi-icon-amber",
                        label="Avg Engagement Score",
                        value_id="kpi-avg-engagement",
                        default_value="--",
                    ),
                    _kpi_card(
                        icon_class="kpi-icon-red",
                        label="Dropout Rate",
                        value_id="kpi-dropout-rate",
                        default_value="--",
                    ),
                ],
                className="kpi-grid",
            ),

            # ---- Middle Row: Pipeline Status + Quick Actions ----
            html.Div(
                [
                    # Left card - Pipeline Status
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Pipeline Status", className="card-title"),
                                ],
                                className="card-header",
                            ),
                            html.Div(
                                id="pipeline-steps-list",
                                children=_build_pipeline_steps({}),
                            ),
                        ],
                        className="card",
                    ),

                    # Right card - Quick Actions
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Quick Actions", className="card-title"),
                                ],
                                className="card-header",
                            ),
                            html.Div(
                                [
                                    dcc.Link(
                                        html.Button(
                                            "Upload Data",
                                            className="btn btn-primary",
                                        ),
                                        href="/upload",
                                        style={"textDecoration": "none"},
                                    ),
                                    dcc.Link(
                                        html.Button(
                                            "Run Analysis",
                                            className="btn btn-success",
                                        ),
                                        href="/analyze",
                                        style={"textDecoration": "none"},
                                    ),
                                    dcc.Link(
                                        html.Button(
                                            "Generate Report",
                                            className="btn btn-secondary",
                                        ),
                                        href="/reports",
                                        style={"textDecoration": "none"},
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "flexDirection": "column",
                                    "gap": "12px",
                                },
                            ),
                        ],
                        className="card",
                    ),
                ],
                className="grid-2",
            ),

            # ---- Recent Activity ----
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Recent Activity", className="card-title"),
                        ],
                        className="card-header",
                    ),
                    html.Div(
                        id="activity-log-container",
                        children=[
                            html.Div(
                                "No activity yet. Upload a dataset to get started.",
                                style={"color": "#64748b", "fontStyle": "italic"},
                            ),
                        ],
                        className="log-container",
                    ),
                ],
                className="card",
            ),
        ],
        className="page-content",
    )


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _kpi_card(icon_class: str, label: str, value_id: str, default_value: str):
    """Build a single KPI card element."""
    return html.Div(
        [
            html.Div(className=f"kpi-icon {icon_class}"),
            html.Div(label, className="kpi-label"),
            html.Div(default_value, id=value_id, className="kpi-value"),
        ],
        className="kpi-card",
    )


def _build_pipeline_steps(status_map: dict):
    """Build the list of pipeline steps with status dots.

    Parameters
    ----------
    status_map : dict
        Mapping of step key -> bool (True = done, False/missing = pending).
    """
    items = []
    for step in PIPELINE_STEPS:
        done = status_map.get(step["key"], False)
        dot_class = "status-dot status-dot-green" if done else "status-dot status-dot-gray"
        items.append(
            html.Div(
                [
                    html.Span(className=dot_class),
                    html.Span(step["label"]),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "padding": "8px 0",
                    "fontSize": "14px",
                },
            )
        )
    return items


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def register_callbacks(app):
    """Register Dash callbacks for the home/dashboard page."""

    @app.callback(
        [
            Output("kpi-total-students", "children"),
            Output("kpi-datasets-loaded", "children"),
            Output("kpi-avg-engagement", "children"),
            Output("kpi-dropout-rate", "children"),
            Output("pipeline-steps-list", "children"),
            Output("activity-log-container", "children"),
        ],
        [
            Input("session-store", "data"),
            Input("pipeline-status-store", "data"),
        ],
    )
    def update_dashboard(session_data, pipeline_data):
        """Read session and pipeline stores, then update KPIs and status."""

        session_data = session_data or {}
        pipeline_data = pipeline_data or {}

        # KPI values
        total_students = session_data.get("total_students", "--")
        datasets_loaded = session_data.get("datasets_loaded", "0")
        avg_engagement = session_data.get("avg_engagement", "--")
        dropout_rate = session_data.get("dropout_rate", "--")

        # Format percentage values when available
        if isinstance(avg_engagement, (int, float)):
            avg_engagement = f"{avg_engagement:.1f}%"
        if isinstance(dropout_rate, (int, float)):
            dropout_rate = f"{dropout_rate:.1f}%"

        # Pipeline status
        pipeline_steps = _build_pipeline_steps(pipeline_data)

        # Activity log
        log_messages = session_data.get("pipeline_log", [])
        if log_messages:
            log_children = [
                html.Div(msg) for msg in log_messages
            ]
        else:
            log_children = [
                html.Div(
                    "No activity yet. Upload a dataset to get started.",
                    style={"color": "#64748b", "fontStyle": "italic"},
                ),
            ]

        return (
            str(total_students),
            str(datasets_loaded),
            str(avg_engagement),
            str(dropout_rate),
            pipeline_steps,
            log_children,
        )
