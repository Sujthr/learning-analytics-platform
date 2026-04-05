"""Reports page for the Learning Analytics Platform."""

from dash import html, dcc, Input, Output, State, no_update, callback_context
from dash import dash_table
import pandas as pd
import json
import io
import yaml
import tempfile
from pathlib import Path

from src.reporting.reporter import ReportGenerator


def _store_to_df(data) -> pd.DataFrame:
    """Deserialize a DataFrame from dcc.Store."""
    if data is None:
        return None
    if isinstance(data, str):
        return pd.read_json(io.StringIO(data), orient="split")
    if isinstance(data, dict):
        # Dash auto-parses JSON strings in stores into dicts
        return pd.read_json(io.StringIO(json.dumps(data)), orient="split")
    return None


# Load pipeline config
_config_path = Path(__file__).parent.parent.parent / "config" / "pipeline_config.yaml"
with open(_config_path) as _f:
    _config = yaml.safe_load(_f)

REPORT_SECTIONS = [
    {"label": "Summary Statistics", "value": "summary_statistics"},
    {"label": "Distribution Analysis", "value": "distribution_analysis"},
    {"label": "Correlation Analysis", "value": "correlation_analysis"},
    {"label": "Hypothesis Test Results", "value": "hypothesis_test_results"},
    {"label": "ML Model Results", "value": "ml_model_results"},
]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout():
    """Return the reports page layout."""

    return html.Div(
        [
            # Info alert
            html.Div(
                "Generate comprehensive reports from your analytics results. "
                "Select the sections to include, choose an export format, and "
                "download publication-ready reports in PDF or CSV format.",
                className="alert-info",
            ),

            # Top grid: Generate Reports + Report Preview
            html.Div(
                [
                    # Left card - Generate Reports
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Generate Reports", className="card-title"),
                                ],
                                className="card-header",
                            ),
                            html.Div(
                                [
                                    # Report type checklist
                                    html.Div(
                                        [
                                            html.Label(
                                                "Report Sections",
                                                className="form-label",
                                            ),
                                            dcc.Checklist(
                                                id="report-types",
                                                options=REPORT_SECTIONS,
                                                value=[],
                                                labelStyle={
                                                    "display": "block",
                                                    "marginBottom": "6px",
                                                },
                                            ),
                                        ],
                                        className="form-group",
                                    ),

                                    # Export format
                                    html.Div(
                                        [
                                            html.Label(
                                                "Export Format",
                                                className="form-label",
                                            ),
                                            dcc.RadioItems(
                                                id="export-format",
                                                options=[
                                                    {"label": "PDF", "value": "pdf"},
                                                    {"label": "CSV", "value": "csv"},
                                                ],
                                                value="pdf",
                                                labelStyle={
                                                    "display": "inline-block",
                                                    "marginRight": "16px",
                                                },
                                            ),
                                        ],
                                        className="form-group",
                                    ),

                                    # Decimal places
                                    html.Div(
                                        [
                                            html.Label(
                                                "Decimal Places",
                                                className="form-label",
                                            ),
                                            dcc.Input(
                                                id="decimal-places",
                                                type="number",
                                                value=3,
                                                min=1,
                                                max=6,
                                                style={"width": "80px"},
                                            ),
                                        ],
                                        className="form-group",
                                    ),

                                    # Generate button
                                    html.Button(
                                        "Generate Report",
                                        id="btn-generate-report",
                                        className="btn btn-primary btn-lg",
                                    ),

                                    # Hidden download component
                                    dcc.Download(id="download-report"),
                                ],
                            ),
                        ],
                        className="card",
                    ),

                    # Right card - Report Preview
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Report Preview", className="card-title"),
                                ],
                                className="card-header",
                            ),
                            html.Div(
                                id="report-preview",
                                children=[
                                    html.Div(
                                        "Select report sections and click "
                                        "'Generate Report' to preview content.",
                                        style={
                                            "color": "#64748b",
                                            "fontStyle": "italic",
                                        },
                                    ),
                                ],
                            ),
                        ],
                        className="card",
                    ),
                ],
                className="grid-2",
            ),

            # Export Data card
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Export Data", className="card-title"),
                        ],
                        className="card-header",
                    ),
                    html.Div(
                        [
                            html.Button(
                                "Export Integrated Dataset (CSV)",
                                id="btn-export-integrated",
                                className="btn btn-secondary",
                            ),
                            html.Button(
                                "Export Feature Dataset (CSV)",
                                id="btn-export-features",
                                className="btn btn-secondary",
                            ),
                            html.Button(
                                "Export Summary Stats (CSV)",
                                id="btn-export-summary",
                                className="btn btn-secondary",
                            ),
                        ],
                        style={
                            "display": "flex",
                            "gap": "12px",
                            "flexWrap": "wrap",
                        },
                    ),
                    dcc.Download(id="download-integrated"),
                    dcc.Download(id="download-features"),
                    dcc.Download(id="download-summary"),
                ],
                className="card",
            ),
        ],
        className="page-content",
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def register_callbacks(app):
    """Register Dash callbacks for the reports page."""

    # ------------------------------------------------------------------
    # 1. Generate report and trigger download
    # ------------------------------------------------------------------
    @app.callback(
        Output("download-report", "data"),
        Input("btn-generate-report", "n_clicks"),
        [
            State("report-types", "value"),
            State("export-format", "value"),
            State("decimal-places", "value"),
            State("featured-data-store", "data"),
            State("analytics-results-store", "data"),
        ],
        prevent_initial_call=True,
    )
    def generate_report(n_clicks, report_types, export_format, decimal_places,
                        featured_data, analytics_results):
        if not n_clicks or not report_types:
            return no_update

        decimal_places = int(decimal_places or 3)
        analytics_results = analytics_results or {}

        # Build results dict filtered to selected sections
        results = _filter_results(report_types, analytics_results, featured_data)

        report_config = {
            **_config.get("reporting", {}),
            "decimal_places": decimal_places,
        }
        generator = ReportGenerator(report_config)

        if export_format == "pdf":
            # Generate PDF to temp file, then send
            with tempfile.TemporaryDirectory() as tmp_dir:
                generator.export_pdf(results, "analytics_report.pdf", tmp_dir)
                pdf_path = Path(tmp_dir) / "analytics_report.pdf"
                return dcc.send_file(str(pdf_path), filename="analytics_report.pdf")

        else:
            # CSV export - build a combined summary DataFrame
            df = _build_summary_dataframe(results, featured_data, generator)
            return dcc.send_data_frame(df.to_csv, "analytics_report.csv")

    # ------------------------------------------------------------------
    # 2. Report preview when sections are selected
    # ------------------------------------------------------------------
    @app.callback(
        Output("report-preview", "children"),
        [
            Input("report-types", "value"),
            Input("featured-data-store", "data"),
            Input("analytics-results-store", "data"),
        ],
    )
    def update_preview(report_types, featured_data, analytics_results):
        if not report_types:
            return html.Div(
                "Select report sections to preview content.",
                style={"color": "#64748b", "fontStyle": "italic"},
            )

        analytics_results = analytics_results or {}
        preview_children = []

        # Summary Statistics
        if "summary_statistics" in report_types:
            preview_children.append(
                html.H4("Summary Statistics", style={"marginTop": "12px"})
            )
            if featured_data:
                try:
                    df = _store_to_df(featured_data)
                    report_config = _config.get("reporting", {})
                    generator = ReportGenerator(report_config)
                    stats_df = generator.generate_summary_stats(df)
                    stats_df = stats_df.reset_index().rename(
                        columns={"index": "Feature"}
                    )
                    preview_children.append(
                        dash_table.DataTable(
                            data=stats_df.to_dict("records"),
                            columns=[
                                {"name": c, "id": c} for c in stats_df.columns
                            ],
                            style_table={"overflowX": "auto"},
                            style_cell={"textAlign": "left", "padding": "6px"},
                            style_header={
                                "fontWeight": "bold",
                                "backgroundColor": "#f0f4f8",
                            },
                            page_size=10,
                        )
                    )
                except Exception:
                    preview_children.append(
                        html.Div("Could not generate summary statistics.")
                    )
            else:
                preview_children.append(
                    html.Div("No feature data available.", style={"color": "#64748b"})
                )

        # Distribution Analysis
        if "distribution_analysis" in report_types:
            preview_children.append(
                html.H4("Distribution Analysis", style={"marginTop": "12px"})
            )
            if "distributions" in analytics_results:
                dist = analytics_results["distributions"]
                dist_df = pd.DataFrame(dist).T.reset_index().rename(
                    columns={"index": "Feature"}
                )
                preview_children.append(
                    dash_table.DataTable(
                        data=dist_df.to_dict("records"),
                        columns=[{"name": c, "id": c} for c in dist_df.columns],
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "padding": "6px"},
                        style_header={
                            "fontWeight": "bold",
                            "backgroundColor": "#f0f4f8",
                        },
                        page_size=10,
                    )
                )
            else:
                preview_children.append(
                    html.Div(
                        "No distribution data available.",
                        style={"color": "#64748b"},
                    )
                )

        # Correlation Analysis
        if "correlation_analysis" in report_types:
            preview_children.append(
                html.H4("Correlation Analysis", style={"marginTop": "12px"})
            )
            if "correlations" in analytics_results:
                corr = analytics_results["correlations"]
                corr_df = pd.DataFrame(corr)
                corr_df = corr_df.reset_index().rename(columns={"index": "Feature"})
                preview_children.append(
                    dash_table.DataTable(
                        data=corr_df.to_dict("records"),
                        columns=[{"name": c, "id": c} for c in corr_df.columns],
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "padding": "6px"},
                        style_header={
                            "fontWeight": "bold",
                            "backgroundColor": "#f0f4f8",
                        },
                        page_size=10,
                    )
                )
            else:
                preview_children.append(
                    html.Div(
                        "No correlation data available.",
                        style={"color": "#64748b"},
                    )
                )

        # Hypothesis Test Results
        if "hypothesis_test_results" in report_types:
            preview_children.append(
                html.H4("Hypothesis Test Results", style={"marginTop": "12px"})
            )
            test_keys = ["t_test_engagement", "anova_engagement"]
            found_tests = False
            for key in test_keys:
                if key in analytics_results:
                    found_tests = True
                    result = analytics_results[key]
                    items = []
                    for k, v in result.items():
                        if isinstance(v, float):
                            items.append(html.Li(f"{k}: {v:.4f}"))
                        else:
                            items.append(html.Li(f"{k}: {v}"))
                    preview_children.append(
                        html.Div(
                            [
                                html.Strong(result.get("test", key)),
                                html.Ul(items),
                            ],
                            style={"marginBottom": "8px"},
                        )
                    )
            if not found_tests:
                preview_children.append(
                    html.Div(
                        "No hypothesis test results available.",
                        style={"color": "#64748b"},
                    )
                )

        # ML Model Results
        if "ml_model_results" in report_types:
            preview_children.append(
                html.H4("ML Model Results", style={"marginTop": "12px"})
            )
            ml_keys = ["regression", "classification", "clustering"]
            found_ml = False
            for key in ml_keys:
                if key in analytics_results:
                    found_ml = True
                    result = analytics_results[key]
                    metrics = []
                    skip = {
                        "model", "feature_importances", "cv_scores",
                        "labels", "cluster_centers", "confusion_matrix",
                        "features",
                    }
                    for k, v in result.items():
                        if k in skip:
                            continue
                        if isinstance(v, float):
                            metrics.append(html.Li(f"{k}: {v:.4f}"))
                        else:
                            metrics.append(html.Li(f"{k}: {v}"))
                    preview_children.append(
                        html.Div(
                            [
                                html.Strong(key.title()),
                                html.Ul(metrics),
                            ],
                            style={"marginBottom": "8px"},
                        )
                    )
            if not found_ml:
                preview_children.append(
                    html.Div(
                        "No ML model results available.",
                        style={"color": "#64748b"},
                    )
                )

        # Key findings summary
        if analytics_results:
            findings = _build_key_findings(analytics_results)
            if findings:
                preview_children.append(
                    html.Div(
                        [
                            html.H4("Key Findings", style={"marginTop": "16px"}),
                            html.Ul([html.Li(f) for f in findings]),
                        ],
                        className="alert-success",
                    )
                )

        return preview_children if preview_children else html.Div(
            "No data available for the selected sections.",
            style={"color": "#64748b", "fontStyle": "italic"},
        )

    # ------------------------------------------------------------------
    # 3. Export callbacks
    # ------------------------------------------------------------------
    @app.callback(
        Output("download-integrated", "data"),
        Input("btn-export-integrated", "n_clicks"),
        State("integrated-data-store", "data"),
        prevent_initial_call=True,
    )
    def export_integrated(n_clicks, integrated_data):
        if not n_clicks or not integrated_data:
            return no_update
        df = _store_to_df(integrated_data)
        return dcc.send_data_frame(df.to_csv, "integrated_dataset.csv", index=False)

    @app.callback(
        Output("download-features", "data"),
        Input("btn-export-features", "n_clicks"),
        State("featured-data-store", "data"),
        prevent_initial_call=True,
    )
    def export_features(n_clicks, featured_data):
        if not n_clicks or not featured_data:
            return no_update
        df = _store_to_df(featured_data)
        return dcc.send_data_frame(df.to_csv, "feature_dataset.csv", index=False)

    @app.callback(
        Output("download-summary", "data"),
        Input("btn-export-summary", "n_clicks"),
        State("featured-data-store", "data"),
        prevent_initial_call=True,
    )
    def export_summary(n_clicks, featured_data):
        if not n_clicks or not featured_data:
            return no_update
        df = _store_to_df(featured_data)
        report_config = _config.get("reporting", {})
        generator = ReportGenerator(report_config)
        stats_df = generator.generate_summary_stats(df)
        return dcc.send_data_frame(
            stats_df.to_csv, "summary_statistics.csv", index=True
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_results(report_types, analytics_results, featured_data):
    """Build a results dict filtered to selected report sections."""
    results = {}

    if "summary_statistics" in report_types and featured_data:
        try:
            df = _store_to_df(featured_data)
            report_config = _config.get("reporting", {})
            generator = ReportGenerator(report_config)
            stats = generator.generate_summary_stats(df)
            results["summary_statistics"] = stats.to_dict()
        except Exception:
            pass

    if "distribution_analysis" in report_types and "distributions" in analytics_results:
        results["distributions"] = analytics_results["distributions"]

    if "correlation_analysis" in report_types and "correlations" in analytics_results:
        results["correlations"] = analytics_results["correlations"]

    test_keys = ["t_test_engagement", "anova_engagement"]
    if "hypothesis_test_results" in report_types:
        for key in test_keys:
            if key in analytics_results:
                results[key] = analytics_results[key]

    ml_keys = ["regression", "classification", "clustering"]
    if "ml_model_results" in report_types:
        for key in ml_keys:
            if key in analytics_results:
                results[key] = analytics_results[key]

    return results


def _build_summary_dataframe(results, featured_data, generator):
    """Build a combined CSV-friendly DataFrame from results."""
    frames = []

    if "summary_statistics" in results:
        stats_df = pd.DataFrame(results["summary_statistics"])
        stats_df.insert(0, "section", "Summary Statistics")
        frames.append(stats_df)

    if "distributions" in results:
        dist_df = pd.DataFrame(results["distributions"]).T
        dist_df.index.name = "feature"
        dist_df = dist_df.reset_index()
        dist_df.insert(0, "section", "Distribution Analysis")
        frames.append(dist_df)

    if "correlations" in results:
        corr_df = pd.DataFrame(results["correlations"])
        corr_df.index.name = "feature"
        corr_df = corr_df.reset_index()
        corr_df.insert(0, "section", "Correlation Analysis")
        frames.append(corr_df)

    for key in ["t_test_engagement", "anova_engagement"]:
        if key in results:
            test_df = pd.DataFrame([results[key]])
            test_df.insert(0, "section", "Hypothesis Test")
            frames.append(test_df)

    for key in ["regression", "classification", "clustering"]:
        if key in results:
            ml_data = {
                k: v for k, v in results[key].items()
                if not isinstance(v, (list, dict))
            }
            ml_df = pd.DataFrame([ml_data])
            ml_df.insert(0, "section", f"ML - {key.title()}")
            frames.append(ml_df)

    if frames:
        return pd.concat(frames, ignore_index=True, sort=False)

    # Fallback: return featured data summary
    if featured_data:
        return generator.generate_summary_stats(_store_to_df(featured_data))

    return pd.DataFrame({"message": ["No data available for report"]})


def _build_key_findings(analytics_results):
    """Extract key findings from analytics results for preview."""
    findings = []

    for key in ["t_test_engagement", "anova_engagement"]:
        if key in analytics_results:
            r = analytics_results[key]
            sig = "significant" if r.get("significant") else "not significant"
            findings.append(
                f"{r.get('test', key)}: p={r.get('p_value', 0):.4f} ({sig})"
            )

    if "regression" in analytics_results:
        r = analytics_results["regression"]
        findings.append(
            f"Regression ({r.get('model_type', 'N/A')}): "
            f"R2={r.get('r2', 0):.3f}, RMSE={r.get('rmse', 0):.3f}"
        )

    if "classification" in analytics_results:
        r = analytics_results["classification"]
        findings.append(
            f"Classification ({r.get('model_type', 'N/A')}): "
            f"Accuracy={r.get('accuracy', 0):.3f}, F1={r.get('f1', 0):.3f}"
        )

    if "clustering" in analytics_results:
        r = analytics_results["clustering"]
        findings.append(
            f"Clustering ({r.get('method', 'N/A')}): "
            f"{r.get('n_clusters', '?')} clusters, "
            f"silhouette={r.get('silhouette_score', 0):.3f}"
        )

    return findings
