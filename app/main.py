"""Learning Analytics Platform - Main Application Entry Point.

Production-ready web application with modular page routing,
session state management, and scalable architecture.

Usage:
    python -m app.main
    # or
    python app/main.py
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dash import Dash, Input, Output, State, callback_context, html, dcc, no_update

from app.layout import create_layout

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    assets_folder=str(Path(__file__).parent / "assets"),
    title="LAP | Sujit Kumar Thakur",
    update_title="Processing...",
)

server = app.server  # expose for WSGI servers (gunicorn, waitress)

app.layout = create_layout()

# ---------------------------------------------------------------------------
# Import page modules (after app creation to avoid circular imports)
# ---------------------------------------------------------------------------
from app.pages import home, upload, manage, analyze, visualize, reports
from app.components.sidebar import create_sidebar
from app.components.header import create_header

# Register every page's callbacks on this app instance
home.register_callbacks(app)
upload.register_callbacks(app)
manage.register_callbacks(app)
analyze.register_callbacks(app)
visualize.register_callbacks(app)
reports.register_callbacks(app)

# ---------------------------------------------------------------------------
# Page titles & subtitles
# ---------------------------------------------------------------------------
PAGE_META = {
    "/":          ("Dashboard",        "Pipeline overview"),
    "/upload":    ("Data Upload",      "Coursera, LMS & academic records"),
    "/manage":    ("Data Management",  "Clean, integrate & engineer features"),
    "/analyze":   ("Analytics",        "EDA, hypothesis tests & ML"),
    "/visualize": ("Visualization",    "Interactive charts"),
    "/reports":   ("Reports",          "PDF & CSV export"),
}

# ---------------------------------------------------------------------------
# Core routing callback
# ---------------------------------------------------------------------------
@app.callback(
    [Output("sidebar-container", "children"),
     Output("header-container", "children"),
     Output("page-content", "children")],
    [Input("url", "pathname")],
)
def route_page(pathname):
    pathname = pathname or "/"

    # Determine active page key for sidebar highlight
    page_key = pathname.rstrip("/") or "/"
    active_map = {
        "/": "home", "/upload": "upload", "/manage": "manage",
        "/analyze": "analyze", "/visualize": "visualize", "/reports": "reports",
    }
    active = active_map.get(page_key, "home")

    # Build sidebar
    sidebar = create_sidebar(active_page=active)

    # Build header
    title, subtitle = PAGE_META.get(page_key, ("Page", ""))
    header = create_header(title, subtitle)

    # Select page layout
    pages = {
        "/":          home.layout,
        "/upload":    upload.layout,
        "/manage":    manage.layout,
        "/analyze":   analyze.layout,
        "/visualize": visualize.layout,
        "/reports":   reports.layout,
    }
    page_func = pages.get(page_key, home.layout)
    content = page_func()

    return sidebar, header, content


# ---------------------------------------------------------------------------
# Pipeline status aggregation – keeps session-store in sync
# ---------------------------------------------------------------------------
@app.callback(
    Output("session-store", "data"),
    [Input("uploaded-data-store", "data"),
     Input("cleaned-data-store", "data"),
     Input("integrated-data-store", "data"),
     Input("featured-data-store", "data"),
     Input("analytics-results-store", "data")],
    [State("session-store", "data")],
)
def sync_session(uploaded, cleaned, integrated, featured, analytics, session):
    import json as _json
    session = session or {}
    session["has_uploads"] = bool(uploaded)
    session["has_cleaned"] = bool(cleaned)
    session["has_integrated"] = bool(integrated)
    session["has_featured"] = bool(featured)
    session["has_analytics"] = bool(analytics)

    # Count datasets and students
    if isinstance(uploaded, dict):
        session["n_datasets"] = len(uploaded)
        session["datasets_loaded"] = len(uploaded)
        # Try to count unique students across all uploaded sources
        total_students = set()
        for key, val in uploaded.items():
            try:
                meta = _json.loads(val) if isinstance(val, str) else val
                total_students.add(meta.get("rows", 0))
            except Exception:
                pass
        session["total_students"] = sum(total_students) if total_students else "--"
    else:
        session["n_datasets"] = 0
        session["datasets_loaded"] = 0
        session["total_students"] = "--"

    # Extract KPIs from featured data if available
    if featured:
        try:
            import io as _io, pandas as _pd
            if isinstance(featured, str):
                df = _pd.read_json(_io.StringIO(featured), orient="split")
            elif isinstance(featured, dict):
                df = _pd.read_json(_io.StringIO(_json.dumps(featured)), orient="split")
            else:
                df = None
            if df is not None:
                if "student_id" in df.columns:
                    session["total_students"] = df["student_id"].nunique()
                else:
                    session["total_students"] = len(df)
                if "engagement_score" in df.columns:
                    session["avg_engagement"] = round(df["engagement_score"].mean() * 100, 1)
                if "dropout" in df.columns:
                    session["dropout_rate"] = round(df["dropout"].mean() * 100, 1)
        except Exception:
            pass

    # Build pipeline log
    log = []
    if session.get("has_uploads"):
        log.append(f"Loaded {session.get('datasets_loaded', 0)} dataset(s)")
    if session.get("has_cleaned"):
        log.append("Data cleaned successfully")
    if session.get("has_integrated"):
        log.append("Datasets integrated")
    if session.get("has_featured"):
        log.append("Features engineered")
    if session.get("has_analytics"):
        log.append("Analytics completed")
    session["pipeline_log"] = log

    return session


@app.callback(
    Output("pipeline-status-store", "data"),
    [Input("session-store", "data")],
)
def update_pipeline_status(session):
    session = session or {}
    return {
        "upload": session.get("has_uploads", False),
        "clean": session.get("has_cleaned", False),
        "integrate": session.get("has_integrated", False),
        "features": session.get("has_featured", False),
        "analytics": session.get("has_analytics", False),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Learning Analytics Platform")
    print("  Developed by Sujit Kumar Thakur")
    print("  Open http://localhost:8050 in your browser")
    print("=" * 60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=8050)
