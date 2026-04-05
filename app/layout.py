"""Main application layout with sidebar navigation and page routing."""

from dash import dcc, html


def create_layout():
    """Return the top-level app layout with routing, sidebar, and shared stores."""
    return html.Div([
        # URL routing
        dcc.Location(id="url", refresh=False),

        # ===== Shared data stores (session-scoped) =====
        dcc.Store(id="session-store", storage_type="memory", data={}),
        dcc.Store(id="uploaded-data-store", storage_type="memory", data={}),
        dcc.Store(id="cleaned-data-store", storage_type="memory", data={}),
        dcc.Store(id="integrated-data-store", storage_type="memory", data=None),
        dcc.Store(id="featured-data-store", storage_type="memory", data=None),
        dcc.Store(id="analytics-results-store", storage_type="memory", data={}),
        dcc.Store(id="pipeline-status-store", storage_type="memory", data={
            "upload": False,
            "clean": False,
            "integrate": False,
            "features": False,
            "analytics": False,
        }),

        # ===== Sidebar =====
        html.Div(id="sidebar-container"),

        # ===== Main content area =====
        html.Div([
            # Header
            html.Div(id="header-container"),
            # Page content
            html.Div(id="page-content", className="page-content"),
        ], className="main-content"),
    ])
