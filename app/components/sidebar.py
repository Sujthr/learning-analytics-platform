"""Sidebar navigation with branding and tooltips."""

from dash import dcc, html

NAV_ITEMS = [
    {"section": "MAIN"},
    {"key": "home",      "icon": "\u25cb", "label": "Dashboard",       "href": "/",          "tip": "Overview & KPIs"},
    {"key": "upload",    "icon": "\u21e1", "label": "Upload Data",     "href": "/upload",    "tip": "Import CSV / Excel files"},
    {"key": "manage",    "icon": "\u2699", "label": "Data Management", "href": "/manage",    "tip": "Clean, integrate & engineer features"},
    {"section": "ANALYSIS"},
    {"key": "analyze",   "icon": "\u03a3", "label": "Analytics",       "href": "/analyze",   "tip": "EDA, hypothesis tests & ML models"},
    {"key": "visualize", "icon": "\u2587", "label": "Visualization",   "href": "/visualize", "tip": "Interactive charts & plots"},
    {"key": "reports",   "icon": "\u2637", "label": "Reports",         "href": "/reports",   "tip": "Generate & download PDF / CSV"},
]


def create_sidebar(active_page: str = "home") -> html.Div:
    """Build the sidebar with branding, nav links with tooltips, and footer."""
    nav_children = []

    for item in NAV_ITEMS:
        if "section" in item:
            nav_children.append(html.Div(item["section"], className="sidebar-section"))
        else:
            is_active = item["key"] == active_page
            nav_children.append(
                dcc.Link(
                    html.Div([
                        html.Span(item["icon"], className="nav-icon"),
                        html.Span(item["label"]),
                    ], className=f"nav-item{'  active' if is_active else ''}",
                       title=item["tip"]),
                    href=item["href"],
                    style={"textDecoration": "none"},
                )
            )

    return html.Div([
        # ── Brand ──
        html.Div([
            html.Img(
                src="/assets/logo.svg",
                style={"width": "38px", "height": "38px", "borderRadius": "10px"},
            ),
            html.Div([
                html.Div("Learning Analytics", className="sidebar-brand-text"),
                html.Div("by Sujit Kumar Thakur", className="sidebar-brand-sub"),
            ]),
        ], className="sidebar-brand"),

        # ── Navigation ──
        html.Nav(nav_children, className="sidebar-nav"),

        # ── Footer ──
        html.Div([
            html.Div("v1.0.0", style={"marginBottom": "4px"}),
            html.Div(
                "\u00a9 2026 Sujit Kumar Thakur",
                style={"fontSize": "10px", "opacity": "0.5"},
            ),
            html.Div(
                "All rights reserved",
                style={"fontSize": "9px", "opacity": "0.35"},
            ),
        ], className="sidebar-footer"),
    ], className="sidebar")
