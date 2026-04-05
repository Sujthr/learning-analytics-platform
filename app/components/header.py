"""Top header bar with branding badge."""

from dash import html


def create_header(title: str, subtitle: str = "", badges: list | None = None) -> html.Div:
    """Page header with title, subtitle, and optional status badges."""
    badge_elements = []
    for b in (badges or []):
        badge_elements.append(
            html.Span(
                b.get("text", ""),
                className=f"header-badge badge-{b.get('type', 'info')}",
                title=b.get("tip", ""),
            )
        )

    # Always show the developer badge
    badge_elements.append(
        html.Span(
            "SKT",
            className="header-badge badge-info",
            title="Developed by Sujit Kumar Thakur",
            style={"cursor": "default"},
        )
    )

    return html.Div([
        html.Div([
            html.H1(title, className="header-title"),
            html.Div(subtitle, className="header-subtitle") if subtitle else None,
        ]),
        html.Div(badge_elements, className="header-actions"),
    ], className="header")
