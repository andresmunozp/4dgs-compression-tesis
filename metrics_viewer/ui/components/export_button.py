"""Export button + download component for the UI.

Provides a reusable ``export_button()`` that renders a dropdown-style
button group (CSV / PNG) along with a ``dcc.Download`` component.
"""

from __future__ import annotations

from typing import Optional

import dash_bootstrap_components as dbc
from dash import dcc, html


def export_button(
    button_id: str = "export-btn",
    download_id: str = "export-download",
    className: str = "",
) -> html.Div:
    """Render an export button with format selector and a hidden Download target.

    Args:
        button_id: Base ID used for the button group.
        download_id: ID of the ``dcc.Download`` component.
        className: Additional CSS classes.

    Returns:
        An ``html.Div`` containing the button group and the Download.
    """
    return html.Div([
        dbc.ButtonGroup([
            dbc.Button(
                [html.Span("\U0001f4e5", **{"aria-hidden": "true"}), " Export CSV"],
                id=f"{button_id}-csv",
                color="info",
                outline=True,
                size="sm",
                className="me-1",
                title="Export data as CSV file",
            ),
            dbc.Button(
                [html.Span("\U0001f5bc\ufe0f", **{"aria-hidden": "true"}), " Export PNG"],
                id=f"{button_id}-png",
                color="info",
                outline=True,
                size="sm",
                title="Export chart as PNG image",
            ),
        ]),
        dcc.Download(id=download_id),
    ], className=f"d-inline-block {className}")
