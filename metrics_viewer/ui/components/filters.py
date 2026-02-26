"""FilterPanel — reusable dropdown/checkbox filter controls."""

from __future__ import annotations

from typing import List, Optional

import dash_bootstrap_components as dbc
from dash import dcc, html

from ...domain.models import FilterOptions


def filter_panel(
    filter_options: FilterOptions,
    panel_id_prefix: str = "filter",
) -> html.Div:
    """Build a filter panel with dropdowns for source, category, scene, name."""

    def _dropdown(label: str, options: list, dropdown_id: str, multi: bool = True):
        return dbc.Col([
            html.Label(label, htmlFor=dropdown_id, className="mb-1"),
            dcc.Dropdown(
                id=dropdown_id,
                options=[{"label": str(o.value if hasattr(o, "value") else o),
                          "value": str(o.value if hasattr(o, "value") else o)}
                         for o in options],
                multi=multi,
                placeholder=f"All {label.lower()}s\u2026",
                style={
                    "backgroundColor": "#0b1e3d",
                    "color": "#e0e0e0",
                    "border": "1px solid #0f3460",
                },
            ),
        ], xs=12, sm=6, md=3)

    children = []

    if filter_options.sources:
        children.append(_dropdown(
            "Source", filter_options.sources,
            f"{panel_id_prefix}-source",
        ))

    if filter_options.categories:
        children.append(_dropdown(
            "Category", filter_options.categories,
            f"{panel_id_prefix}-category",
        ))

    if filter_options.scenes:
        children.append(_dropdown(
            "Scene", filter_options.scenes,
            f"{panel_id_prefix}-scene",
        ))

    if filter_options.names:
        children.append(_dropdown(
            "Strategy", filter_options.names,
            f"{panel_id_prefix}-name",
        ))

    return html.Div(
        dbc.Row(children, className="g-3"),
        className="filter-panel",
        role="search",
        **{"aria-label": "Filter controls"},
    )


def axis_selector(
    selector_id: str = "axis-selector",
    default: str = "end_to_end",
) -> html.Div:
    """Dropdown to pick a ComparisonAxis."""
    return html.Div([
        html.Label("Comparison Axis", htmlFor=selector_id, className="mb-1",
                    style={"fontSize": "0.8rem", "color": "#a0b4d0", "textTransform": "uppercase"}),
        dcc.Dropdown(
            id=selector_id,
            options=[
                {"label": "Compression Fidelity (decompressed vs original)", "value": "compression_fidelity"},
                {"label": "End-to-End (decompressed vs ground truth)", "value": "end_to_end"},
                {"label": "Training Baseline (original vs ground truth)", "value": "training_baseline"},
            ],
            value=default,
            clearable=False,
            style={
                "backgroundColor": "#0b1e3d",
                "color": "#e0e0e0",
                "border": "1px solid #0f3460",
            },
        ),
    ])
