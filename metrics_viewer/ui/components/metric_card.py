"""MetricCard — summary card showing a single metric value."""

from __future__ import annotations

from typing import Optional

import dash_bootstrap_components as dbc
from dash import html


def metric_card(
    label: str,
    value: float | str,
    unit: str = "",
    delta: Optional[float] = None,
    delta_label: str = "vs baseline",
    icon: str = "",
    color: str = "#e94560",
) -> dbc.Card:
    """Render a compact card with a metric value and optional delta.

    Args:
        label: Short metric name (e.g. "PSNR").
        value: The metric value.
        unit: Unit string (e.g. "dB", "%").
        delta: Optional delta vs reference.
        delta_label: Description of the delta.
        icon: Optional emoji/icon character.
        color: Accent colour for the value.
    """
    # Format value
    if isinstance(value, float):
        if abs(value) >= 100:
            val_str = f"{value:,.1f}"
        elif abs(value) >= 1:
            val_str = f"{value:.2f}"
        else:
            val_str = f"{value:.4f}"
    else:
        val_str = str(value)

    # Delta badge
    delta_el = None
    if delta is not None:
        sign = "+" if delta >= 0 else ""
        css_class = "metric-delta positive" if delta >= 0 else "metric-delta negative"
        delta_el = html.Div(
            f"{sign}{delta:.2f} {delta_label}",
            className=css_class,
        )

    return dbc.Card(
        dbc.CardBody([
            html.Div(
                f"{icon} {label}" if icon else label,
                className="metric-label",
            ),
            html.Div([
                html.Span(val_str, className="metric-value", style={"color": color}),
                html.Span(unit, className="metric-unit") if unit else None,
            ]),
            delta_el,
        ]),
        className="metric-card",
    )


def metric_card_row(cards: list, cols_per_card: int = 3) -> dbc.Row:
    """Wrap a list of metric_card elements into a responsive Row."""
    return dbc.Row(
        [dbc.Col(card, xs=12, sm=6, md=cols_per_card) for card in cards],
        className="g-3 mb-4",
    )
