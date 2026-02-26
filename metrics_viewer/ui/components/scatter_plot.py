"""Scatter plot component with optional Pareto frontier."""

from __future__ import annotations

from typing import List, Optional

import plotly.graph_objects as go
from dash import dcc, html

from ...domain.enums import ComparisonAxis, MetricType
from ...domain.models import MetricRecord
from ...domain.services import _extract_metric

COLORS = [
    "#e94560", "#533483", "#2ecc71", "#f39c12", "#3498db",
    "#e74c3c", "#1abc9c", "#9b59b6", "#d35400", "#0f3460",
]


def _pareto_frontier(xs: List[float], ys: List[float]) -> tuple:
    """Compute 2D Pareto frontier (maximize both x and y)."""
    points = sorted(zip(xs, ys), key=lambda p: (-p[0], -p[1]))
    frontier_x, frontier_y = [], []
    max_y = float("-inf")
    for x, y in points:
        if y >= max_y:
            frontier_x.append(x)
            frontier_y.append(y)
            max_y = y
    return frontier_x, frontier_y


def build_scatter_figure(
    records: List[MetricRecord],
    x_metric: MetricType = MetricType.COMPRESSION_RATIO,
    y_metric: MetricType = MetricType.PSNR,
    axis: Optional[ComparisonAxis] = None,
    show_pareto: bool = True,
    title: str = "Compression Ratio vs Quality",
) -> go.Figure:
    """Build a scatter plot figure with optional Pareto frontier."""
    fig = go.Figure()

    xs, ys, names = [], [], []
    for i, record in enumerate(records):
        x = _extract_metric(record, x_metric, axis)
        y = _extract_metric(record, y_metric, axis)
        if x is None or y is None:
            continue

        xs.append(x)
        ys.append(y)
        names.append(record.display_name)

        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=14, color=COLORS[i % len(COLORS)], line=dict(width=1, color="#e0e0e0")),
            text=[record.display_name],
            textposition="top center",
            textfont=dict(size=10),
            name=record.display_name,
            hovertemplate=(
                f"<b>{record.display_name}</b><br>"
                f"{x_metric.value}: %{{x:.3f}}<br>"
                f"{y_metric.value}: %{{y:.3f}}<extra></extra>"
            ),
        ))

    # Pareto frontier
    if show_pareto and len(xs) >= 2:
        px, py = _pareto_frontier(xs, ys)
        fig.add_trace(go.Scatter(
            x=px, y=py,
            mode="lines",
            line=dict(color="#f39c12", width=2, dash="dash"),
            name="Pareto Frontier",
            showlegend=True,
        ))

    x_label = x_metric.value.upper().replace("_", " ")
    y_label = y_metric.value.upper().replace("_", " ")

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,33,62,0.8)",
        font=dict(color="#e0e0e0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=50),
        height=450,
    )

    return fig


def scatter_plot(
    records: List[MetricRecord],
    x_metric: MetricType = MetricType.COMPRESSION_RATIO,
    y_metric: MetricType = MetricType.PSNR,
    axis: Optional[ComparisonAxis] = None,
    show_pareto: bool = True,
    chart_id: str = "scatter-plot",
    title: str = "Compression Ratio vs Quality",
) -> html.Div:
    """Scatter plot wrapped in a Dash component."""
    fig = build_scatter_figure(records, x_metric, y_metric, axis, show_pareto, title)
    return html.Div(
        dcc.Graph(id=chart_id, figure=fig, config={"displaylogo": False}),
        className="chart-container",
    )
