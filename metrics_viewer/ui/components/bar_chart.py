"""Bar chart component for comparing metrics across strategies."""

from __future__ import annotations

from typing import Dict, List, Optional

import plotly.graph_objects as go
from dash import dcc, html

from ...domain.enums import ComparisonAxis, MetricType
from ...domain.models import MetricRecord
from ...domain.services import _extract_metric


# Colour palette for strategies
COLORS = [
    "#e94560", "#533483", "#0f3460", "#2ecc71", "#f39c12",
    "#3498db", "#e74c3c", "#1abc9c", "#9b59b6", "#d35400",
]


def build_bar_figure(
    records: List[MetricRecord],
    metrics: List[MetricType],
    axis: Optional[ComparisonAxis] = None,
    title: str = "Metrics Comparison",
) -> go.Figure:
    """Build a grouped bar chart figure (one group per metric, one bar per strategy)."""
    fig = go.Figure()

    for i, record in enumerate(records):
        values = []
        labels = []
        for m in metrics:
            val = _extract_metric(record, m, axis)
            values.append(val if val is not None else 0)
            labels.append(m.value.upper().replace("_", " "))

        fig.add_trace(go.Bar(
            name=record.display_name,
            x=labels,
            y=values,
            marker_color=COLORS[i % len(COLORS)],
            text=[f"{v:.3f}" if isinstance(v, float) else str(v) for v in values],
            textposition="auto",
        ))

    fig.update_layout(
        barmode="group",
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,33,62,0.8)",
        font=dict(color="#e0e0e0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        height=400,
    )

    return fig


def bar_chart(
    records: List[MetricRecord],
    metrics: List[MetricType],
    axis: Optional[ComparisonAxis] = None,
    chart_id: str = "bar-chart",
    title: str = "Metrics Comparison",
) -> html.Div:
    """Grouped bar chart wrapped in a Dash component."""
    fig = build_bar_figure(records, metrics, axis, title)
    return html.Div(
        dcc.Graph(id=chart_id, figure=fig, config={"displaylogo": False}),
        className="chart-container",
    )


def build_timing_figure(
    records: List[MetricRecord],
    title: str = "Timing Breakdown",
) -> go.Figure:
    """Build a stacked bar chart figure for time breakdowns."""
    fig = go.Figure()

    time_fields = [
        ("Compress", lambda r: r.timing_metrics.compress_time_s if r.timing_metrics else None),
        ("Decompress", lambda r: r.timing_metrics.decompress_time_s if r.timing_metrics else None),
        ("Train", lambda r: r.timing_metrics.train_time_s if r.timing_metrics else None),
        ("Render", lambda r: r.timing_metrics.render_time_s if r.timing_metrics else None),
    ]

    names = [r.display_name for r in records]

    for i, (label, getter) in enumerate(time_fields):
        vals = [getter(r) or 0 for r in records]
        if any(v > 0 for v in vals):
            fig.add_trace(go.Bar(
                name=label,
                x=names,
                y=vals,
                marker_color=COLORS[i % len(COLORS)],
            ))

    fig.update_layout(
        barmode="stack",
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,33,62,0.8)",
        font=dict(color="#e0e0e0"),
        yaxis_title="Time (s)",
        margin=dict(l=40, r=20, t=60, b=40),
        height=400,
    )

    return fig


def stacked_bar_timing(
    records: List[MetricRecord],
    chart_id: str = "timing-bar",
    title: str = "Timing Breakdown",
) -> html.Div:
    """Stacked bar chart wrapped in a Dash component."""
    fig = build_timing_figure(records, title)
    return html.Div(
        dcc.Graph(id=chart_id, figure=fig, config={"displaylogo": False}),
        className="chart-container",
    )
