"""Per-frame line chart component."""

from __future__ import annotations

from typing import List, Optional

import plotly.graph_objects as go
from dash import dcc, html

from ...domain.enums import ComparisonAxis
from ...domain.models import MetricRecord, PerFrameMetrics

COLORS = [
    "#e94560", "#533483", "#2ecc71", "#f39c12", "#3498db",
    "#e74c3c", "#1abc9c", "#9b59b6", "#d35400", "#0f3460",
]


def build_per_frame_figure(
    records: List[MetricRecord],
    metric_name: str = "psnr",
    axis: Optional[ComparisonAxis] = None,
    title: str = "Per-Frame Analysis",
) -> go.Figure:
    """Build a per-frame line chart figure."""
    fig = go.Figure()

    for i, record in enumerate(records):
        pf = record.get_per_frame(axis)
        if pf is None:
            continue

        values = getattr(pf, metric_name, None)
        if values is None:
            continue

        fig.add_trace(go.Scatter(
            x=pf.frame_indices,
            y=values,
            mode="lines",
            name=record.display_name,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            hovertemplate=f"<b>{record.display_name}</b><br>Frame %{{x}}<br>{metric_name.upper()}: %{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Frame",
        yaxis_title=metric_name.upper(),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,33,62,0.8)",
        font=dict(color="#e0e0e0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=40),
        height=400,
        hovermode="x unified",
    )

    return fig


def per_frame_chart(
    records: List[MetricRecord],
    metric_name: str = "psnr",
    axis: Optional[ComparisonAxis] = None,
    chart_id: str = "per-frame-chart",
    title: str = "Per-Frame Analysis",
) -> html.Div:
    """Per-frame line chart wrapped in a Dash component."""
    fig = build_per_frame_figure(records, metric_name, axis, title)
    return html.Div(
        dcc.Graph(id=chart_id, figure=fig, config={"displaylogo": False}),
        className="chart-container",
    )


def build_heatmap_figure(
    records: List[MetricRecord],
    metric_name: str = "psnr",
    axis: Optional[ComparisonAxis] = None,
    title: str = "Per-Frame Heatmap",
) -> go.Figure:
    """Build a per-frame heatmap figure."""
    z = []
    y_labels = []

    for record in records:
        pf = record.get_per_frame(axis)
        if pf is None:
            continue
        values = getattr(pf, metric_name, None)
        if values is None:
            continue
        z.append(values)
        y_labels.append(record.display_name)

    if not z:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text="No per-frame data", showarrow=False,
                              font=dict(size=14, color="#6c7a8e"))],
        )
        return fig

    x_labels = list(range(len(z[0])))

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale="Viridis",
        colorbar=dict(title=metric_name.upper()),
        hovertemplate="Strategy: %{y}<br>Frame: %{x}<br>Value: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Frame",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,33,62,0.8)",
        font=dict(color="#e0e0e0"),
        margin=dict(l=150, r=20, t=60, b=40),
        height=max(250, 80 * len(y_labels)),
    )

    return fig


def per_frame_heatmap(
    records: List[MetricRecord],
    metric_name: str = "psnr",
    axis: Optional[ComparisonAxis] = None,
    chart_id: str = "per-frame-heatmap",
    title: str = "Per-Frame Heatmap",
) -> html.Div:
    """Per-frame heatmap wrapped in a Dash component."""
    fig = build_heatmap_figure(records, metric_name, axis, title)
    return html.Div(
        dcc.Graph(id=chart_id, figure=fig, config={"displaylogo": False}),
        className="chart-container",
    )
