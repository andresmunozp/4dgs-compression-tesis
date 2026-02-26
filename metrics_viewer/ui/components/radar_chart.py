"""Radar / spider chart for multi-metric comparison."""

from __future__ import annotations

from typing import Dict, List, Optional

import plotly.graph_objects as go
from dash import dcc, html

from ...domain.enums import ComparisonAxis, MetricType
from ...domain.models import MetricRecord
from ...domain.services import _extract_metric

COLORS = [
    "#e94560", "#533483", "#2ecc71", "#f39c12", "#3498db",
    "#e74c3c", "#1abc9c", "#9b59b6", "#d35400", "#0f3460",
]

# Normalisation direction: True = higher is better
_HIGHER_BETTER = {
    MetricType.PSNR: True,
    MetricType.SSIM: True,
    MetricType.MS_SSIM: True,
    MetricType.VMAF: True,
    MetricType.COMPRESSION_RATIO: True,
    MetricType.SAVINGS_PCT: True,
    MetricType.QOE_SCORE: True,
    MetricType.LPIPS_VGG: False,
    MetricType.LPIPS_ALEX: False,
    MetricType.D_SSIM: False,
    MetricType.COMPRESS_TIME: False,
    MetricType.DECOMPRESS_TIME: False,
}


def _normalize(values: List[Optional[float]], higher_is_better: bool) -> List[float]:
    """Min-max normalize to [0, 1]. Inverts if lower is better."""
    clean = [v for v in values if v is not None]
    if not clean:
        return [0.0] * len(values)
    mn, mx = min(clean), max(clean)
    rng = mx - mn if mx != mn else 1.0

    result = []
    for v in values:
        if v is None:
            result.append(0.0)
        else:
            norm = (v - mn) / rng
            result.append(norm if higher_is_better else 1.0 - norm)
    return result


def build_radar_figure(
    records: List[MetricRecord],
    metrics: List[MetricType],
    axis: Optional[ComparisonAxis] = None,
    title: str = "Multi-Metric Radar",
) -> go.Figure:
    """Build a spider/radar chart figure."""
    fig = go.Figure()

    labels = [m.value.upper().replace("_", " ") for m in metrics]

    # Collect raw values per metric for normalization
    raw: Dict[MetricType, List[Optional[float]]] = {m: [] for m in metrics}
    for record in records:
        for m in metrics:
            raw[m].append(_extract_metric(record, m, axis))

    # Normalize
    normed: Dict[MetricType, List[float]] = {}
    for m in metrics:
        higher = _HIGHER_BETTER.get(m, True)
        normed[m] = _normalize(raw[m], higher)

    # Plot each record
    for i, record in enumerate(records):
        values = [normed[m][i] for m in metrics]
        values_closed = values + [values[0]]  # close the polygon
        labels_closed = labels + [labels[0]]

        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            name=record.display_name,
            fill="toself",
            fillcolor=f"rgba({_hex_to_rgb(COLORS[i % len(COLORS)])}, 0.1)",
            line=dict(color=COLORS[i % len(COLORS)], width=2),
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(22,33,62,0.8)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#0f3460", linecolor="#0f3460"),
            angularaxis=dict(gridcolor="#0f3460", linecolor="#0f3460"),
        ),
        title=title,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=60, r=60, t=60, b=60),
        height=450,
    )

    return fig


def radar_chart(
    records: List[MetricRecord],
    metrics: List[MetricType],
    axis: Optional[ComparisonAxis] = None,
    chart_id: str = "radar-chart",
    title: str = "Multi-Metric Radar",
) -> html.Div:
    """Spider/radar chart wrapped in a Dash component."""
    fig = build_radar_figure(records, metrics, axis, title)
    return html.Div(
        dcc.Graph(id=chart_id, figure=fig, config={"displaylogo": False}),
        className="chart-container",
    )


def _hex_to_rgb(hex_color: str) -> str:
    """'#e94560' → '233, 69, 96'."""
    h = hex_color.lstrip("#")
    return ", ".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))
