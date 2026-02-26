"""PNG exporter — renders a summary figure to PNG bytes.

Implements the ``IResultExporter`` protocol from ``domain.ports``.
Requires the ``kaleido`` package for Plotly static image export.
If kaleido is not available, falls back to SVG via plotly's built-in
writer.
"""

from __future__ import annotations

import io
from typing import List

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...domain.enums import ComparisonAxis, MetricType
from ...domain.models import MetricRecord
from ...domain.ports import ExportConfig
from ...domain.services import _extract_metric


COLORS = [
    "#e94560", "#533483", "#2ecc71", "#f39c12", "#3498db",
    "#e74c3c", "#1abc9c", "#9b59b6", "#d35400", "#0f3460",
]


def _build_summary_figure(records: List[MetricRecord],
                          config: ExportConfig) -> go.Figure:
    """Build a multi-subplot summary of all records for export."""
    metrics = [MetricType.PSNR, MetricType.SSIM, MetricType.LPIPS_VGG]
    metric_labels = [m.value.upper().replace("_", " ") for m in metrics]

    axis_str = config.extra.get("axis")
    axis = ComparisonAxis(axis_str) if axis_str else None

    has_compression = any(r.compression_metrics for r in records)
    num_cols = 2 if has_compression else 1
    subplot_titles = ["Quality Metrics"]
    if has_compression:
        subplot_titles.append("Compression Ratio")

    fig = make_subplots(
        rows=1, cols=num_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.15,
    )

    names = [r.display_name for r in records]

    # Quality bars
    for i, metric in enumerate(metrics):
        values = [_extract_metric(r, metric, axis) or 0 for r in records]
        fig.add_trace(
            go.Bar(
                x=names,
                y=values,
                name=metric_labels[i],
                marker_color=COLORS[i % len(COLORS)],
                text=[f"{v:.3f}" for v in values],
                textposition="auto",
            ),
            row=1, col=1,
        )

    # Compression ratio bars
    if has_compression:
        ratios = [
            r.compression_metrics.compression_ratio if r.compression_metrics else 0
            for r in records
        ]
        fig.add_trace(
            go.Bar(
                x=names,
                y=ratios,
                name="Ratio",
                marker_color="#3498db",
                text=[f"{v:.1f}×" for v in ratios],
                textposition="auto",
                showlegend=False,
            ),
            row=1, col=num_cols,
        )

    title = config.title or "4DGS Metrics Summary"
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#16213e",
        plot_bgcolor="#0b1e3d",
        font=dict(color="#e0e0e0"),
        barmode="group",
        width=config.width,
        height=config.height,
        margin=dict(l=60, r=40, t=80, b=100),
    )

    return fig


class PngExporter:
    """Exports MetricRecords to PNG bytes using a plotly summary figure."""

    @property
    def supported_formats(self) -> List[str]:
        return ["png", "svg"]

    def export(self, records: List[MetricRecord], config: ExportConfig) -> bytes:
        """Render a summary figure and return image bytes."""
        fig = _build_summary_figure(records, config)

        fmt = config.format if config.format in self.supported_formats else "png"

        try:
            return fig.to_image(format=fmt, width=config.width, height=config.height)
        except (ValueError, ImportError):
            # kaleido not installed — fallback to HTML bytes
            html_str = fig.to_html(include_plotlyjs="cdn")
            return html_str.encode("utf-8")
