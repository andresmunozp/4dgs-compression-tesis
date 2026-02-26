"""Compression Detail page — deep dive into a single compression strategy.

Shows pipeline stages, file size waterfall, per-frame quality curves,
model info, and side-by-side quality axis comparison.
"""

from __future__ import annotations

from typing import List

import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import dcc, html

from ...domain.enums import ComparisonAxis, MetricType, ResultSource
from ...domain.models import MetricRecord, PipelineStageStats
from ...domain.services import MetricsService, _extract_metric
from ..components.metric_card import metric_card, metric_card_row
from ..components.comparison_table import comparison_table
from ..components.per_frame_chart import build_per_frame_figure

COLORS = [
    "#e94560", "#533483", "#2ecc71", "#f39c12", "#3498db",
    "#e74c3c", "#1abc9c", "#9b59b6", "#d35400", "#0f3460",
]

AXIS_LABELS = {
    ComparisonAxis.COMPRESSION_FIDELITY: "Compression Fidelity",
    ComparisonAxis.END_TO_END: "End-to-End",
    ComparisonAxis.TRAINING_BASELINE: "Training Baseline",
}


def _build_waterfall(record: MetricRecord) -> go.Figure:
    """Waterfall chart showing file size reduction through pipeline stages."""
    fig = go.Figure()

    cm = record.compression_metrics
    if not cm or cm.original_size_bytes == 0:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text="No compression data", showarrow=False,
                              font=dict(size=14, color="#6c7a8e"))],
        )
        return fig

    orig_mb = cm.original_size_bytes / 1e6
    comp_mb = cm.compressed_size_bytes / 1e6
    saved_mb = orig_mb - comp_mb

    # If pipeline stages exist, show them
    if record.pipeline_stats:
        labels = ["Original"]
        values = [orig_mb]
        measures = ["absolute"]

        running = orig_mb
        for stage in record.pipeline_stats:
            stage_saved = running * (stage.savings_pct / 100.0)
            labels.append(stage.strategy)
            values.append(-stage_saved)
            measures.append("relative")
            running -= stage_saved

        labels.append("Final")
        values.append(comp_mb)
        measures.append("total")

        fig.add_trace(go.Waterfall(
            x=labels,
            y=values,
            measure=measures,
            connector=dict(line=dict(color="#0f3460", width=1)),
            increasing=dict(marker=dict(color="#2ecc71")),
            decreasing=dict(marker=dict(color="#e94560")),
            totals=dict(marker=dict(color="#3498db")),
            textposition="outside",
            text=[f"{abs(v):.2f} MB" for v in values],
            textfont=dict(size=11),
        ))
    else:
        # Simple two-bar comparison
        fig.add_trace(go.Waterfall(
            x=["Original", "Savings", "Compressed"],
            y=[orig_mb, -saved_mb, comp_mb],
            measure=["absolute", "relative", "total"],
            connector=dict(line=dict(color="#0f3460", width=1)),
            increasing=dict(marker=dict(color="#2ecc71")),
            decreasing=dict(marker=dict(color="#e94560")),
            totals=dict(marker=dict(color="#3498db")),
            textposition="outside",
            text=[f"{orig_mb:.2f} MB", f"-{saved_mb:.2f} MB", f"{comp_mb:.2f} MB"],
            textfont=dict(size=11),
        ))

    fig.update_layout(
        title="File Size Waterfall",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,33,62,0.8)",
        font=dict(color="#e0e0e0"),
        yaxis_title="Size (MB)",
        margin=dict(l=50, r=20, t=60, b=40),
        height=380,
        showlegend=False,
    )

    return fig


def _build_axis_comparison(record: MetricRecord) -> go.Figure:
    """Grouped bar chart comparing quality metrics across all 3 axes."""
    fig = go.Figure()

    metrics = [MetricType.PSNR, MetricType.SSIM, MetricType.LPIPS_VGG, MetricType.VMAF]
    metric_labels = [m.value.upper().replace("_", " ") for m in metrics]

    for i, (axis, label) in enumerate(AXIS_LABELS.items()):
        q = record.get_quality(axis)
        if not q:
            continue
        values = [_extract_metric(record, m, axis) or 0 for m in metrics]
        fig.add_trace(go.Bar(
            name=label,
            x=metric_labels,
            y=values,
            marker_color=COLORS[i % len(COLORS)],
            text=[f"{v:.3f}" if v else "—" for v in values],
            textposition="auto",
        ))

    fig.update_layout(
        barmode="group",
        title="Quality Across Comparison Axes",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,33,62,0.8)",
        font=dict(color="#e0e0e0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=40),
        height=400,
    )

    return fig


def _build_pipeline_table(stages: List[PipelineStageStats]) -> html.Div:
    """Table showing pipeline stage details."""
    if not stages:
        return html.P("No pipeline stage data.", className="text-muted")

    rows = []
    for i, s in enumerate(stages):
        rows.append(html.Tr([
            html.Td(str(i + 1), style={"textAlign": "center"}),
            html.Td(s.strategy),
            html.Td(f"{s.ratio:.2f}×", style={"textAlign": "center"}),
            html.Td(f"{s.savings_pct:.1f}%", style={"textAlign": "center"}),
            html.Td(f"{s.compress_time_s:.3f}s", style={"textAlign": "center"}),
            html.Td(f"{s.decompress_time_s:.3f}s", style={"textAlign": "center"}),
        ]))

    return html.Div(
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("#", style={"width": "50px"}),
                html.Th("Strategy"),
                html.Th("Ratio"),
                html.Th("Savings"),
                html.Th("Compress"),
                html.Th("Decompress"),
            ])),
            html.Tbody(rows),
        ], bordered=True, striped=True, hover=True, color="dark",
           className="mt-2", style={"fontSize": "0.85rem"}),
        style={"overflowX": "auto"},
    )


def _model_info_card(record: MetricRecord) -> html.Div:
    """Show model info (gaussians, SH degree)."""
    mi = record.model_info
    if not mi:
        return html.Div()

    items = []
    if mi.num_gaussians_original is not None:
        items.append(html.Li(f"Gaussians (original): {mi.num_gaussians_original:,}"))
    if mi.num_gaussians_compressed is not None:
        items.append(html.Li(f"Gaussians (compressed): {mi.num_gaussians_compressed:,}"))
    if mi.sh_degree_original is not None:
        items.append(html.Li(f"SH degree (original): {mi.sh_degree_original}"))
    if mi.sh_degree_compressed is not None:
        items.append(html.Li(f"SH degree (compressed): {mi.sh_degree_compressed}"))
    if mi.num_ply_files is not None:
        items.append(html.Li(f"PLY files: {mi.num_ply_files}"))
    if mi.total_ply_size_bytes is not None:
        gb = mi.total_ply_size_bytes / 1e9
        items.append(html.Li(f"Total PLY size: {gb:.2f} GB"))

    if not items:
        return html.Div()

    return html.Div([
        html.H6("🧊 Model Info", className="mt-3", style={"color": "#a0b4d0"}),
        html.Ul(items, style={"color": "#6c7a8e", "fontSize": "0.85rem"}),
    ])


def build_compression_detail(service: MetricsService) -> html.Div:
    """Construct the Compression Detail page with strategy selector."""
    # Get records with compression data
    all_records = service.get_all_records()
    records_with_comp = [
        r for r in all_records
        if r.compression_metrics is not None and r.compression_metrics.original_size_bytes > 0
    ]

    if not records_with_comp:
        return html.Div([
            html.H4("Compression Detail", className="section-title"),
            html.P("No compression data available.", className="text-muted"),
        ])

    # Strategy selector
    strategy_options = [{"label": r.display_name, "value": r.id} for r in records_with_comp]
    default_record = records_with_comp[0]

    # ── Build detail view for default record ────────────────
    cm = default_record.compression_metrics

    # Summary cards
    cards = [
        metric_card("Original Size", cm.original_size_bytes / 1e6 if cm else 0,
                     unit="MB", icon="📂", color="#3498db"),
        metric_card("Compressed Size", cm.compressed_size_bytes / 1e6 if cm else 0,
                     unit="MB", icon="📦", color="#e94560"),
        metric_card("Ratio", cm.compression_ratio if cm else 0,
                     unit="×", icon="🔄", color="#2ecc71"),
        metric_card("Savings", cm.savings_pct if cm else 0,
                     unit="%", icon="💰", color="#f39c12"),
    ]
    summary = metric_card_row(cards, cols_per_card=3)

    # Timing cards
    timing_cards = []
    if default_record.timing_metrics:
        t = default_record.timing_metrics
        if t.compress_time_s is not None:
            timing_cards.append(
                metric_card("Compress Time", t.compress_time_s, unit="s", icon="⚡", color="#1abc9c"))
        if t.decompress_time_s is not None:
            timing_cards.append(
                metric_card("Decompress Time", t.decompress_time_s, unit="s", icon="🔓", color="#9b59b6"))
    timing_row = metric_card_row(timing_cards, cols_per_card=3) if timing_cards else html.Div()

    # Waterfall chart
    waterfall = _build_waterfall(default_record)

    # Axis comparison
    axis_chart = _build_axis_comparison(default_record)

    # Per-frame PSNR curves for all axes
    pf_charts = []
    for axis, label in AXIS_LABELS.items():
        pf = default_record.get_per_frame(axis)
        if pf and pf.psnr:
            fig = build_per_frame_figure(
                [default_record], "psnr", axis=axis,
                title=f"PSNR per Frame — {label}",
            )
            pf_charts.append(html.Div(
                dcc.Graph(figure=fig, config={"displaylogo": False}),
                className="chart-container",
            ))

    # Pipeline stages
    pipeline_section = html.Div()
    if default_record.pipeline_stats:
        pipeline_section = html.Div([
            html.H5("Pipeline Stages", className="section-title mt-4"),
            _build_pipeline_table(default_record.pipeline_stats),
        ])

    # Model info
    model_section = _model_info_card(default_record)

    return html.Div([
        html.H4("Compression Detail", className="section-title"),
        html.P("Deep dive into a single compression strategy — file sizes, pipeline stages, and per-axis quality.",
               className="section-subtitle"),

        # Strategy picker
        dbc.Row([
            dbc.Col([
                html.Label("Select Strategy", htmlFor="detail-strategy-select", style={
                    "fontSize": "0.8rem", "color": "#a0b4d0", "textTransform": "uppercase"}),
                dcc.Dropdown(
                    id="detail-strategy-select",
                    options=strategy_options,
                    value=default_record.id,
                    clearable=False,
                    style={"backgroundColor": "#0b1e3d", "color": "#e0e0e0",
                           "border": "1px solid #0f3460"},
                ),
            ], md=6),
        ], className="filter-panel mb-4 p-3"),

        # Dynamic content container
        html.Div(id="detail-content-container", children=[
            summary,
            timing_row,

            dbc.Row([
                dbc.Col(
                    html.Div(dcc.Graph(id="detail-waterfall", figure=waterfall,
                                       config={"displaylogo": False}),
                             className="chart-container"),
                    md=6,
                ),
                dbc.Col(
                    html.Div(dcc.Graph(id="detail-axis-compare", figure=axis_chart,
                                       config={"displaylogo": False}),
                             className="chart-container"),
                    md=6,
                ),
            ]),

            # Per-frame curves
            html.H5("Per-Frame Quality", className="section-title mt-4") if pf_charts else None,
            *pf_charts,

            # Pipeline stages
            pipeline_section,

            # Model info
            model_section,
        ]),
    ])
