"""Compression Analysis page — compare compression techniques and their performance.

Shows comparative charts for compression ratios, savings, timing, and file sizes
across different compression techniques (balanced, aggressive, lossless, etc).
"""

from __future__ import annotations

from typing import List

import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import dcc, html

from ...domain.enums import ResultSource
from ...domain.models import MetricRecord
from ...domain.services import MetricsService
from ..components.metric_card import metric_card

COLORS = [
    "#e94560", "#533483", "#2ecc71", "#f39c12", "#3498db",
    "#e74c3c", "#1abc9c", "#9b59b6", "#d35400", "#0f3460",
]


def _filter_compression_reports(service: MetricsService) -> List[MetricRecord]:
    """Get all compression report records."""
    all_records = service.get_all_records()
    return [r for r in all_records if r.source == ResultSource.COMPRESSION_REPORT]


def _build_comparison_bar_chart(records: List[MetricRecord]) -> go.Figure:
    """Bar chart comparing compression ratios across techniques."""
    fig = go.Figure()

    techniques = []
    ratios = []
    savings = []
    for r in records:
        if r.compression_metrics:
            techniques.append(r.tags.get("technique", r.name))
            ratios.append(r.compression_metrics.compression_ratio)
            savings.append(r.compression_metrics.savings_pct)

    # Compression ratio bars
    fig.add_trace(go.Bar(
        name="Compression Ratio",
        x=techniques,
        y=ratios,
        marker_color=COLORS[0],
        text=[f"{v:.2f}×" for v in ratios],
        textposition="auto",
        yaxis="y",
    ))

    # Savings percentage as a line on secondary axis
    fig.add_trace(go.Scatter(
        name="Savings %",
        x=techniques,
        y=savings,
        mode="lines+markers",
        marker=dict(size=10, color=COLORS[2]),
        line=dict(width=2, color=COLORS[2]),
        text=[f"{v:.1f}%" for v in savings],
        textposition="top center",
        yaxis="y2",
    ))

    fig.update_layout(
        title="Compression Ratio & Savings by Technique",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,33,62,0.8)",
        font=dict(color="#e0e0e0"),
        xaxis=dict(title="Technique"),
        yaxis=dict(title="Compression Ratio (×)", title_font=dict(color=COLORS[0])),
        yaxis2=dict(
            title="Savings (%)",
            title_font=dict(color=COLORS[2]),
            overlaying="y",
            side="right",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=80, t=60, b=80),
        height=450,
    )

    return fig


def _build_file_size_chart(records: List[MetricRecord]) -> go.Figure:
    """Grouped bar chart comparing original vs compressed sizes."""
    fig = go.Figure()

    techniques = []
    original_sizes = []
    compressed_sizes = []

    for r in records:
        if r.compression_metrics:
            techniques.append(r.tags.get("technique", r.name))
            original_sizes.append(r.compression_metrics.original_size_bytes / 1e6)  # MB
            compressed_sizes.append(r.compression_metrics.compressed_size_bytes / 1e6)  # MB

    fig.add_trace(go.Bar(
        name="Original Size",
        x=techniques,
        y=original_sizes,
        marker_color=COLORS[3],
        text=[f"{v:.1f} MB" for v in original_sizes],
        textposition="auto",
    ))

    fig.add_trace(go.Bar(
        name="Compressed Size",
        x=techniques,
        y=compressed_sizes,
        marker_color=COLORS[0],
        text=[f"{v:.1f} MB" for v in compressed_sizes],
        textposition="auto",
    ))

    fig.update_layout(
        barmode="group",
        title="File Size Comparison",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,33,62,0.8)",
        font=dict(color="#e0e0e0"),
        xaxis=dict(title="Technique"),
        yaxis=dict(title="Size (MB)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=80),
        height=450,
    )

    return fig


def _build_timing_chart(records: List[MetricRecord]) -> go.Figure:
    """Bar chart showing compression timing across techniques."""
    fig = go.Figure()

    techniques = []
    times = []

    for r in records:
        if r.timing_metrics and r.timing_metrics.compress_time_s is not None:
            techniques.append(r.tags.get("technique", r.name))
            times.append(r.timing_metrics.compress_time_s)

    if not techniques:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text="No timing data available", showarrow=False,
                            font=dict(size=14, color="#6c7a8e"))],
        )
        return fig

    fig.add_trace(go.Bar(
        x=techniques,
        y=times,
        marker_color=COLORS[4],
        text=[f"{v:.2f}s" for v in times],
        textposition="auto",
    ))

    fig.update_layout(
        title="Compression Time by Technique",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,33,62,0.8)",
        font=dict(color="#e0e0e0"),
        xaxis=dict(title="Technique"),
        yaxis=dict(title="Time (seconds)"),
        margin=dict(l=50, r=20, t=60, b=80),
        height=400,
    )

    return fig


def _build_summary_cards(records: List[MetricRecord]) -> html.Div:
    """Build summary metric cards."""
    if not records:
        return html.Div(html.P("No compression data available.", className="text-muted"))

    # Calculate aggregate stats
    total_original = sum(r.compression_metrics.original_size_bytes for r in records if r.compression_metrics)
    total_compressed = sum(r.compression_metrics.compressed_size_bytes for r in records if r.compression_metrics)
    avg_ratio = sum(r.compression_metrics.compression_ratio for r in records if r.compression_metrics) / len(records)
    avg_savings = sum(r.compression_metrics.savings_pct for r in records if r.compression_metrics) / len(records)

    cards = [
        metric_card("Total Techniques", str(len(records)), "🗜️"),
        metric_card("Avg Compression Ratio", f"{avg_ratio:.2f}×", "📊"),
        metric_card("Avg Savings", f"{avg_savings:.1f}%", "💾"),
        metric_card(
            "Total Space Saved",
            f"{(total_original - total_compressed) / 1e6:.1f} MB",
            "✨"
        ),
    ]

    return html.Div(
        dbc.Row([dbc.Col(card, width=12, md=6, lg=3) for card in cards]),
        className="mb-4"
    )


def _build_techniques_table(records: List[MetricRecord]) -> html.Div:
    """Detailed table of all compression techniques."""
    if not records:
        return html.Div()

    rows = []
    for r in sorted(records, key=lambda x: x.compression_metrics.compression_ratio if x.compression_metrics else 0, reverse=True):
        cm = r.compression_metrics
        if not cm:
            continue

        technique = r.tags.get("technique", r.name)
        scene = r.scene or "—"
        config = r.tags.get("config_file", "—")
        if config != "—":
            config = config.split("/")[-1] if "/" in config else config.split("\\")[-1]

        rows.append(html.Tr([
            html.Td(technique),
            html.Td(scene),
            html.Td(config, style={"fontSize": "0.8rem"}),
            html.Td(f"{cm.original_size_bytes / 1e6:.2f}", style={"textAlign": "right"}),
            html.Td(f"{cm.compressed_size_bytes / 1e6:.2f}", style={"textAlign": "right"}),
            html.Td(f"{cm.compression_ratio:.2f}×", style={"textAlign": "right"}),
            html.Td(f"{cm.savings_pct:.1f}%", style={"textAlign": "right"}),
            html.Td(f"{cm.num_chunks:,}", style={"textAlign": "right"}),
        ]))

    return dbc.Table([
        html.Thead(html.Tr([
            html.Th("Technique"),
            html.Th("Scene"),
            html.Th("Config"),
            html.Th("Original (MB)", style={"textAlign": "right"}),
            html.Th("Compressed (MB)", style={"textAlign": "right"}),
            html.Th("Ratio", style={"textAlign": "right"}),
            html.Th("Savings", style={"textAlign": "right"}),
            html.Th("Chunks", style={"textAlign": "right"}),
        ])),
        html.Tbody(rows),
    ], bordered=True, striped=True, hover=True, color="dark",
       responsive=True, style={"fontSize": "0.9rem"})


def build_compression_analysis(service: MetricsService) -> html.Div:
    """Build the complete compression analysis page."""
    records = _filter_compression_reports(service)

    if not records:
        return html.Div([
            html.H2("Compression Analysis", className="mb-4"),
            dbc.Alert(
                [
                    html.H4("No Compression Data", className="alert-heading"),
                    html.P(
                        "No compression_report.json files found. "
                        "Run compress.py to generate compression reports."
                    ),
                ],
                color="info",
            ),
        ], className="page-container")

    return html.Div([
        html.H2("Compression Analysis", className="mb-4"),
        html.P(
            f"Comparing {len(records)} compression technique(s) from compressed_output/",
            className="text-muted mb-4"
        ),

        _build_summary_cards(records),

        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    figure=_build_comparison_bar_chart(records),
                    config={"displayModeBar": False},
                )
            ], width=12, lg=6),
            dbc.Col([
                dcc.Graph(
                    figure=_build_file_size_chart(records),
                    config={"displayModeBar": False},
                )
            ], width=12, lg=6),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    figure=_build_timing_chart(records),
                    config={"displayModeBar": False},
                )
            ], width=12),
        ], className="mb-4"),

        html.H4("Detailed Comparison", className="mt-4 mb-3"),
        _build_techniques_table(records),

    ], className="page-container")
