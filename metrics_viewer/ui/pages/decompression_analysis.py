"""Decompression Analysis page — analyze decompression performance across techniques.

Shows timing breakdowns (assembly, decode, export), throughput metrics,
and performance comparisons across different decompression strategies.
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


def _filter_decompression_reports(service: MetricsService) -> List[MetricRecord]:
    """Get all decompression report records."""
    all_records = service.get_all_records()
    return [r for r in all_records if r.source == ResultSource.DECOMPRESSION_REPORT]


def _build_timing_breakdown_chart(records: List[MetricRecord]) -> go.Figure:
    """Stacked bar chart showing timing breakdown by phase."""
    fig = go.Figure()

    techniques = []
    assemble_times = []
    decode_times = []
    export_times = []

    for r in records:
        technique = r.tags.get("technique", r.name)
        techniques.append(technique)
        
        assemble_times.append(float(r.tags.get("assemble_time_s", 0)))
        decode_times.append(float(r.tags.get("decode_time_s", 0)))
        export_times.append(float(r.tags.get("export_time_s", 0)))

    fig.add_trace(go.Bar(
        name="Assembly",
        x=techniques,
        y=assemble_times,
        marker_color=COLORS[4],
        text=[f"{v:.3f}s" if v > 0 else "" for v in assemble_times],
        textposition="inside",
    ))

    fig.add_trace(go.Bar(
        name="Decode",
        x=techniques,
        y=decode_times,
        marker_color=COLORS[0],
        text=[f"{v:.3f}s" if v > 0 else "" for v in decode_times],
        textposition="inside",
    ))

    fig.add_trace(go.Bar(
        name="Export",
        x=techniques,
        y=export_times,
        marker_color=COLORS[2],
        text=[f"{v:.3f}s" if v > 0 else "" for v in export_times],
        textposition="inside",
    ))

    fig.update_layout(
        barmode="stack",
        title="Decompression Timing Breakdown",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,33,62,0.8)",
        font=dict(color="#e0e0e0"),
        xaxis=dict(title="Technique"),
        yaxis=dict(title="Time (seconds)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=80),
        height=450,
    )

    return fig


def _build_total_time_chart(records: List[MetricRecord]) -> go.Figure:
    """Bar chart comparing total decompression times."""
    fig = go.Figure()

    techniques = []
    total_times = []

    for r in records:
        if r.timing_metrics and r.timing_metrics.decompress_time_s is not None:
            techniques.append(r.tags.get("technique", r.name))
            total_times.append(r.timing_metrics.decompress_time_s)

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
        y=total_times,
        marker_color=COLORS[0],
        text=[f"{v:.3f}s" for v in total_times],
        textposition="auto",
    ))

    fig.update_layout(
        title="Total Decompression Time",
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


def _build_throughput_chart(records: List[MetricRecord]) -> go.Figure:
    """Chart showing frames per second during export."""
    fig = go.Figure()

    techniques = []
    fps_values = []

    for r in records:
        num_frames = int(r.tags.get("num_frames", 0)) if r.tags.get("num_frames") else 0
        export_time = float(r.tags.get("export_time_s", 0)) if r.tags.get("export_time_s") else 0
        
        if num_frames > 0 and export_time > 0:
            fps = num_frames / export_time
            techniques.append(r.tags.get("technique", r.name))
            fps_values.append(fps)

    if not techniques:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text="No throughput data available", showarrow=False,
                            font=dict(size=14, color="#6c7a8e"))],
        )
        return fig

    fig.add_trace(go.Bar(
        x=techniques,
        y=fps_values,
        marker_color=COLORS[2],
        text=[f"{v:.1f} fps" for v in fps_values],
        textposition="auto",
    ))

    fig.update_layout(
        title="Export Throughput (Frames Per Second)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,33,62,0.8)",
        font=dict(color="#e0e0e0"),
        xaxis=dict(title="Technique"),
        yaxis=dict(title="FPS"),
        margin=dict(l=50, r=20, t=60, b=80),
        height=400,
    )

    return fig


def _build_summary_cards(records: List[MetricRecord]) -> html.Div:
    """Build summary metric cards."""
    if not records:
        return html.Div(html.P("No decompression data available.", className="text-muted"))

    # Calculate aggregate stats
    total_techniques = len(records)
    
    avg_decode_time = 0
    decode_count = 0
    for r in records:
        decode_time = float(r.tags.get("decode_time_s", 0)) if r.tags.get("decode_time_s") else 0
        if decode_time > 0:
            avg_decode_time += decode_time
            decode_count += 1
    
    if decode_count > 0:
        avg_decode_time /= decode_count

    avg_total_time = 0
    total_count = 0
    for r in records:
        if r.timing_metrics and r.timing_metrics.decompress_time_s is not None:
            avg_total_time += r.timing_metrics.decompress_time_s
            total_count += 1
    
    if total_count > 0:
        avg_total_time /= total_count

    total_gaussians = sum(
        r.model_info.num_gaussians_compressed
        for r in records
        if r.model_info and r.model_info.num_gaussians_compressed
    )

    cards = [
        metric_card("Total Techniques", str(total_techniques), "🔄"),
        metric_card("Avg Decode Time", f"{avg_decode_time:.3f}s", "⚡"),
        metric_card("Avg Total Time", f"{avg_total_time:.3f}s", "⏱️"),
        metric_card("Total Gaussians", f"{total_gaussians:,}", "✨"),
    ]

    return html.Div(
        dbc.Row([dbc.Col(card, width=12, md=6, lg=3) for card in cards]),
        className="mb-4"
    )


def _build_techniques_table(records: List[MetricRecord]) -> html.Div:
    """Detailed table of all decompression techniques."""
    if not records:
        return html.Div()

    rows = []
    for r in sorted(records, key=lambda x: x.timing_metrics.decompress_time_s if x.timing_metrics and x.timing_metrics.decompress_time_s else 0):
        technique = r.tags.get("technique", r.name)
        scene = r.scene or "—"
        
        num_frames = r.tags.get("num_frames", "—")
        assemble = float(r.tags.get("assemble_time_s", 0))
        decode = float(r.tags.get("decode_time_s", 0))
        export = float(r.tags.get("export_time_s", 0))
        total = r.timing_metrics.decompress_time_s if r.timing_metrics else 0

        num_gaussians = "—"
        if r.model_info and r.model_info.num_gaussians_compressed:
            num_gaussians = f"{r.model_info.num_gaussians_compressed:,}"

        rows.append(html.Tr([
            html.Td(technique),
            html.Td(scene),
            html.Td(str(num_frames), style={"textAlign": "right"}),
            html.Td(num_gaussians, style={"textAlign": "right"}),
            html.Td(f"{assemble:.3f}s", style={"textAlign": "right"}),
            html.Td(f"{decode:.3f}s", style={"textAlign": "right"}),
            html.Td(f"{export:.3f}s", style={"textAlign": "right"}),
            html.Td(f"{total:.3f}s", style={"textAlign": "right", "fontWeight": "bold"}),
        ]))

    return dbc.Table([
        html.Thead(html.Tr([
            html.Th("Technique"),
            html.Th("Scene"),
            html.Th("Frames", style={"textAlign": "right"}),
            html.Th("Gaussians", style={"textAlign": "right"}),
            html.Th("Assembly", style={"textAlign": "right"}),
            html.Th("Decode", style={"textAlign": "right"}),
            html.Th("Export", style={"textAlign": "right"}),
            html.Th("Total", style={"textAlign": "right"}),
        ])),
        html.Tbody(rows),
    ], bordered=True, striped=True, hover=True, color="dark",
       responsive=True, style={"fontSize": "0.9rem"})


def build_decompression_analysis(service: MetricsService) -> html.Div:
    """Build the complete decompression analysis page."""
    records = _filter_decompression_reports(service)

    if not records:
        return html.Div([
            html.H2("Decompression Analysis", className="mb-4"),
            dbc.Alert(
                [
                    html.H4("No Decompression Data", className="alert-heading"),
                    html.P(
                        "No decompression_report.json files found. "
                        "Run decompress.py to generate decompression reports."
                    ),
                ],
                color="info",
            ),
        ], className="page-container")

    return html.Div([
        html.H2("Decompression Analysis", className="mb-4"),
        html.P(
            f"Analyzing {len(records)} decompression technique(s) from decompressed_output/",
            className="text-muted mb-4"
        ),

        _build_summary_cards(records),

        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    figure=_build_timing_breakdown_chart(records),
                    config={"displayModeBar": False},
                )
            ], width=12, lg=6),
            dbc.Col([
                dcc.Graph(
                    figure=_build_total_time_chart(records),
                    config={"displayModeBar": False},
                )
            ], width=12, lg=6),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    figure=_build_throughput_chart(records),
                    config={"displayModeBar": False},
                )
            ], width=12),
        ], className="mb-4"),

        html.H4("Detailed Breakdown", className="mt-4 mb-3"),
        _build_techniques_table(records),

    ], className="page-container")
