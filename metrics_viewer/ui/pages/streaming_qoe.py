"""Streaming QoE page — quality of experience dashboard."""

from __future__ import annotations

import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import dcc, html

from ...domain.enums import ResultSource
from ...domain.models import MetricRecord
from ...domain.services import MetricsService
from ..components.comparison_table import comparison_table
from ..components.metric_card import metric_card, metric_card_row

COLORS = [
    "#e94560", "#533483", "#2ecc71", "#f39c12", "#3498db",
    "#e74c3c", "#1abc9c", "#9b59b6", "#d35400", "#0f3460",
]


def build_streaming_qoe(service: MetricsService) -> html.Div:
    """Construct the Streaming QoE page."""
    # Find records that have streaming metrics
    records = [
        r for r in service.get_all_records()
        if r.streaming_metrics is not None and r.streaming_metrics.qoe_score > 0
    ]

    if not records:
        return html.Div([
            html.H4("Streaming QoE", className="section-title"),
            html.P("No streaming QoE data found.", className="text-muted"),
        ])

    # ── Gauge charts for each strategy ──────────────────────
    gauge_rows = []

    for i, r in enumerate(records):
        sm = r.streaming_metrics
        if sm is None:
            continue

        # QoE Gauge
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sm.qoe_score,
            title={"text": f"{r.display_name}", "font": {"size": 14, "color": "#e0e0e0"}},
            number={"suffix": " / 5.0", "font": {"size": 22, "color": "#e0e0e0"}},
            gauge=dict(
                axis=dict(range=[1, 5], tickcolor="#e0e0e0"),
                bar=dict(color="#e94560"),
                bgcolor="rgba(22,33,62,0.8)",
                borderwidth=1,
                bordercolor="#0f3460",
                steps=[
                    dict(range=[1, 2], color="rgba(231,76,60,0.3)"),
                    dict(range=[2, 3], color="rgba(243,156,18,0.3)"),
                    dict(range=[3, 4], color="rgba(52,152,219,0.3)"),
                    dict(range=[4, 5], color="rgba(46,204,113,0.3)"),
                ],
                threshold=dict(
                    line=dict(color="#f39c12", width=3),
                    thickness=0.75,
                    value=sm.qoe_score,
                ),
            ),
        ))

        gauge.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"),
            height=250,
            margin=dict(l=30, r=30, t=50, b=20),
        )

        gauge_rows.append(dbc.Col(
            html.Div(
                dcc.Graph(id=f"qoe-gauge-{i}", figure=gauge, config={"displaylogo": False}),
                className="chart-container",
            ),
            md=4,
        ))

    # ── Bar chart: buffering metrics ────────────────────────
    names = [r.display_name for r in records]

    fig_buffer = go.Figure()
    fig_buffer.add_trace(go.Bar(
        x=names,
        y=[r.streaming_metrics.startup_delay_s for r in records],
        name="Startup Delay (s)",
        marker_color="#f39c12",
        text=[f"{r.streaming_metrics.startup_delay_s:.2f}s" for r in records],
        textposition="auto",
    ))
    fig_buffer.add_trace(go.Bar(
        x=names,
        y=[r.streaming_metrics.total_stall_duration_s for r in records],
        name="Total Stall (s)",
        marker_color="#e74c3c",
        text=[f"{r.streaming_metrics.total_stall_duration_s:.1f}s" for r in records],
        textposition="auto",
    ))
    fig_buffer.update_layout(
        barmode="group",
        title="Buffer & Stall Performance",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,33,62,0.8)",
        font=dict(color="#e0e0e0"),
        yaxis_title="Time (s)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=20, t=60, b=40),
        height=350,
    )

    # ── Bar chart: rebuffer events ──────────────────────────
    fig_rebuf = go.Figure()
    fig_rebuf.add_trace(go.Bar(
        x=names,
        y=[r.streaming_metrics.rebuffer_events for r in records],
        name="Rebuffer Events",
        marker_color="#e94560",
        text=[str(r.streaming_metrics.rebuffer_events) for r in records],
        textposition="auto",
    ))
    fig_rebuf.update_layout(
        title="Rebuffer Events Count",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,33,62,0.8)",
        font=dict(color="#e0e0e0"),
        yaxis_title="Count",
        margin=dict(l=50, r=20, t=60, b=40),
        height=350,
    )

    # ── Latency + throughput bar chart ──────────────────────
    fig_latency = go.Figure()
    fig_latency.add_trace(go.Bar(
        x=names,
        y=[r.streaming_metrics.e2e_latency_s for r in records],
        name="E2E Latency (s)",
        marker_color="#3498db",
    ))
    fig_latency.add_trace(go.Bar(
        x=names,
        y=[r.streaming_metrics.effective_throughput_mbps for r in records],
        name="Throughput (MB/s)",
        marker_color="#2ecc71",
        yaxis="y2",
    ))
    fig_latency.update_layout(
        barmode="group",
        title="Latency & Throughput",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,33,62,0.8)",
        font=dict(color="#e0e0e0"),
        yaxis=dict(title="Latency (s)"),
        yaxis2=dict(title="Throughput (MB/s)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=60, t=60, b=40),
        height=350,
    )

    # ── Summary cards ───────────────────────────────────────
    best_qoe_r = max(records, key=lambda r: r.streaming_metrics.qoe_score)
    sm_best = best_qoe_r.streaming_metrics

    summary = metric_card_row([
        metric_card("Best QoE", sm_best.qoe_score, unit="/ 5.0", icon="📡", color="#e94560"),
        metric_card("Strategies", len(records), icon="🔧", color="#533483"),
        metric_card("Min Startup", min(r.streaming_metrics.startup_delay_s for r in records), unit="s", icon="⏱️", color="#2ecc71"),
        metric_card("Target FPS", sm_best.target_fps, unit="fps", icon="🎞️", color="#3498db"),
    ])

    return html.Div([
        html.H4("Streaming QoE", className="section-title"),
        html.P("Quality of Experience metrics for streaming 4DGS content. Based on ITU-T P.1203 inspired scoring.",
               className="section-subtitle"),

        summary,

        html.H5("QoE Score Gauges", className="section-title mt-3"),
        dbc.Row(gauge_rows, className="g-3"),

        dbc.Row([
            dbc.Col(
                html.Div(dcc.Graph(id="qoe-buffer", figure=fig_buffer, config={"displaylogo": False}), className="chart-container"),
                md=6,
            ),
            dbc.Col(
                html.Div(dcc.Graph(id="qoe-rebuf", figure=fig_rebuf, config={"displaylogo": False}), className="chart-container"),
                md=6,
            ),
        ]),

        html.Div(
            dcc.Graph(id="qoe-latency", figure=fig_latency, config={"displaylogo": False}),
            className="chart-container",
        ),

        html.H5("Detailed Data", className="section-title mt-4"),
        comparison_table(records, table_id="qoe-table"),
    ])
