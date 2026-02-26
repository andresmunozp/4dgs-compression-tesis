"""Benchmark Compare page — compare compression strategies side by side."""

from __future__ import annotations

from typing import List, Optional

import dash_bootstrap_components as dbc
from dash import dcc, html

from ...domain.enums import ComparisonAxis, MetricType, ResultSource
from ...domain.models import MetricRecord
from ...domain.services import MetricsService
from ..components.bar_chart import bar_chart, stacked_bar_timing
from ..components.comparison_table import comparison_table
from ..components.filters import axis_selector
from ..components.radar_chart import radar_chart
from ..components.scatter_plot import scatter_plot


QUALITY_METRICS = [
    MetricType.PSNR,
    MetricType.SSIM,
    MetricType.LPIPS_VGG,
    MetricType.VMAF,
]

RADAR_METRICS = [
    MetricType.PSNR,
    MetricType.SSIM,
    MetricType.VMAF,
    MetricType.COMPRESSION_RATIO,
    MetricType.SAVINGS_PCT,
]


def build_benchmark_compare(service: MetricsService) -> html.Div:
    """Construct the Benchmark Compare page."""
    records = service.get_records_by_source(ResultSource.BENCHMARK_JSON)

    if not records:
        return html.Div([
            html.H4("Benchmark Compare", className="section-title"),
            html.P("No benchmark JSON results found. Run benchmark_compression.py first.",
                   className="text-muted"),
        ])

    # Strategy selector (for multi-select)
    strategy_options = [{"label": r.display_name, "value": r.id} for r in records]

    # ── Axis selector ───────────────────────────────────────
    axis_sel = axis_selector(selector_id="benchmark-axis")

    # ── Charts ──────────────────────────────────────────────

    # 1. Bar chart: quality metrics
    quality_bar = bar_chart(
        records, QUALITY_METRICS,
        axis=ComparisonAxis.END_TO_END,
        chart_id="benchmark-quality-bar",
        title="Quality Metrics by Strategy (End-to-End)",
    )

    # 2. Compression fidelity bar
    fidelity_bar = bar_chart(
        records, QUALITY_METRICS,
        axis=ComparisonAxis.COMPRESSION_FIDELITY,
        chart_id="benchmark-fidelity-bar",
        title="Quality Metrics by Strategy (Compression Fidelity)",
    )

    # 3. Scatter: ratio vs PSNR
    scatter = scatter_plot(
        records,
        x_metric=MetricType.COMPRESSION_RATIO,
        y_metric=MetricType.PSNR,
        axis=ComparisonAxis.END_TO_END,
        chart_id="benchmark-scatter",
        title="Compression Ratio vs PSNR (Pareto Frontier)",
    )

    # 4. Radar
    radar = radar_chart(
        records, RADAR_METRICS,
        axis=ComparisonAxis.END_TO_END,
        chart_id="benchmark-radar",
        title="Multi-Metric Comparison (Normalized)",
    )

    # 5. Timing
    timing = stacked_bar_timing(
        records,
        chart_id="benchmark-timing",
        title="Compression / Decompression Time",
    )

    # 6. Comparison table
    table = comparison_table(
        records,
        axis=ComparisonAxis.END_TO_END,
        table_id="benchmark-table",
    )

    return html.Div([
        html.H4("Benchmark Compare", className="section-title"),
        html.P("Compare compression strategies across quality, ratio, and timing metrics.",
               className="section-subtitle"),

        # Axis selector
        dbc.Row([
            dbc.Col(axis_sel, md=6),
            dbc.Col(
                html.Div([
                    html.Label("Select Strategies", htmlFor="benchmark-strategy-select",
                               style={"fontSize": "0.8rem", "color": "#a0b4d0", "textTransform": "uppercase"}),
                    dcc.Dropdown(
                        id="benchmark-strategy-select",
                        options=strategy_options,
                        value=[r.id for r in records],
                        multi=True,
                        placeholder="Select strategies\u2026",
                        style={"backgroundColor": "#0b1e3d", "color": "#e0e0e0", "border": "1px solid #0f3460"},
                    ),
                ]), md=6,
            ),
        ], className="filter-panel mb-4 p-3"),

        # Quality charts side by side
        dbc.Row([
            dbc.Col(quality_bar, md=6),
            dbc.Col(fidelity_bar, md=6),
        ]),

        # Scatter + Radar
        dbc.Row([
            dbc.Col(scatter, md=6),
            dbc.Col(radar, md=6),
        ]),

        # Timing
        dbc.Row([
            dbc.Col(timing, md=12),
        ]),

        # Detailed table
        html.H5("Detailed Comparison", className="section-title mt-4"),
        table,
    ])
