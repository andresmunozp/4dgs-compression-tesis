"""Dynamic Dash callbacks — Phase 3 + Phase 5.

Registers all interactive callbacks that make filters, axis selectors,
strategy pickers, export buttons, and compression detail page work.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import dash
from dash import html, dcc, no_update
import dash_bootstrap_components as dbc

from ..domain.enums import ComparisonAxis, MetricType, ResultCategory, ResultSource
from ..domain.models import MetricRecord
from ..domain.ports import ExportConfig
from ..domain.services import MetricsService, _extract_metric
from ..adapters.exporters.csv_exporter import CsvExporter
from ..adapters.exporters.png_exporter import PngExporter
from .components.bar_chart import build_bar_figure, build_timing_figure
from .components.comparison_table import build_table_data, records_to_dataframe
from .components.metric_card import metric_card, metric_card_row
from .components.per_frame_chart import build_per_frame_figure, build_heatmap_figure
from .components.radar_chart import build_radar_figure
from .components.scatter_plot import build_scatter_figure

logger = logging.getLogger(__name__)


# ── Axis string → enum mapping ──────────────────────────────────

_AXIS_MAP = {
    "compression_fidelity": ComparisonAxis.COMPRESSION_FIDELITY,
    "end_to_end": ComparisonAxis.END_TO_END,
    "training_baseline": ComparisonAxis.TRAINING_BASELINE,
}


# ── Quality / Radar metric lists (shared with benchmark page) ───

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


# ── Helpers ─────────────────────────────────────────────────────


def _filter_records(
    service: MetricsService,
    sources: list | None,
    categories: list | None,
    scenes: list | None,
    names: list | None,
) -> List[MetricRecord]:
    """Apply multi-select filter values from dropdowns to the full record set."""
    records = service.get_all_records()

    if sources:
        src_set = set(sources)
        records = [r for r in records if r.source.value in src_set]

    if categories:
        cat_set = set(categories)
        records = [r for r in records if r.category.value in cat_set]

    if scenes:
        scene_set = set(scenes)
        records = [r for r in records if r.scene in scene_set]

    if names:
        name_set = set(names)
        records = [r for r in records if r.name in name_set]

    return records


def _best(records: List[MetricRecord], metric: MetricType, axis=None, higher=True):
    """Get the best value + record name for a metric."""
    best_val = None
    best_name = "—"
    for r in records:
        v = _extract_metric(r, metric, axis)
        if v is None:
            continue
        if best_val is None or (higher and v > best_val) or (not higher and v < best_val):
            best_val = v
            best_name = r.display_name
    return best_val, best_name


# ═══════════════════════════════════════════════════════════════
#  REGISTRATION
# ═══════════════════════════════════════════════════════════════


def register_callbacks(app: dash.Dash, service: MetricsService) -> None:
    """Register all dynamic callbacks onto *app*."""

    _register_overview_callbacks(app, service)
    _register_benchmark_callbacks(app, service)
    _register_perframe_callbacks(app, service)
    _register_export_callbacks(app, service)
    _register_compression_detail_callbacks(app, service)
    _register_training_callbacks(app, service)

    logger.info("Phase 3 + Phase 5 + Phase 4 (training) callbacks registered")


# ═══════════════════════════════════════════════════════════════
#  1. OVERVIEW CALLBACKS
# ═══════════════════════════════════════════════════════════════


def _register_overview_callbacks(app: dash.Dash, service: MetricsService):

    @app.callback(
        dash.Output("overview-summary-container", "children"),
        dash.Output("overview-table-container", "children"),
        dash.Output("overview-performers-container", "children"),
        dash.Output("sidebar-record-count", "children"),
        dash.Input("overview-filter-source", "value"),
        dash.Input("overview-filter-category", "value"),
        dash.Input("overview-filter-scene", "value"),
        dash.Input("overview-filter-name", "value"),
        prevent_initial_call=True,
    )
    def update_overview(sources, categories, scenes, names):
        filtered = _filter_records(service, sources, categories, scenes, names)

        # ── Summary cards ───────────────────────────────
        benchmark = [r for r in filtered if r.source == ResultSource.BENCHMARK_JSON]
        training = [r for r in filtered if r.source == ResultSource.TRAINING_JSON]

        num_strategies = len(set(r.name for r in benchmark))
        f = service.get_available_filters()
        num_scenes = len(f.scenes)
        num_sources = len(set(r.source for r in filtered))

        summary = metric_card_row([
            metric_card("Total Records", len(filtered), icon="📊", color="#3498db"),
            metric_card("Strategies", num_strategies, icon="🔧", color="#533483"),
            metric_card("Scenes", num_scenes or "1", icon="🎬", color="#2ecc71"),
            metric_card("Data Sources", num_sources, icon="📁", color="#f39c12"),
        ])

        best_psnr, best_psnr_name = _best(benchmark, MetricType.PSNR, ComparisonAxis.END_TO_END)
        best_ratio, best_ratio_name = _best(benchmark, MetricType.COMPRESSION_RATIO)
        best_ssim, best_ssim_name = _best(benchmark, MetricType.SSIM, ComparisonAxis.COMPRESSION_FIDELITY)
        best_train_psnr, best_train_name = _best(training, MetricType.PSNR)

        best_cards = metric_card_row([
            metric_card("Best E2E PSNR", best_psnr or 0, unit="dB", icon="🏆", color="#e94560"),
            metric_card("Best Ratio", best_ratio or 0, unit="×", icon="📦", color="#533483"),
            metric_card("Best Fidelity SSIM", best_ssim or 0, icon="✨", color="#2ecc71"),
            metric_card("Best Training PSNR", best_train_psnr or 0, unit="dB", icon="🎯", color="#3498db"),
        ], cols_per_card=3)

        cards_content = [summary, best_cards]

        # ── Table ───────────────────────────────────────
        from .components.comparison_table import comparison_table
        table = comparison_table(filtered, axis=ComparisonAxis.END_TO_END, table_id="overview-table")
        table_content = [table]

        # ── Top performers ──────────────────────────────
        notes = []
        if best_psnr_name != "—":
            notes.append(html.Li(f"Best E2E PSNR: {best_psnr_name} ({best_psnr:.2f} dB)"))
        if best_ratio_name != "—":
            notes.append(html.Li(f"Best Compression Ratio: {best_ratio_name} ({best_ratio:.2f}×)"))
        if best_train_name != "—":
            notes.append(html.Li(f"Best Training PSNR: {best_train_name} ({best_train_psnr:.2f} dB)"))

        performers = [
            html.Div([
                html.H6("🏆 Top Performers", className="mt-4 mb-2", style={"color": "#f39c12"}),
                html.Ul(notes, style={"color": "#a0b4d0", "fontSize": "0.9rem"}),
            ]) if notes else None
        ]

        sidebar_text = f"{len(filtered)} records shown"

        return cards_content, table_content, performers, sidebar_text


# ═══════════════════════════════════════════════════════════════
#  2. BENCHMARK COMPARE CALLBACKS
# ═══════════════════════════════════════════════════════════════


def _register_benchmark_callbacks(app: dash.Dash, service: MetricsService):

    @app.callback(
        dash.Output("benchmark-quality-bar", "figure"),
        dash.Output("benchmark-fidelity-bar", "figure"),
        dash.Output("benchmark-scatter", "figure"),
        dash.Output("benchmark-radar", "figure"),
        dash.Output("benchmark-timing", "figure"),
        dash.Output("benchmark-table", "data"),
        dash.Output("benchmark-table", "columns"),
        dash.Output("benchmark-table", "style_data_conditional"),
        dash.Input("benchmark-axis", "value"),
        dash.Input("benchmark-strategy-select", "value"),
        prevent_initial_call=True,
    )
    def update_benchmark(axis_value, strategy_ids):
        axis = _AXIS_MAP.get(axis_value, ComparisonAxis.END_TO_END)
        all_bench = service.get_records_by_source(ResultSource.BENCHMARK_JSON)

        # Filter to selected strategies
        if strategy_ids:
            id_set = set(strategy_ids)
            records = [r for r in all_bench if r.id in id_set]
        else:
            records = all_bench

        if not records:
            empty = _empty_figure("No strategies selected")
            return empty, empty, empty, empty, empty, [], [], []

        # Build figures with selected axis
        quality_fig = build_bar_figure(
            records, QUALITY_METRICS,
            axis=axis,
            title=f"Quality Metrics ({axis.value.replace('_', ' ').title()})",
        )

        fidelity_axis = (ComparisonAxis.COMPRESSION_FIDELITY
                         if axis != ComparisonAxis.COMPRESSION_FIDELITY
                         else ComparisonAxis.END_TO_END)
        fidelity_fig = build_bar_figure(
            records, QUALITY_METRICS,
            axis=fidelity_axis,
            title=f"Quality Metrics ({fidelity_axis.value.replace('_', ' ').title()})",
        )

        scatter_fig = build_scatter_figure(
            records,
            x_metric=MetricType.COMPRESSION_RATIO,
            y_metric=MetricType.PSNR,
            axis=axis,
            title="Compression Ratio vs PSNR (Pareto Frontier)",
        )

        radar_fig = build_radar_figure(
            records, RADAR_METRICS,
            axis=axis,
            title="Multi-Metric Comparison (Normalized)",
        )

        timing_fig = build_timing_figure(
            records,
            title="Compression / Decompression Time",
        )

        # Table data
        data, columns, styles = build_table_data(records, axis)

        return quality_fig, fidelity_fig, scatter_fig, radar_fig, timing_fig, data, columns, styles


# ═══════════════════════════════════════════════════════════════
#  3. PER-FRAME ANALYSIS CALLBACKS
# ═══════════════════════════════════════════════════════════════


def _register_perframe_callbacks(app: dash.Dash, service: MetricsService):

    @app.callback(
        dash.Output("pf-charts-container", "children"),
        dash.Input("pf-metric-select", "value"),
        dash.Input("pf-strategy-select", "value"),
        dash.Input("pf-axis", "value"),
        prevent_initial_call=True,
    )
    def update_perframe(metric_name, strategy_ids, axis_value):
        axis = _AXIS_MAP.get(axis_value)

        # Get all records with per-frame data
        all_pf = []
        for r in service.get_all_records():
            has_pf = r.per_frame_metrics is not None
            if not has_pf:
                for apf in r.per_frame_axes.values():
                    if apf is not None:
                        has_pf = True
                        break
            if has_pf:
                all_pf.append(r)

        # Filter to selected strategies
        if strategy_ids:
            id_set = set(strategy_ids)
            records = [r for r in all_pf if r.id in id_set]
        else:
            records = all_pf

        if not records or not metric_name:
            return [html.P("No data matching current selection.", className="text-muted p-3")]

        benchmark_records = [r for r in records if r.source == ResultSource.BENCHMARK_JSON]
        vmaf_records = [r for r in records if r.source == ResultSource.VMAF_JSON]

        charts = []

        # Benchmark per-frame charts
        if benchmark_records:
            charts.append(html.H5(
                f"Benchmark — {axis.value.replace('_', ' ').title() if axis else 'All Axes'}",
                className="section-title mt-3",
            ))

            fig = build_per_frame_figure(
                benchmark_records, metric_name,
                axis=axis,
                title=f"{metric_name.upper()} per Frame",
            )
            charts.append(html.Div(
                dcc.Graph(figure=fig, config={"displaylogo": False}),
                className="chart-container",
            ))

            # Also show the complementary axis if we have data
            other_axis = (ComparisonAxis.END_TO_END
                          if axis == ComparisonAxis.COMPRESSION_FIDELITY
                          else ComparisonAxis.COMPRESSION_FIDELITY)
            fig_other = build_per_frame_figure(
                benchmark_records, metric_name,
                axis=other_axis,
                title=f"{metric_name.upper()} per Frame ({other_axis.value.replace('_', ' ').title()})",
            )
            if fig_other.data:  # only show if there's actual trace data
                charts.append(html.Div(
                    dcc.Graph(figure=fig_other, config={"displaylogo": False}),
                    className="chart-container",
                ))

            # Heatmap
            heatmap_fig = build_heatmap_figure(
                benchmark_records, metric_name,
                axis=axis,
                title=f"{metric_name.upper()} Heatmap",
            )
            charts.append(html.Div(
                dcc.Graph(figure=heatmap_fig, config={"displaylogo": False}),
                className="chart-container mt-3",
            ))

        # VMAF per-frame charts
        if vmaf_records and metric_name == "vmaf":
            charts.append(html.H5("VMAF per Frame", className="section-title mt-4"))

            vmaf_fig = build_per_frame_figure(
                vmaf_records, "vmaf",
                title="VMAF Score per Frame (All Strategies)",
            )
            charts.append(html.Div(
                dcc.Graph(figure=vmaf_fig, config={"displaylogo": False}),
                className="chart-container",
            ))

            vmaf_heatmap = build_heatmap_figure(
                vmaf_records, "vmaf",
                title="VMAF Heatmap (Strategies × Frames)",
            )
            charts.append(html.Div(
                dcc.Graph(figure=vmaf_heatmap, config={"displaylogo": False}),
                className="chart-container",
            ))

        if not charts:
            charts.append(html.P(
                f"No per-frame data for metric '{metric_name}' with current selection.",
                className="text-muted p-3",
            ))

        return charts


# ═══════════════════════════════════════════════════════════════
#  4. EXPORT CALLBACKS
# ═══════════════════════════════════════════════════════════════

_csv_exporter = CsvExporter()
_png_exporter = PngExporter()


def _register_export_callbacks(app: dash.Dash, service: MetricsService):

    @app.callback(
        dash.Output("overview-download", "data"),
        dash.Input("overview-export-csv", "n_clicks"),
        dash.Input("overview-export-png", "n_clicks"),
        prevent_initial_call=True,
    )
    def export_overview(csv_clicks, png_clicks):
        triggered = dash.ctx.triggered_id
        records = service.get_all_records()

        if triggered == "overview-export-csv":
            config = ExportConfig(format="csv", title="4DGS Metrics Overview")
            data = _csv_exporter.export(records, config)
            return dcc.send_bytes(data, "4dgs_metrics_overview.csv")

        if triggered == "overview-export-png":
            config = ExportConfig(format="png", title="4DGS Metrics Overview", width=1400, height=700)
            data = _png_exporter.export(records, config)
            return dcc.send_bytes(data, "4dgs_metrics_overview.png")

        return no_update


# ═══════════════════════════════════════════════════════════════
#  5. COMPRESSION DETAIL CALLBACKS
# ═══════════════════════════════════════════════════════════════


def _register_compression_detail_callbacks(app: dash.Dash, service: MetricsService):

    @app.callback(
        dash.Output("detail-content-container", "children"),
        dash.Input("detail-strategy-select", "value"),
        prevent_initial_call=True,
    )
    def update_compression_detail(record_id):
        if not record_id:
            return html.P("Select a strategy.", className="text-muted p-3")

        record = service.get_record(record_id)
        if not record:
            return html.P("Record not found.", className="text-muted p-3")

        # Rebuild the detail view (same logic as compression_detail.py initial render)
        from .pages.compression_detail import (
            _build_waterfall, _build_axis_comparison,
            _build_pipeline_table, _model_info_card,
            AXIS_LABELS,
        )

        cm = record.compression_metrics

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
        if record.timing_metrics:
            t = record.timing_metrics
            if t.compress_time_s is not None:
                timing_cards.append(
                    metric_card("Compress Time", t.compress_time_s, unit="s", icon="⚡", color="#1abc9c"))
            if t.decompress_time_s is not None:
                timing_cards.append(
                    metric_card("Decompress Time", t.decompress_time_s, unit="s", icon="🔓", color="#9b59b6"))
        timing_row = metric_card_row(timing_cards, cols_per_card=3) if timing_cards else html.Div()

        waterfall = _build_waterfall(record)
        axis_chart = _build_axis_comparison(record)

        # Per-frame PSNR
        pf_charts = []
        for axis, label in AXIS_LABELS.items():
            pf = record.get_per_frame(axis)
            if pf and pf.psnr:
                fig = build_per_frame_figure(
                    [record], "psnr", axis=axis,
                    title=f"PSNR per Frame — {label}",
                )
                pf_charts.append(html.Div(
                    dcc.Graph(figure=fig, config={"displaylogo": False}),
                    className="chart-container",
                ))

        pipeline_section = html.Div()
        if record.pipeline_stats:
            pipeline_section = html.Div([
                html.H5("Pipeline Stages", className="section-title mt-4"),
                _build_pipeline_table(record.pipeline_stats),
            ])

        model_section = _model_info_card(record)

        import dash_bootstrap_components as dbc_inner
        children = [
            summary,
            timing_row,
            dbc_inner.Row([
                dbc_inner.Col(
                    html.Div(dcc.Graph(id="detail-waterfall", figure=waterfall,
                                       config={"displaylogo": False}),
                             className="chart-container"),
                    md=6,
                ),
                dbc_inner.Col(
                    html.Div(dcc.Graph(id="detail-axis-compare", figure=axis_chart,
                                       config={"displaylogo": False}),
                             className="chart-container"),
                    md=6,
                ),
            ]),
        ]

        if pf_charts:
            children.append(html.H5("Per-Frame Quality", className="section-title mt-4"))
            children.extend(pf_charts)

        children.append(pipeline_section)
        children.append(model_section)

        return children


# ═══════════════════════════════════════════════════════════════
#  TRAINING EVALUATION CALLBACKS
# ═══════════════════════════════════════════════════════════════


def _register_training_callbacks(app: dash.Dash, service: MetricsService) -> None:
    """Dynamic callbacks for the Training Evaluation page."""
    from .pages.training_eval import (
        build_quality_vs_iteration,
        build_delta_chart,
        _iter_sort_key,
        LOWER_IS_BETTER,
        METRIC_COLORS,
    )
    from .components.comparison_table import comparison_table
    import plotly.graph_objects as go

    @app.callback(
        [
            dash.Output("training-quality-line", "figure"),
            dash.Output("training-delta-bar", "figure"),
            dash.Output("training-table-container", "children"),
        ],
        [
            dash.Input("training-metric-select", "value"),
            dash.Input("training-run-select", "value"),
            dash.Input("training-chart-style", "value"),
        ],
    )
    def update_training_charts(selected_metrics, selected_runs, chart_style):
        all_records = service.get_records_by_source(ResultSource.TRAINING_JSON)
        all_records.sort(key=_iter_sort_key)

        if not all_records:
            empty = _empty_figure("No training data")
            return empty, empty, html.P("No data", className="text-muted")

        # Filter by selected runs
        if selected_runs:
            run_set = set(selected_runs)
            records = [r for r in all_records if r.id in run_set]
        else:
            records = all_records

        if not records:
            empty = _empty_figure("No runs selected")
            return empty, empty, html.P("Select at least one run", className="text-muted")

        records.sort(key=_iter_sort_key)

        # Metrics selection
        metrics = selected_metrics if selected_metrics else ["psnr", "ssim", "vmaf"]
        mode = chart_style or "lines+markers"

        # Build quality chart with selected mode
        fig_q = _build_training_quality_chart(records, metrics, mode)
        fig_d = build_delta_chart(records)

        table = comparison_table(records, table_id="training-table")

        return fig_q, fig_d, table

    def _build_training_quality_chart(records, selected_metrics, mode):
        """Build quality vs iteration chart honouring metric and style selections."""
        iterations = [_iter_sort_key(r) for r in records]

        fig = go.Figure()

        lower_metrics = [m for m in selected_metrics if m in LOWER_IS_BETTER]
        upper_metrics = [m for m in selected_metrics if m not in LOWER_IS_BETTER]

        for metric_name in upper_metrics:
            vals = []
            for r in records:
                q = r.quality_metrics
                vals.append(getattr(q, metric_name, None) if q else None)
            if any(v is not None for v in vals):
                fig.add_trace(go.Scatter(
                    x=iterations, y=vals,
                    mode=mode,
                    name=metric_name.upper().replace("_", "-"),
                    line=dict(color=METRIC_COLORS.get(metric_name, "#ffffff"), width=2),
                    marker=dict(size=10),
                ))

        for metric_name in lower_metrics:
            vals = []
            for r in records:
                q = r.quality_metrics
                vals.append(getattr(q, metric_name, None) if q else None)
            if any(v is not None for v in vals):
                fig.add_trace(go.Scatter(
                    x=iterations, y=vals,
                    mode=mode,
                    name=f"{metric_name.upper().replace('_', '-')} (↓)",
                    line=dict(color=METRIC_COLORS.get(metric_name, "#ffffff"), width=2, dash="dash"),
                    marker=dict(size=10),
                    yaxis="y2",
                ))

        fig.update_layout(
            title="Quality Metrics vs Training Iteration",
            xaxis_title="Iteration",
            yaxis_title="Metric Value (↑ higher is better)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(22,33,62,0.8)",
            font=dict(color="#e0e0e0"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=60, t=60, b=40),
            height=420,
        )
        if lower_metrics:
            fig.update_layout(
                yaxis2=dict(
                    title="↓ lower is better",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
            )

        return fig

    logger.info("Training callbacks registered")


# ── Utility ─────────────────────────────────────────────────────

def _empty_figure(message: str = "No data"):
    """Return a minimal empty plotly figure with a centered message."""
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,33,62,0.8)",
        font=dict(color="#e0e0e0"),
        annotations=[dict(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="#6c7a8e"),
        )],
        height=300,
    )
    return fig
