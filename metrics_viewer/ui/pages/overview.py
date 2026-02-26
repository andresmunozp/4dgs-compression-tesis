"""Overview page — summary dashboard with metric cards and full comparison table."""

from __future__ import annotations

from typing import List, Optional

import dash_bootstrap_components as dbc
from dash import html

from ...domain.enums import ComparisonAxis, MetricType, ResultSource
from ...domain.models import MetricRecord
from ...domain.services import MetricsService, _extract_metric
from ..components.comparison_table import comparison_table
from ..components.metric_card import metric_card, metric_card_row
from ..components.filters import filter_panel
from ..components.export_button import export_button


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


def build_overview(service: MetricsService) -> html.Div:
    """Construct the full Overview page layout."""
    all_records = service.get_all_records()
    filters = service.get_available_filters()

    # Separate by source for stats
    benchmark_records = service.get_records_by_source(ResultSource.BENCHMARK_JSON)
    training_records = service.get_records_by_source(ResultSource.TRAINING_JSON)

    # Quick stats
    total_records = len(all_records)
    num_strategies = len(set(r.name for r in benchmark_records))
    num_scenes = len(filters.scenes)
    num_sources = len(filters.sources)

    # Best metrics across benchmarks
    best_psnr, best_psnr_name = _best(benchmark_records, MetricType.PSNR, ComparisonAxis.END_TO_END)
    best_ratio, best_ratio_name = _best(benchmark_records, MetricType.COMPRESSION_RATIO)
    best_ssim, best_ssim_name = _best(benchmark_records, MetricType.SSIM, ComparisonAxis.COMPRESSION_FIDELITY)

    # Best training
    best_train_psnr, best_train_name = _best(training_records, MetricType.PSNR)

    # ── Summary cards ───────────────────────────────────────
    summary_cards = metric_card_row([
        metric_card("Total Records", total_records, icon="📊", color="#3498db"),
        metric_card("Strategies", num_strategies, icon="🔧", color="#533483"),
        metric_card("Scenes", num_scenes or "1", icon="🎬", color="#2ecc71"),
        metric_card("Data Sources", num_sources, icon="📁", color="#f39c12"),
    ])

    # Best metrics cards
    best_cards = metric_card_row([
        metric_card("Best E2E PSNR", best_psnr or 0, unit="dB", icon="🏆", color="#e94560"),
        metric_card("Best Ratio", best_ratio or 0, unit="×", icon="📦", color="#533483"),
        metric_card("Best Fidelity SSIM", best_ssim or 0, icon="✨", color="#2ecc71"),
        metric_card("Best Training PSNR", best_train_psnr or 0, unit="dB", icon="🎯", color="#3498db"),
    ], cols_per_card=3)

    # ── Filter panel ────────────────────────────────────────
    filter_section = filter_panel(filters, panel_id_prefix="overview-filter")

    # ── Comparison table with benchmark JSON records ────────
    # Show benchmark JSON records first (richest data), then training
    priority_records = benchmark_records + training_records

    # Add any remaining records that aren't duplicates
    shown_ids = {r.id for r in priority_records}
    for r in all_records:
        if r.id not in shown_ids:
            priority_records.append(r)

    table = comparison_table(
        priority_records,
        axis=ComparisonAxis.END_TO_END,
        table_id="overview-table",
    )

    # ── Best performers note ────────────────────────────────
    best_notes = []
    if best_psnr_name != "—":
        best_notes.append(html.Li(f"Best E2E PSNR: {best_psnr_name} ({best_psnr:.2f} dB)"))
    if best_ratio_name != "—":
        best_notes.append(html.Li(f"Best Compression Ratio: {best_ratio_name} ({best_ratio:.2f}×)"))
    if best_train_name != "—":
        best_notes.append(html.Li(f"Best Training PSNR: {best_train_name} ({best_train_psnr:.2f} dB)"))

    return html.Div([
        html.H4("Overview", className="section-title"),
        html.P("Summary of all loaded benchmark, training, and VMAF results.",
               className="section-subtitle"),

        # Export controls
        html.Div([
            export_button(button_id="overview-export", download_id="overview-download"),
        ], className="mb-3"),

        html.Div(id="overview-summary-container", children=[summary_cards, best_cards]),

        html.H5("Filters", className="section-title mt-4"),
        filter_section,

        html.H5("All Results", className="section-title mt-4"),
        html.Div(id="overview-table-container", children=[table]),

        html.Div(id="overview-performers-container", children=[
            html.Div([
                html.H6("🏆 Top Performers", className="mt-4 mb-2", style={"color": "#f39c12"}),
                html.Ul(best_notes, style={"color": "#a0b4d0", "fontSize": "0.9rem"}),
            ]) if best_notes else None,
        ]),
    ])
