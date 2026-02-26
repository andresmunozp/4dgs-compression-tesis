"""Training Evaluation page — compare training runs across iterations."""

from __future__ import annotations

import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import dcc, html

from ...domain.enums import MetricType, ResultSource
from ...domain.models import MetricRecord
from ...domain.services import MetricsService, _extract_metric
from ..components.bar_chart import stacked_bar_timing
from ..components.comparison_table import comparison_table
from ..components.metric_card import metric_card, metric_card_row

COLORS = [
    "#e94560", "#533483", "#2ecc71", "#f39c12", "#3498db",
    "#e74c3c", "#1abc9c", "#9b59b6", "#d35400", "#0f3460",
]

# Available metrics for the line charts
TRAINING_METRICS = [
    {"label": "PSNR (dB)", "value": "psnr"},
    {"label": "SSIM", "value": "ssim"},
    {"label": "VMAF", "value": "vmaf"},
    {"label": "MS-SSIM", "value": "ms_ssim"},
    {"label": "LPIPS-vgg (↓)", "value": "lpips_vgg"},
    {"label": "D-SSIM (↓)", "value": "d_ssim"},
]

METRIC_COLORS = {
    "psnr": "#e94560",
    "ssim": "#2ecc71",
    "vmaf": "#f39c12",
    "ms_ssim": "#3498db",
    "lpips_vgg": "#e74c3c",
    "d_ssim": "#9b59b6",
}

LOWER_IS_BETTER = {"lpips_vgg", "d_ssim"}


def _iter_sort_key(r: MetricRecord) -> int:
    it = r.tags.get("iteration", "0")
    try:
        return int(it)
    except ValueError:
        return 0


def build_quality_vs_iteration(records: list[MetricRecord],
                                selected_metrics: list[str] | None = None,
                                ) -> go.Figure:
    """Build a line chart of selected quality metrics vs training iteration."""
    if selected_metrics is None:
        selected_metrics = ["psnr", "ssim", "vmaf"]

    records = sorted(records, key=_iter_sort_key)
    iterations = [_iter_sort_key(r) for r in records]

    fig = go.Figure()

    # Separate higher-is-better from lower-is-better
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
                mode="lines+markers",
                name=metric_name.upper().replace("_", "-"),
                line=dict(color=METRIC_COLORS.get(metric_name, "#ffffff"), width=2),
                marker=dict(size=10),
            ))

    # Lower-is-better metrics on secondary y-axis
    for metric_name in lower_metrics:
        vals = []
        for r in records:
            q = r.quality_metrics
            vals.append(getattr(q, metric_name, None) if q else None)
        if any(v is not None for v in vals):
            fig.add_trace(go.Scatter(
                x=iterations, y=vals,
                mode="lines+markers",
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


def build_delta_chart(records: list[MetricRecord]) -> go.Figure:
    """Show improvement (delta) between successive iterations."""
    records = sorted(records, key=_iter_sort_key)
    if len(records) < 2:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text="Need ≥ 2 iterations for delta", showarrow=False,
                              font=dict(size=14, color="#6c7a8e"))],
        )
        return fig

    metrics = ["psnr", "ssim", "vmaf"]
    metric_labels = ["ΔPSNR", "ΔSSIM", "ΔVMAF"]
    fig = go.Figure()

    for i in range(1, len(records)):
        prev_q = records[i - 1].quality_metrics
        curr_q = records[i].quality_metrics
        if not prev_q or not curr_q:
            continue
        it_label = f"it{_iter_sort_key(records[i-1])}→it{_iter_sort_key(records[i])}"
        deltas = []
        for m in metrics:
            p = getattr(prev_q, m, None)
            c = getattr(curr_q, m, None)
            deltas.append((c - p) if (c is not None and p is not None) else 0)

        fig.add_trace(go.Bar(
            x=metric_labels,
            y=deltas,
            name=it_label,
            marker_color=COLORS[(i - 1) % len(COLORS)],
            text=[f"{d:+.4f}" for d in deltas],
            textposition="auto",
        ))

    fig.update_layout(
        barmode="group",
        title="Improvement Between Iterations",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(22,33,62,0.8)",
        font=dict(color="#e0e0e0"),
        yaxis_title="Delta",
        margin=dict(l=50, r=20, t=60, b=40),
        height=350,
    )
    return fig


def build_training_eval(service: MetricsService) -> html.Div:
    """Construct the Training Evaluation page with interactive selectors."""
    records = service.get_records_by_source(ResultSource.TRAINING_JSON)
    records.sort(key=_iter_sort_key)

    if not records:
        return html.Div([
            html.H4("Training Evaluation", className="section-title"),
            html.P("No training evaluation JSON files found in results_json/.",
                   className="text-muted"),
        ])

    # ── Summary cards ───────────────────────────────────────
    latest = records[-1]
    q = latest.quality_metrics

    cards = [
        metric_card("Iterations Evaluated", len(records), icon="🔄", color="#3498db"),
    ]
    if q:
        if q.psnr:
            cards.append(metric_card("Latest PSNR", q.psnr, unit="dB", icon="📈", color="#e94560"))
        if q.ssim:
            cards.append(metric_card("Latest SSIM", q.ssim, icon="✨", color="#2ecc71"))
        if q.vmaf:
            cards.append(metric_card("Latest VMAF", q.vmaf, icon="🎥", color="#f39c12"))

    summary_row = metric_card_row(cards, cols_per_card=3)

    # ── Build run selector options ──────────────────────────
    run_options = [{"label": r.display_name, "value": r.id} for r in records]
    default_run_ids = [r.id for r in records]

    # ── Default metric selection ────────────────────────────
    default_metrics = ["psnr", "ssim", "vmaf"]

    # ── Build initial charts ────────────────────────────────
    fig_quality = build_quality_vs_iteration(records, default_metrics)
    fig_delta = build_delta_chart(records)

    # ── Timing chart ────────────────────────────────────────
    timing_chart = stacked_bar_timing(
        records,
        chart_id="training-timing",
        title="Time Breakdown per Training Run",
    )

    # ── PLY size info ───────────────────────────────────────
    ply_info_items = []
    for r in records:
        mi = r.model_info
        if mi and mi.total_ply_size_bytes:
            gb = mi.total_ply_size_bytes / 1e9
            ply_info_items.append(
                html.Li(f"{r.display_name}: {mi.num_ply_files or '?'} PLY files, {gb:.2f} GB total")
            )

    # ── Table ───────────────────────────────────────────────
    table = comparison_table(records, table_id="training-table")

    return html.Div([
        html.H4("Training Evaluation", className="section-title"),
        html.P("Compare training runs at different iterations. Track quality improvement over training time.",
               className="section-subtitle"),

        summary_row,

        # ── Selectors ───────────────────────────────────────
        dbc.Row([
            dbc.Col([
                html.Label("Metrics to Plot", htmlFor="training-metric-select", style={
                    "fontSize": "0.8rem", "color": "#a0b4d0", "textTransform": "uppercase"}),
                dcc.Dropdown(
                    id="training-metric-select",
                    options=TRAINING_METRICS,
                    value=default_metrics,
                    multi=True,
                    placeholder="Select metrics…",
                    style={"backgroundColor": "#0b1e3d", "color": "#e0e0e0",
                           "border": "1px solid #0f3460"},
                ),
            ], md=5),
            dbc.Col([
                html.Label("Training Runs", htmlFor="training-run-select", style={
                    "fontSize": "0.8rem", "color": "#a0b4d0", "textTransform": "uppercase"}),
                dcc.Dropdown(
                    id="training-run-select",
                    options=run_options,
                    value=default_run_ids,
                    multi=True,
                    placeholder="Select runs…",
                    style={"backgroundColor": "#0b1e3d", "color": "#e0e0e0",
                           "border": "1px solid #0f3460"},
                ),
            ], md=5),
            dbc.Col([
                html.Label("Chart Style", htmlFor="training-chart-style", style={
                    "fontSize": "0.8rem", "color": "#a0b4d0", "textTransform": "uppercase"}),
                dcc.Dropdown(
                    id="training-chart-style",
                    options=[
                        {"label": "Lines + Markers", "value": "lines+markers"},
                        {"label": "Lines Only", "value": "lines"},
                        {"label": "Markers Only", "value": "markers"},
                    ],
                    value="lines+markers",
                    clearable=False,
                    style={"backgroundColor": "#0b1e3d", "color": "#e0e0e0",
                           "border": "1px solid #0f3460"},
                ),
            ], md=2),
        ], className="filter-panel mb-4 p-3"),

        # ── Dynamic charts container ─────────────────────────
        html.Div(id="training-charts-container", children=[
            dbc.Row([
                dbc.Col(
                    html.Div(
                        dcc.Graph(id="training-quality-line", figure=fig_quality,
                                  config={"displaylogo": False}),
                        className="chart-container",
                    ),
                    md=7,
                ),
                dbc.Col(
                    html.Div(
                        dcc.Graph(id="training-delta-bar", figure=fig_delta,
                                  config={"displaylogo": False}),
                        className="chart-container",
                    ),
                    md=5,
                ),
            ]),
        ]),

        timing_chart,

        # PLY info
        html.Div([
            html.H6("💾 Per-frame PLY Export Info", className="mt-3", style={"color": "#a0b4d0"}),
            html.Ul(ply_info_items, style={"color": "#6c7a8e", "fontSize": "0.85rem"}),
        ]) if ply_info_items else None,

        html.H5("Detailed Results", className="section-title mt-4"),
        html.Div(id="training-table-container", children=[table]),
    ])
