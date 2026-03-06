"""Main Dash layout — sidebar navigation + page content area.

Phase 6: Polished with Vercel Web Interface Guidelines compliance —
skip-nav, aria-labels, focus-visible, loading states, meta theme-color.
"""

from __future__ import annotations

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from ..domain.services import MetricsService
from .callbacks import register_callbacks
from .pages.overview import build_overview
from .pages.benchmark_compare import build_benchmark_compare
from .pages.per_frame_analysis import build_per_frame_analysis
from .pages.training_eval import build_training_eval
from .pages.streaming_qoe import build_streaming_qoe
from .pages.compression_detail import build_compression_detail
from .pages.compression_analysis import build_compression_analysis
from .pages.decompression_analysis import build_decompression_analysis


def _compute_data_hash(service: MetricsService) -> str:
    """Fast hash of current data state (record count + IDs) for change detection."""
    records = service.get_all_records()
    parts = sorted(r.id for r in records)
    return f"{len(parts)}:{hash(tuple(parts))}"


# ── Navigation items ────────────────────────────────────────────

NAV_ITEMS = [
    {"label": "Overview",              "href": "/",                       "id": "nav-overview",          "icon": "📊"},
    {"label": "Benchmark Compare",     "href": "/benchmark",              "id": "nav-benchmark",         "icon": "🔧"},
    {"label": "Per-Frame Analysis",    "href": "/per-frame",              "id": "nav-perframe",          "icon": "📈"},
    {"label": "Training Eval",         "href": "/training",               "id": "nav-training",          "icon": "🎯"},
    {"label": "Streaming QoE",         "href": "/streaming",              "id": "nav-streaming",         "icon": "📡"},
    {"label": "Compression Detail",    "href": "/compression",            "id": "nav-compression",       "icon": "🗜️"},
    {"label": "Compression Analysis",  "href": "/compression-analysis",   "id": "nav-comp-analysis",     "icon": "📦"},
    {"label": "Decompression Analysis", "href": "/decompression-analysis", "id": "nav-decomp-analysis",   "icon": "🔄"},
]


def _sidebar(initial_count: int = 0) -> html.Nav:
    """Build the sidebar with navigation links.

    Uses <nav> with aria-label for accessibility.
    Icons are wrapped with aria-hidden to not be read by screen readers.
    """
    nav_links = [
        dbc.NavLink(
            [
                html.Span(item["icon"], className="nav-icon", **{"aria-hidden": "true"}),
                item["label"],
            ],
            href=item["href"],
            id=item["id"],
            active="exact",
            style={"fontSize": "0.9rem"},
        )
        for item in NAV_ITEMS
    ]

    return html.Nav([
        html.Div([
            html.Div("4DGaussians", style={"fontSize": "1.1rem"}),
            html.Div("Metrics Viewer", style={"fontSize": "0.9rem"}),
            html.Small("v0.2.0"),
        ], className="sidebar-brand"),
        dbc.Nav(nav_links, vertical=True, pills=True),
        html.Hr(style={"borderColor": "#0f3460"}),
        html.Div([
            html.Small("Data loaded at startup", style={"color": "#6c7a8e"}),
            html.Br(),
            html.Small(f"{initial_count} records loaded",
                       id="sidebar-record-count", style={"color": "#a0b4d0"}),
        ], className="mt-3 px-2"),
    ], className="sidebar", **{"aria-label": "Main navigation"})


def create_app(service: MetricsService, assets_folder: str | None = None,
               refresh_interval_s: int = 30) -> dash.Dash:
    """Create and configure the Dash application.

    Args:
        service: Fully initialized MetricsService with data loaded.
        assets_folder: Absolute path to the ``assets/`` directory.
        refresh_interval_s: Seconds between auto-refresh checks.  0 = disabled.

    Returns:
        A configured ``dash.Dash`` instance ready to ``.run()``.
    """
    if assets_folder is None:
        assets_folder = str(Path(__file__).resolve().parent.parent / "assets")

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        assets_folder=assets_folder,
        suppress_callback_exceptions=True,
        title="4DGS Metrics Viewer",
        update_title="Loading\u2026",
        meta_tags=[
            {"name": "theme-color", "content": "#1a1a2e"},
            {"name": "color-scheme", "content": "dark"},
            {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        ],
    )

    # Pre-build all pages
    pages = {
        "/": build_overview(service),
        "/benchmark": build_benchmark_compare(service),
        "/per-frame": build_per_frame_analysis(service),
        "/training": build_training_eval(service),
        "/streaming": build_streaming_qoe(service),
        "/compression": build_compression_detail(service),
        "/compression-analysis": build_compression_analysis(service),
        "/decompression-analysis": build_decompression_analysis(service),
    }

    total_records = len(service.get_all_records())

    app.layout = html.Div([
        dcc.Location(id="url", refresh=False),

        dbc.Row([
            # Sidebar
            dbc.Col(
                _sidebar(initial_count=total_records),
                width=2,
                style={"padding": 0},
            ),
            # Main content
            dbc.Col(
                html.Main(
                    dcc.Loading(
                        html.Div(id="page-content", className="page-content"),
                        type="circle",
                        color="#e94560",
                    ),
                    id="main-content",
                    role="main",
                ),
                width=10,
            ),
        ], style={"minHeight": "100dvh"}),

        # Hidden store for record count
        dcc.Store(id="record-count-store", data=total_records),

        # Auto-refresh interval (disabled if 0)
        dcc.Interval(
            id="auto-refresh-interval",
            interval=refresh_interval_s * 1000 if refresh_interval_s > 0 else 86400_000,
            disabled=refresh_interval_s <= 0,
            n_intervals=0,
        ),
        dcc.Store(id="last-data-hash", data=_compute_data_hash(service)),

        # Live region for async status messages — a11y
        html.Div(id="live-status", className="sr-only",
                 **{"aria-live": "polite", "aria-atomic": "true"}),
    ])

    # ── Routing callback ────────────────────────────────────
    @app.callback(
        dash.Output("page-content", "children"),
        dash.Input("url", "pathname"),
    )
    def display_page(pathname: str):
        return pages.get(pathname, pages["/"])

    # ── Dynamic callbacks ───────────────────────────────────
    register_callbacks(app, service)

    # ── Auto-refresh: detect data changes on disk ───────────
    @app.callback(
        [dash.Output("live-status", "children"),
         dash.Output("sidebar-record-count", "children", allow_duplicate=True),
         dash.Output("last-data-hash", "data")],
        dash.Input("auto-refresh-interval", "n_intervals"),
        dash.State("last-data-hash", "data"),
        prevent_initial_call=True,
    )
    def _check_data_refresh(n_intervals, prev_hash):
        """Check if underlying data files changed — notify user via live region."""
        current_hash = _compute_data_hash(service)
        if current_hash == prev_hash:
            return dash.no_update, dash.no_update, dash.no_update

        count = len(service.get_all_records())
        status_msg = f"Data refreshed \u2014 {count} records loaded"
        return status_msg, f"{count} records loaded", current_hash

    return app
