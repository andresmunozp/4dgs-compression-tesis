"""ComparisonTable — DataTable component with colour-coded best/worst values."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import dash_bootstrap_components as dbc
from dash import dash_table, html
import pandas as pd

from ...domain.enums import ComparisonAxis, MetricType
from ...domain.models import MetricRecord


# Metrics where higher is better
_HIGHER_IS_BETTER = {
    "psnr", "ssim", "ms_ssim", "vmaf",
    "compression_ratio", "savings_pct",
    "effective_throughput_mbps",
}

# Metrics where lower is better
_LOWER_IS_BETTER = {
    "lpips_vgg", "lpips_alex", "d_ssim",
    "compress_time", "decompress_time",
    "startup_delay_s", "rebuffer_events", "total_stall_duration_s",
    "e2e_latency_s",
}


def records_to_dataframe(
    records: List[MetricRecord],
    axis: Optional[ComparisonAxis] = None,
) -> pd.DataFrame:
    """Convert a list of MetricRecords into a DataFrame for display."""
    rows: List[Dict[str, Any]] = []

    for r in records:
        row: Dict[str, Any] = {
            "Name": r.display_name,
            "Source": r.source.value,
            "Category": r.category.value,
            "Scene": r.scene or "—",
        }

        # Quality metrics
        q = r.get_quality(axis)
        if q:
            d = q.non_none_dict()
            for k, v in d.items():
                col = k.upper().replace("_", " ")
                row[col] = round(v, 4) if isinstance(v, float) else v

        # Compression
        if r.compression_metrics:
            cm = r.compression_metrics
            row["Ratio"] = round(cm.compression_ratio, 2)
            row["Savings %"] = round(cm.savings_pct, 1)
            row["Orig MB"] = round(cm.original_size_bytes / 1e6, 2)
            row["Comp MB"] = round(cm.compressed_size_bytes / 1e6, 2)

        # Timing
        if r.timing_metrics:
            t = r.timing_metrics
            if t.compress_time_s is not None:
                row["Compress (s)"] = round(t.compress_time_s, 3)
            if t.decompress_time_s is not None:
                row["Decompress (s)"] = round(t.decompress_time_s, 3)
            if t.train_time_s is not None:
                row["Train (s)"] = round(t.train_time_s, 1)

        # Streaming
        if r.streaming_metrics:
            s = r.streaming_metrics
            row["QoE"] = round(s.qoe_score, 2)
            row["Startup (s)"] = round(s.startup_delay_s, 2)
            row["Stalls"] = s.rebuffer_events

        rows.append(row)

    return pd.DataFrame(rows)


def _conditional_styles(df: pd.DataFrame) -> list:
    """Generate conditional formatting rules for the DataTable."""
    styles = []
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    for col in numeric_cols:
        col_lower = col.lower().replace(" ", "_")

        # Determine direction
        higher_good = any(k in col_lower for k in ["psnr", "ssim", "vmaf", "ratio", "saving", "qoe"])
        lower_good = any(k in col_lower for k in ["lpips", "compress", "decompress", "startup", "stall", "latency"])

        vals = df[col].dropna()
        if len(vals) < 2:
            continue

        best_val = vals.max() if higher_good else vals.min()
        worst_val = vals.min() if higher_good else vals.max()

        # Highlight best in green
        styles.append({
            "if": {
                "filter_query": f'{{{col}}} = {best_val}',
                "column_id": col,
            },
            "backgroundColor": "rgba(46, 204, 113, 0.2)",
            "color": "#2ecc71",
            "fontWeight": "bold",
        })

        # Highlight worst in red
        if best_val != worst_val:
            styles.append({
                "if": {
                    "filter_query": f'{{{col}}} = {worst_val}',
                    "column_id": col,
                },
                "backgroundColor": "rgba(231, 76, 60, 0.15)",
                "color": "#e74c3c",
            })

    return styles


def build_table_data(
    records: List[MetricRecord],
    axis: Optional[ComparisonAxis] = None,
) -> tuple:
    """Return (data, columns, style_data_conditional) for a DataTable update."""
    df = records_to_dataframe(records, axis)
    if df.empty:
        return [], [], []
    columns = [{"name": c, "id": c} for c in df.columns]
    cond_styles = _conditional_styles(df)
    styles = [{"if": {"row_index": "odd"}, "backgroundColor": "#1a1a40"}, *cond_styles]
    return df.to_dict("records"), columns, styles


def comparison_table(
    records: List[MetricRecord],
    axis: Optional[ComparisonAxis] = None,
    table_id: str = "comparison-table",
) -> html.Div:
    """Build a Dash DataTable from MetricRecords with colour-coded cells."""
    df = records_to_dataframe(records, axis)

    if df.empty:
        return html.Div("No data available.", className="text-muted p-3")

    columns = [{"name": c, "id": c} for c in df.columns]

    cond_styles = _conditional_styles(df)

    table = dash_table.DataTable(
        id=table_id,
        data=df.to_dict("records"),
        columns=columns,
        sort_action="native",
        filter_action="native",
        page_size=20,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#0f3460",
            "color": "#e0e0e0",
            "fontWeight": "600",
            "textAlign": "center",
            "border": "1px solid #16213e",
            "fontSize": "0.8rem",
        },
        style_cell={
            "backgroundColor": "#16213e",
            "color": "#e0e0e0",
            "border": "1px solid #0f3460",
            "textAlign": "center",
            "fontSize": "0.82rem",
            "padding": "8px 12px",
            "minWidth": "80px",
        },
        style_cell_conditional=[
            {"if": {"column_id": "Name"}, "textAlign": "left", "minWidth": "180px"},
            {"if": {"column_id": "Source"}, "minWidth": "120px"},
        ],
        style_data_conditional=[
            # Zebra stripes
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "#1a1a40",
            },
            *cond_styles,
        ],
        style_filter={
            "backgroundColor": "#0b1e3d",
            "color": "#e0e0e0",
        },
    )

    return html.Div(table, className="comparison-table")
