"""CSV exporter — writes MetricRecords to a CSV byte stream.

Implements the ``IResultExporter`` protocol from ``domain.ports``.
"""

from __future__ import annotations

import io
from typing import List, Optional

import pandas as pd

from ...domain.enums import ComparisonAxis
from ...domain.models import MetricRecord
from ...domain.ports import ExportConfig
from ...domain.services import _extract_metric


_ALL_QUALITY_COLS = ["psnr", "ssim", "lpips_vgg", "lpips_alex", "ms_ssim", "d_ssim", "vmaf"]


def _records_to_df(records: List[MetricRecord], axis: Optional[ComparisonAxis] = None) -> pd.DataFrame:
    """Convert a list of MetricRecord into a flat DataFrame."""
    rows = []
    for r in records:
        row = {
            "id": r.id,
            "name": r.display_name,
            "source": r.source.value,
            "category": r.category.value,
            "scene": r.scene,
        }

        # Quality metrics
        q = r.get_quality(axis)
        if not q and r.quality_axes:
            # Try any available axis
            q = next(iter(r.quality_axes.values()), None)
        if q:
            for col in _ALL_QUALITY_COLS:
                row[col] = getattr(q, col, None)

        # Compression metrics
        cm = r.compression_metrics
        if cm:
            row["original_size_bytes"] = cm.original_size_bytes
            row["compressed_size_bytes"] = cm.compressed_size_bytes
            row["compression_ratio"] = cm.compression_ratio
            row["savings_pct"] = cm.savings_pct

        # Timing metrics
        tm = r.timing_metrics
        if tm:
            row["compress_time_s"] = tm.compress_time_s
            row["decompress_time_s"] = tm.decompress_time_s
            row["train_time_s"] = tm.train_time_s

        # Streaming
        sm = r.streaming_metrics
        if sm:
            row["qoe_score"] = sm.qoe_score
            row["bandwidth_mbps"] = sm.bandwidth_mbps
            row["rebuffer_events"] = sm.rebuffer_events

        # Tags
        for k, v in r.tags.items():
            row[f"tag_{k}"] = v

        rows.append(row)

    return pd.DataFrame(rows)


class CsvExporter:
    """Exports MetricRecords to CSV bytes."""

    @property
    def supported_formats(self) -> List[str]:
        return ["csv"]

    def export(self, records: List[MetricRecord], config: ExportConfig) -> bytes:
        """Serialize records into CSV format and return raw bytes."""
        axis_str = config.extra.get("axis")
        axis = None
        if axis_str:
            axis = ComparisonAxis(axis_str)

        df = _records_to_df(records, axis=axis)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")
