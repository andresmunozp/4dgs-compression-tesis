"""MetricsService — core domain service that orchestrates data loading and queries."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .enums import ComparisonAxis, MetricType, ResultCategory, ResultSource
from .models import (
    ComparisonResult,
    FilterOptions,
    MetricRecord,
    PerFrameMetrics,
    QualityMetrics,
)
from .ports import IDataSourceReader

logger = logging.getLogger(__name__)


# ── Metric extraction helpers ───────────────────────────────────


def _extract_metric(
    record: MetricRecord,
    metric: MetricType,
    axis: Optional[ComparisonAxis] = None,
) -> Optional[float]:
    """Pull a single metric value from a record."""

    # Quality metrics
    _QUALITY_MAP = {
        MetricType.PSNR: "psnr",
        MetricType.SSIM: "ssim",
        MetricType.LPIPS_VGG: "lpips_vgg",
        MetricType.LPIPS_ALEX: "lpips_alex",
        MetricType.MS_SSIM: "ms_ssim",
        MetricType.D_SSIM: "d_ssim",
        MetricType.VMAF: "vmaf",
    }

    if metric in _QUALITY_MAP:
        q = record.get_quality(axis)
        if q:
            return getattr(q, _QUALITY_MAP[metric], None)
        return None

    # Compression metrics
    if metric == MetricType.COMPRESSION_RATIO:
        return record.compression_metrics.compression_ratio if record.compression_metrics else None
    if metric == MetricType.SAVINGS_PCT:
        return record.compression_metrics.savings_pct if record.compression_metrics else None

    # Streaming
    if metric == MetricType.QOE_SCORE:
        return record.streaming_metrics.qoe_score if record.streaming_metrics else None

    # Timing
    if metric == MetricType.COMPRESS_TIME:
        return record.timing_metrics.compress_time_s if record.timing_metrics else None
    if metric == MetricType.DECOMPRESS_TIME:
        return record.timing_metrics.decompress_time_s if record.timing_metrics else None
    if metric == MetricType.TRAIN_TIME:
        return record.timing_metrics.train_time_s if record.timing_metrics else None
    if metric == MetricType.RENDER_TIME:
        return record.timing_metrics.render_time_s if record.timing_metrics else None

    return None


# ── Reader registry ─────────────────────────────────────────────


class ReaderRegistry:
    """Holds all registered readers and dispatches paths to the right one."""

    def __init__(self, readers: List[IDataSourceReader]) -> None:
        self._readers = list(readers)

    def add(self, reader: IDataSourceReader) -> None:
        self._readers.append(reader)

    def get_reader_for(self, path: Path) -> Optional[IDataSourceReader]:
        for reader in self._readers:
            if reader.supports(path):
                return reader
        return None


# ── Service ─────────────────────────────────────────────────────


class MetricsService:
    """Central domain service. Implements ``IMetricsService``."""

    def __init__(self, readers: List[IDataSourceReader] | None = None) -> None:
        self._readers: List[IDataSourceReader] = readers or []
        self._registry = ReaderRegistry(self._readers)
        self._records: Dict[str, MetricRecord] = {}

    # ── Data loading ────────────────────────────────────────────

    def load_from_path(self, path: Path) -> int:
        """Load data from a single file using the matching reader."""
        path = Path(path)
        reader = self._registry.get_reader_for(path)
        if reader is None:
            logger.warning("No reader supports %s", path)
            return 0

        try:
            new_records = reader.read(path)
        except Exception:
            logger.exception("Failed to read %s", path)
            return 0

        count = 0
        for r in new_records:
            self._records[r.id] = r
            count += 1

        logger.info("Loaded %d records from %s", count, path.name)
        return count

    def auto_discover(self, base_dir: Path) -> int:
        """Scan base_dir for known data files and load them all."""
        from ..adapters.readers.directory_scanner import DirectoryScanner

        scanner = DirectoryScanner(self._readers)
        discovered = scanner.scan(Path(base_dir))

        total = 0
        for df in discovered:
            try:
                new_records = df.reader.read(df.path)
                for r in new_records:
                    self._records[r.id] = r
                    total += 1
            except Exception:
                logger.exception("Failed to read %s", df.path)

        logger.info("Auto-discovered %d records total", total)
        return total

    # ── Queries ─────────────────────────────────────────────────

    def get_all_records(self) -> List[MetricRecord]:
        return list(self._records.values())

    def get_record(self, record_id: str) -> Optional[MetricRecord]:
        return self._records.get(record_id)

    def get_records_by_category(self, category: ResultCategory) -> List[MetricRecord]:
        return [r for r in self._records.values() if r.category == category]

    def get_records_by_source(self, source: ResultSource) -> List[MetricRecord]:
        return [r for r in self._records.values() if r.source == source]

    def search_records(
        self,
        *,
        category: Optional[ResultCategory] = None,
        source: Optional[ResultSource] = None,
        scene: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[MetricRecord]:
        """Flexible search with optional filters (all ANDed)."""
        results = list(self._records.values())

        if category is not None:
            results = [r for r in results if r.category == category]
        if source is not None:
            results = [r for r in results if r.source == source]
        if scene is not None:
            results = [r for r in results if r.scene == scene]
        if name is not None:
            low = name.lower()
            results = [r for r in results if low in r.name.lower()]
        if tags:
            for k, v in tags.items():
                results = [r for r in results if r.tags.get(k) == v]

        return results

    # ── Comparison ──────────────────────────────────────────────

    def compare(
        self,
        record_ids: List[str],
        metrics: List[MetricType],
        axis: Optional[ComparisonAxis] = None,
    ) -> ComparisonResult:
        """Build a comparison table for selected records and metrics."""
        records = [self._records[rid] for rid in record_ids if rid in self._records]
        metric_names = [m.value for m in metrics]
        values: List[List[Optional[float]]] = []

        for record in records:
            row = [_extract_metric(record, m, axis) for m in metrics]
            values.append(row)

        return ComparisonResult(
            record_ids=[r.id for r in records],
            record_names=[r.display_name for r in records],
            metric_names=metric_names,
            values=values,
        )

    # ── Per-frame ───────────────────────────────────────────────

    def get_per_frame_data(
        self,
        record_id: str,
        axis: Optional[ComparisonAxis] = None,
    ) -> Optional[PerFrameMetrics]:
        record = self._records.get(record_id)
        if record is None:
            return None
        return record.get_per_frame(axis)

    # ── Filter options (for UI dropdowns) ───────────────────────

    def get_available_filters(self) -> FilterOptions:
        sources = set()
        categories = set()
        scenes = set()
        names = set()
        tag_keys: Dict[str, set] = {}

        for r in self._records.values():
            sources.add(r.source)
            categories.add(r.category)
            if r.scene:
                scenes.add(r.scene)
            names.add(r.name)
            for k, v in r.tags.items():
                tag_keys.setdefault(k, set()).add(v)

        return FilterOptions(
            sources=sorted(sources, key=lambda s: s.value),
            categories=sorted(categories, key=lambda c: c.value),
            scenes=sorted(scenes),
            names=sorted(names),
            tags={k: sorted(v) for k, v in tag_keys.items()},
        )

    # ── Aggregation helpers ─────────────────────────────────────

    def group_by_name(self) -> Dict[str, List[MetricRecord]]:
        groups: Dict[str, List[MetricRecord]] = {}
        for r in self._records.values():
            groups.setdefault(r.name, []).append(r)
        return groups

    def group_by_scene(self) -> Dict[str, List[MetricRecord]]:
        groups: Dict[str, List[MetricRecord]] = {}
        for r in self._records.values():
            key = r.scene or "unknown"
            groups.setdefault(key, []).append(r)
        return groups

    def group_by_source(self) -> Dict[ResultSource, List[MetricRecord]]:
        groups: Dict[ResultSource, List[MetricRecord]] = {}
        for r in self._records.values():
            groups.setdefault(r.source, []).append(r)
        return groups
