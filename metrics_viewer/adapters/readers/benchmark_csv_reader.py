"""Reader for benchmark_results/benchmark_summary*.csv files."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import List, Optional

from ...domain.enums import ComparisonAxis, ResultCategory, ResultSource
from ...domain.models import (
    CompressionMetrics,
    MetricRecord,
    ModelInfo,
    QualityMetrics,
    StreamingMetrics,
    TimingMetrics,
)


def _float(val: str) -> Optional[float]:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _int(val: str) -> Optional[int]:
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


class BenchmarkCsvReader:
    """Adapter that reads ``benchmark_summary*.csv`` files.

    These CSV files contain one row per strategy with mean metrics only
    (no per-frame data).  Useful as a quick-load fallback.
    """

    @property
    def source_type(self) -> ResultSource:
        return ResultSource.BENCHMARK_CSV

    def supports(self, path: Path) -> bool:
        if not path.is_file():
            return False
        name = path.name.lower()
        return name.startswith("benchmark_summary") and name.endswith(".csv")

    def read(self, path: Path) -> List[MetricRecord]:
        version = self._detect_version(path)
        records: List[MetricRecord] = []

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                record = self._parse_row(row, version, path)
                records.append(record)

        return records

    # ── Internals ───────────────────────────────────────────────

    @staticmethod
    def _detect_version(path: Path) -> str:
        stem = path.stem.lower()
        if "antes" in stem:
            return "antes"
        return "current"

    @staticmethod
    def _make_id(config_name: str, version: str) -> str:
        raw = f"benchmark_csv:{config_name}:{version}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _parse_row(self, row: dict, version: str, path: Path) -> MetricRecord:
        config_name = row.get("config", "unknown").strip()
        record_id = self._make_id(config_name, version)

        # Build quality axes
        quality_axes = {}

        # Compression fidelity
        cf = QualityMetrics(
            psnr=_float(row.get("cf_psnr_db")),
            ssim=_float(row.get("cf_ssim")),
            lpips_vgg=_float(row.get("cf_lpips")),
        )
        quality_axes[ComparisonAxis.COMPRESSION_FIDELITY] = cf

        # End-to-end
        e2e = QualityMetrics(
            psnr=_float(row.get("e2e_psnr_db")),
            ssim=_float(row.get("e2e_ssim")),
            lpips_vgg=_float(row.get("e2e_lpips")),
        )
        quality_axes[ComparisonAxis.END_TO_END] = e2e

        # Training baseline
        bl = QualityMetrics(
            psnr=_float(row.get("bl_psnr_db")),
            ssim=_float(row.get("bl_ssim")),
            lpips_vgg=_float(row.get("bl_lpips")),
        )
        quality_axes[ComparisonAxis.TRAINING_BASELINE] = bl

        # Compression metrics
        orig_mb = _float(row.get("orig_MB")) or 0.0
        comp_mb = _float(row.get("comp_MB")) or 0.0
        compression = CompressionMetrics(
            original_size_bytes=int(orig_mb * 1_000_000),
            compressed_size_bytes=int(comp_mb * 1_000_000),
            compression_ratio=_float(row.get("ratio")) or 1.0,
            savings_pct=_float(row.get("savings%")) or 0.0,
            num_chunks=_int(row.get("chunks")) or 0,
        )

        # Timing
        timing = TimingMetrics(
            compress_time_s=_float(row.get("compress_s")),
            decompress_time_s=_float(row.get("decode_s")),
        )

        # Streaming
        streaming = StreamingMetrics(
            startup_delay_s=_float(row.get("startup_s")) or 0.0,
            rebuffer_events=_int(row.get("rebuffers")) or 0,
            total_stall_duration_s=_float(row.get("stall_s")) or 0.0,
            qoe_score=_float(row.get("qoe")) or 0.0,
        )

        # Model info
        model_info = ModelInfo(
            num_gaussians_original=_int(row.get("gaussians_orig")),
            num_gaussians_compressed=_int(row.get("gaussians_comp")),
            sh_degree_original=_int(row.get("sh_orig")),
            sh_degree_compressed=_int(row.get("sh_comp")),
        )

        return MetricRecord(
            id=record_id,
            source=ResultSource.BENCHMARK_CSV,
            category=ResultCategory.COMPRESSION,
            name=config_name,
            tags={
                "version": version,
                "source_path": str(path),
            },
            quality_axes=quality_axes,
            quality_metrics=e2e,
            compression_metrics=compression,
            streaming_metrics=streaming,
            timing_metrics=timing,
            model_info=model_info,
        )
