"""Reader for benchmark_results/benchmark_results*.json files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List

from ...domain.enums import ComparisonAxis, ResultCategory, ResultSource
from ...domain.models import (
    CompressionMetrics,
    MetricRecord,
    ModelInfo,
    PerFrameMetrics,
    PipelineStageStats,
    QualityMetrics,
    StreamingMetrics,
    TimingMetrics,
)


class BenchmarkJsonReader:
    """Adapter that reads ``benchmark_results*.json`` files.

    Each JSON file is an array of objects, one per compression strategy.
    Every object contains three quality axes (compression_fidelity,
    end_to_end, training_baseline) plus compression / streaming / timing
    metrics.
    """

    # ── IDataSourceReader protocol ──────────────────────────────

    @property
    def source_type(self) -> ResultSource:
        return ResultSource.BENCHMARK_JSON

    def supports(self, path: Path) -> bool:
        if not path.is_file():
            return False
        name = path.name.lower()
        return (
            name.startswith("benchmark_results")
            and name.endswith(".json")
        )

    def read(self, path: Path) -> List[MetricRecord]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            data = [data]

        version = self._detect_version(path)
        records: List[MetricRecord] = []

        for entry in data:
            record = self._parse_entry(entry, version, path)
            records.append(record)

        return records

    # ── Internal helpers ────────────────────────────────────────

    @staticmethod
    def _detect_version(path: Path) -> str:
        """Detect version tag from filename (e.g. ``_antes``)."""
        stem = path.stem.lower()
        if stem.endswith("_antes"):
            return "antes"
        return "current"

    @staticmethod
    def _make_id(config_name: str, version: str) -> str:
        raw = f"benchmark_json:{config_name}:{version}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    # ── Parsing ─────────────────────────────────────────────────

    def _parse_entry(self, entry: dict, version: str, path: Path) -> MetricRecord:
        config_name = entry.get("config_name", "unknown")
        record_id = self._make_id(config_name, version)

        # Quality axes
        quality_axes = {}
        per_frame_axes = {}

        for axis, key in [
            (ComparisonAxis.COMPRESSION_FIDELITY, "compression_fidelity"),
            (ComparisonAxis.END_TO_END, "end_to_end_quality"),
            (ComparisonAxis.TRAINING_BASELINE, "training_baseline"),
        ]:
            block = entry.get(key)
            if block:
                quality_axes[axis] = self._parse_quality(block)
                pf = self._parse_per_frame(block)
                if pf:
                    per_frame_axes[axis] = pf

        # Use end_to_end as primary quality if available, else fidelity
        primary_quality = quality_axes.get(
            ComparisonAxis.END_TO_END,
            quality_axes.get(ComparisonAxis.COMPRESSION_FIDELITY),
        )

        return MetricRecord(
            id=record_id,
            source=ResultSource.BENCHMARK_JSON,
            category=ResultCategory.COMPRESSION,
            name=config_name,
            scene=entry.get("scene", ""),
            tags={
                "version": version,
                "config_file": entry.get("config_file", ""),
                "source_path": str(path),
            },
            quality_axes=quality_axes,
            per_frame_axes=per_frame_axes,
            quality_metrics=primary_quality,
            compression_metrics=self._parse_compression(entry),
            streaming_metrics=self._parse_streaming(entry),
            timing_metrics=self._parse_timing(entry),
            model_info=self._parse_model_info(entry),
            pipeline_stats=self._parse_pipeline(entry),
        )

    # ── Sub-parsers ─────────────────────────────────────────────

    @staticmethod
    def _parse_quality(block: dict) -> QualityMetrics:
        return QualityMetrics(
            psnr=block.get("psnr_mean"),
            ssim=block.get("ssim_mean"),
            lpips_vgg=block.get("lpips_mean"),
            vmaf=block.get("vmaf"),
        )

    @staticmethod
    def _parse_per_frame(block: dict) -> PerFrameMetrics | None:
        psnr = block.get("psnr_per_frame")
        ssim = block.get("ssim_per_frame")
        lpips = block.get("lpips_per_frame")
        if not psnr:
            return None
        n = len(psnr)
        return PerFrameMetrics(
            frame_indices=list(range(n)),
            psnr=psnr,
            ssim=ssim,
            lpips=lpips,
        )

    @staticmethod
    def _parse_compression(entry: dict) -> CompressionMetrics:
        return CompressionMetrics(
            original_size_bytes=entry.get("original_size_bytes", 0),
            compressed_size_bytes=entry.get("compressed_size_bytes", 0),
            compression_ratio=entry.get("compression_ratio", 1.0),
            savings_pct=entry.get("savings_pct", 0.0),
            num_chunks=entry.get("num_chunks", 0),
        )

    @staticmethod
    def _parse_streaming(entry: dict) -> StreamingMetrics | None:
        qoe = entry.get("streaming_qoe")
        if not qoe:
            return None
        return StreamingMetrics(
            total_payload_bytes=qoe.get("total_payload_bytes", 0),
            bandwidth_mbps=qoe.get("bandwidth_mbps", 0.0),
            startup_delay_s=qoe.get("startup_delay_s", 0.0),
            rebuffer_events=qoe.get("rebuffer_events", 0),
            total_stall_duration_s=qoe.get("total_stall_duration_s", 0.0),
            e2e_latency_s=qoe.get("e2e_latency_s", 0.0),
            effective_throughput_mbps=qoe.get("effective_throughput_MBps", 0.0),
            qoe_score=qoe.get("qoe_score", 0.0),
            target_fps=qoe.get("target_fps", 30.0),
        )

    @staticmethod
    def _parse_timing(entry: dict) -> TimingMetrics:
        qoe = entry.get("streaming_qoe", {})
        return TimingMetrics(
            compress_time_s=entry.get("compress_time_s"),
            decompress_time_s=entry.get("decode_time_s"),
            export_time_per_frame_s=qoe.get("export_time_per_frame_s"),
        )

    @staticmethod
    def _parse_model_info(entry: dict) -> ModelInfo:
        return ModelInfo(
            num_gaussians_original=entry.get("num_gaussians_original"),
            num_gaussians_compressed=entry.get("num_gaussians_compressed"),
            sh_degree_original=entry.get("sh_degree_original"),
            sh_degree_compressed=entry.get("sh_degree_compressed"),
        )

    @staticmethod
    def _parse_pipeline(entry: dict) -> List[PipelineStageStats]:
        stages = entry.get("pipeline_stats", [])
        result = []
        for s in stages:
            result.append(PipelineStageStats(
                strategy=s.get("strategy", ""),
                ratio=s.get("ratio", 1.0),
                savings_pct=s.get("savings_pct", 0.0),
                compress_time_s=s.get("compress_time_s", 0.0),
                decompress_time_s=s.get("decompress_time_s", 0.0),
                extra=s.get("extra", {}),
            ))
        return result
