"""Domain entities and value objects for the metrics viewer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .enums import ComparisonAxis, ResultCategory, ResultSource


# ── Value Objects ────────────────────────────────────────────────


@dataclass(frozen=True)
class QualityMetrics:
    """Mean image/video quality metrics."""
    psnr: Optional[float] = None        # dB
    ssim: Optional[float] = None
    lpips_vgg: Optional[float] = None
    lpips_alex: Optional[float] = None
    ms_ssim: Optional[float] = None
    d_ssim: Optional[float] = None
    vmaf: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "psnr": self.psnr,
            "ssim": self.ssim,
            "lpips_vgg": self.lpips_vgg,
            "lpips_alex": self.lpips_alex,
            "ms_ssim": self.ms_ssim,
            "d_ssim": self.d_ssim,
            "vmaf": self.vmaf,
        }

    def non_none_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.to_dict().items() if v is not None}


@dataclass(frozen=True)
class PerFrameMetrics:
    """Time-series quality metrics indexed by frame number."""
    frame_indices: List[int] = field(default_factory=list)
    psnr: Optional[List[float]] = None
    ssim: Optional[List[float]] = None
    lpips: Optional[List[float]] = None
    vmaf: Optional[List[float]] = None

    @property
    def num_frames(self) -> int:
        return len(self.frame_indices)


@dataclass(frozen=True)
class CompressionMetrics:
    """File-size and compression statistics."""
    original_size_bytes: int = 0
    compressed_size_bytes: int = 0
    compression_ratio: float = 1.0
    savings_pct: float = 0.0
    num_chunks: int = 0


@dataclass(frozen=True)
class StreamingMetrics:
    """Streaming quality-of-experience metrics."""
    total_payload_bytes: int = 0
    bandwidth_mbps: float = 0.0
    startup_delay_s: float = 0.0
    rebuffer_events: int = 0
    total_stall_duration_s: float = 0.0
    e2e_latency_s: float = 0.0
    effective_throughput_mbps: float = 0.0
    qoe_score: float = 0.0
    target_fps: float = 30.0


@dataclass(frozen=True)
class TimingMetrics:
    """Execution-time measurements."""
    compress_time_s: Optional[float] = None
    decompress_time_s: Optional[float] = None
    train_time_s: Optional[float] = None
    render_time_s: Optional[float] = None
    metrics_eval_time_s: Optional[float] = None
    export_time_per_frame_s: Optional[float] = None
    ply_export_time_s: Optional[float] = None


@dataclass(frozen=True)
class ModelInfo:
    """Gaussian model metadata."""
    num_gaussians_original: Optional[int] = None
    num_gaussians_compressed: Optional[int] = None
    sh_degree_original: Optional[int] = None
    sh_degree_compressed: Optional[int] = None
    iteration: Optional[int] = None
    num_ply_files: Optional[int] = None
    total_ply_size_bytes: Optional[int] = None


@dataclass(frozen=True)
class PipelineStageStats:
    """Statistics for one stage of a multi-stage compression pipeline."""
    strategy: str = ""
    ratio: float = 1.0
    savings_pct: float = 0.0
    compress_time_s: float = 0.0
    decompress_time_s: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


# ── Entity ───────────────────────────────────────────────────────


@dataclass
class MetricRecord:
    """
    Central domain entity: one complete benchmark / training result.

    For benchmark results, ``quality_axes`` maps each ComparisonAxis
    to its own QualityMetrics + PerFrameMetrics so all three viewpoints
    (compression_fidelity, end_to_end, training_baseline) are stored.
    """
    id: str
    source: ResultSource
    category: ResultCategory
    name: str                                         # e.g. "hexplane_downsample"
    scene: str = ""                                   # e.g. "coffee_martini"
    timestamp: Optional[datetime] = None
    tags: Dict[str, str] = field(default_factory=dict)  # free metadata

    # ── Quality metrics (multi-axis for benchmarks) ──
    quality_axes: Dict[ComparisonAxis, QualityMetrics] = field(default_factory=dict)
    per_frame_axes: Dict[ComparisonAxis, PerFrameMetrics] = field(default_factory=dict)

    # ── "Primary" quality shortcut (single-axis sources) ──
    quality_metrics: Optional[QualityMetrics] = None
    per_frame_metrics: Optional[PerFrameMetrics] = None

    # ── Additional metric groups (all optional) ──
    compression_metrics: Optional[CompressionMetrics] = None
    streaming_metrics: Optional[StreamingMetrics] = None
    timing_metrics: Optional[TimingMetrics] = None
    model_info: Optional[ModelInfo] = None
    pipeline_stats: List[PipelineStageStats] = field(default_factory=list)

    # ── Convenience helpers ──

    def get_quality(self, axis: Optional[ComparisonAxis] = None) -> Optional[QualityMetrics]:
        """Return quality metrics for a given axis, falling back to primary."""
        if axis and axis in self.quality_axes:
            return self.quality_axes[axis]
        return self.quality_metrics

    def get_per_frame(self, axis: Optional[ComparisonAxis] = None) -> Optional[PerFrameMetrics]:
        """Return per-frame metrics for a given axis, falling back to primary."""
        if axis and axis in self.per_frame_axes:
            return self.per_frame_axes[axis]
        return self.per_frame_metrics

    @property
    def display_name(self) -> str:
        """Human-readable label built from name + tags."""
        parts = [self.name]
        if "version" in self.tags:
            parts.append(f"({self.tags['version']})")
        if "iteration" in self.tags:
            parts.append(f"it{self.tags['iteration']}")
        return " ".join(parts)


# ── Comparison helpers ───────────────────────────────────────────


@dataclass
class ComparisonResult:
    """Table-like structure returned by MetricsService.compare()."""
    record_ids: List[str]
    record_names: List[str]
    metric_names: List[str]
    values: List[List[Optional[float]]]   # rows = records, cols = metrics

    def as_dicts(self) -> List[Dict[str, Any]]:
        """Return list of {metric: value} dicts, one per record."""
        return [
            {"name": name, **{m: v for m, v in zip(self.metric_names, row)}}
            for name, row in zip(self.record_names, self.values)
        ]


@dataclass
class FilterOptions:
    """Available filter choices for the UI."""
    sources: List[ResultSource] = field(default_factory=list)
    categories: List[ResultCategory] = field(default_factory=list)
    scenes: List[str] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    tags: Dict[str, List[str]] = field(default_factory=dict)
