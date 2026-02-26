"""Domain layer public API."""

from .enums import ComparisonAxis, MetricType, ResultCategory, ResultSource
from .models import (
    ComparisonResult,
    CompressionMetrics,
    FilterOptions,
    MetricRecord,
    ModelInfo,
    PerFrameMetrics,
    PipelineStageStats,
    QualityMetrics,
    StreamingMetrics,
    TimingMetrics,
)
from .ports import ExportConfig, IDataSourceReader, IMetricsService, IResultExporter

__all__ = [
    # Enums
    "ComparisonAxis",
    "MetricType",
    "ResultCategory",
    "ResultSource",
    # Models
    "ComparisonResult",
    "CompressionMetrics",
    "FilterOptions",
    "MetricRecord",
    "ModelInfo",
    "PerFrameMetrics",
    "PipelineStageStats",
    "QualityMetrics",
    "StreamingMetrics",
    "TimingMetrics",
    # Ports
    "ExportConfig",
    "IDataSourceReader",
    "IMetricsService",
    "IResultExporter",
]
