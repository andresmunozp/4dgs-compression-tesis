"""Integration tests for the metrics viewer.

Covers: readers, services, exporters, and page construction.
Run with:  python -m pytest viewers/metrics_viewer/tests/ -v
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

# ── Domain imports ──────────────────────────────────────────────
from viewers.metrics_viewer.domain.enums import (
    ComparisonAxis, MetricType, ResultCategory, ResultSource,
)
from viewers.metrics_viewer.domain.models import (
    CompressionMetrics, MetricRecord, PerFrameMetrics, PipelineStageStats,
    QualityMetrics, TimingMetrics, ModelInfo,
)
from viewers.metrics_viewer.domain.ports import ExportConfig
from viewers.metrics_viewer.domain.services import MetricsService, _extract_metric

# ── Reader imports ──────────────────────────────────────────────
from viewers.metrics_viewer.adapters.readers.benchmark_json_reader import BenchmarkJsonReader
from viewers.metrics_viewer.adapters.readers.benchmark_csv_reader import BenchmarkCsvReader
from viewers.metrics_viewer.adapters.readers.training_json_reader import TrainingJsonReader

# ── Exporter imports ────────────────────────────────────────────
from viewers.metrics_viewer.adapters.exporters.csv_exporter import CsvExporter
from viewers.metrics_viewer.adapters.exporters.png_exporter import PngExporter


# ═══════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def sample_benchmark_json(tmp_path: Path) -> Path:
    """Create a minimal benchmark_results.json matching actual reader format.
    
    The reader expects a list of flat entries with keys like
    psnr_mean, compression_fidelity.psnr_mean, etc.
    """
    data = [{
        "config_name": "test_strategy",
        "scene": "test_scene",
        "compression_fidelity": {
            "psnr_mean": 30.5,
            "ssim_mean": 0.95,
            "lpips_mean": 0.05,
            "psnr_per_frame": [30.0, 31.0, 30.5],
            "ssim_per_frame": [0.94, 0.96, 0.95],
        },
        "end_to_end_quality": {
            "psnr_mean": 29.0,
            "ssim_mean": 0.90,
            "lpips_mean": 0.08,
        },
        "training_baseline": {
            "psnr_mean": 31.0,
            "ssim_mean": 0.97,
        },
        "original_size_bytes": 100000000,
        "compressed_size_bytes": 20000000,
        "compression_ratio": 5.0,
        "savings_pct": 80.0,
        "compress_time_s": 1.5,
        "decode_time_s": 0.8,
        "num_gaussians_original": 100000,
        "num_gaussians_compressed": 50000,
        "sh_degree_original": 3,
        "pipeline_stats": [
            {"strategy": "quantize", "ratio": 2.0, "savings_pct": 50.0,
             "compress_time_s": 0.5, "decompress_time_s": 0.3},
            {"strategy": "huffman", "ratio": 2.5, "savings_pct": 30.0,
             "compress_time_s": 1.0, "decompress_time_s": 0.5},
        ],
    }]
    results_dir = tmp_path / "benchmark_results"
    results_dir.mkdir()
    fp = results_dir / "benchmark_results.json"
    fp.write_text(json.dumps(data))
    return fp


@pytest.fixture
def sample_training_json(tmp_path: Path) -> Path:
    """Create a minimal training result JSON matching actual reader format."""
    data = {
        "scene": "test_scene",
        "expname": "test_exp",
        "metrics_full": {
            "ours_7000": {
                "PSNR": 28.5,
                "SSIM": 0.88,
                "LPIPS-vgg": 0.12,
            }
        },
    }
    results_dir = tmp_path / "results_json"
    results_dir.mkdir()
    fp = results_dir / "4dgs_test_it7000.json"
    fp.write_text(json.dumps(data))
    return fp


@pytest.fixture
def sample_records() -> list[MetricRecord]:
    """Build a handful of MetricRecords in-memory for service tests."""
    records = []
    for i in range(3):
        r = MetricRecord(
            id=f"test-{i}",
            source=ResultSource.BENCHMARK_JSON,
            category=ResultCategory.COMPRESSION,
            name=f"strategy_{i}",
            scene="coffee_martini",
            quality_axes={
                ComparisonAxis.END_TO_END: QualityMetrics(
                    psnr=28.0 + i, ssim=0.90 + i * 0.01, lpips_vgg=0.10 - i * 0.01,
                ),
            },
            compression_metrics=CompressionMetrics(
                original_size_bytes=100_000_000,
                compressed_size_bytes=100_000_000 // (i + 2),
                compression_ratio=float(i + 2),
                savings_pct=100.0 * (1 - 1 / (i + 2)),
            ),
            timing_metrics=TimingMetrics(
                compress_time_s=1.0 + i * 0.5,
                decompress_time_s=0.5 + i * 0.2,
            ),
            pipeline_stats=[
                PipelineStageStats(strategy="quant", ratio=1.5, savings_pct=33.3,
                                   compress_time_s=0.5, decompress_time_s=0.2),
            ],
        )
        records.append(r)
    return records


# ═══════════════════════════════════════════════════════════════
#  READER TESTS
# ═══════════════════════════════════════════════════════════════


class TestBenchmarkJsonReader:
    def test_supports_benchmark_json(self, sample_benchmark_json: Path):
        reader = BenchmarkJsonReader()
        assert reader.supports(sample_benchmark_json)

    def test_read_produces_records(self, sample_benchmark_json: Path):
        reader = BenchmarkJsonReader()
        records = reader.read(sample_benchmark_json)
        assert len(records) >= 1

    def test_record_has_quality_axes(self, sample_benchmark_json: Path):
        reader = BenchmarkJsonReader()
        records = reader.read(sample_benchmark_json)
        r = records[0]
        assert ComparisonAxis.COMPRESSION_FIDELITY in r.quality_axes
        q = r.get_quality(ComparisonAxis.COMPRESSION_FIDELITY)
        assert q is not None
        assert q.psnr == pytest.approx(30.5, abs=0.1)

    def test_record_has_compression_metrics(self, sample_benchmark_json: Path):
        reader = BenchmarkJsonReader()
        records = reader.read(sample_benchmark_json)
        r = records[0]
        assert r.compression_metrics is not None
        assert r.compression_metrics.compression_ratio == pytest.approx(5.0)

    def test_record_has_pipeline_stats(self, sample_benchmark_json: Path):
        reader = BenchmarkJsonReader()
        records = reader.read(sample_benchmark_json)
        r = records[0]
        assert len(r.pipeline_stats) == 2
        assert r.pipeline_stats[0].strategy == "quantize"

    def test_source_type(self):
        reader = BenchmarkJsonReader()
        assert reader.source_type == ResultSource.BENCHMARK_JSON


class TestTrainingJsonReader:
    def test_supports_training_json(self, sample_training_json: Path):
        reader = TrainingJsonReader()
        assert reader.supports(sample_training_json)

    def test_read_produces_records(self, sample_training_json: Path):
        reader = TrainingJsonReader()
        records = reader.read(sample_training_json)
        assert len(records) >= 1

    def test_record_quality(self, sample_training_json: Path):
        reader = TrainingJsonReader()
        records = reader.read(sample_training_json)
        r = records[0]
        q = r.get_quality()
        assert q is not None
        assert q.psnr == pytest.approx(28.5, abs=0.1)


# ═══════════════════════════════════════════════════════════════
#  SERVICE TESTS
# ═══════════════════════════════════════════════════════════════


class TestMetricsService:
    def test_get_all_records(self, sample_records):
        svc = MetricsService(readers=[])
        svc._records = {r.id: r for r in sample_records}
        assert len(svc.get_all_records()) == 3

    def test_get_record_by_id(self, sample_records):
        svc = MetricsService(readers=[])
        svc._records = {r.id: r for r in sample_records}
        r = svc.get_record("test-1")
        assert r is not None
        assert r.name == "strategy_1"

    def test_search_by_scene(self, sample_records):
        svc = MetricsService(readers=[])
        svc._records = {r.id: r for r in sample_records}
        found = svc.search_records(scene="coffee_martini")
        assert len(found) == 3

    def test_search_by_name(self, sample_records):
        svc = MetricsService(readers=[])
        svc._records = {r.id: r for r in sample_records}
        found = svc.search_records(name="strategy_0")
        assert len(found) == 1

    def test_compare_records(self, sample_records):
        svc = MetricsService(readers=[])
        svc._records = {r.id: r for r in sample_records}
        result = svc.compare(
            ["test-0", "test-2"],
            [MetricType.PSNR, MetricType.SSIM],
            axis=ComparisonAxis.END_TO_END,
        )
        assert len(result.record_ids) == 2
        assert len(result.metric_names) == 2
        # test-2 should have higher PSNR
        assert result.values[1][0] > result.values[0][0]

    def test_get_available_filters(self, sample_records):
        svc = MetricsService(readers=[])
        svc._records = {r.id: r for r in sample_records}
        f = svc.get_available_filters()
        assert "coffee_martini" in f.scenes
        assert ResultSource.BENCHMARK_JSON in f.sources

    def test_extract_metric_compression_ratio(self, sample_records):
        r = sample_records[2]
        val = _extract_metric(r, MetricType.COMPRESSION_RATIO)
        assert val == pytest.approx(4.0)


# ═══════════════════════════════════════════════════════════════
#  EXPORTER TESTS
# ═══════════════════════════════════════════════════════════════


class TestCsvExporter:
    def test_supported_formats(self):
        exporter = CsvExporter()
        assert "csv" in exporter.supported_formats

    def test_export_produces_bytes(self, sample_records):
        exporter = CsvExporter()
        config = ExportConfig(format="csv", title="Test")
        data = exporter.export(sample_records, config)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_export_csv_has_header(self, sample_records):
        exporter = CsvExporter()
        config = ExportConfig(format="csv")
        data = exporter.export(sample_records, config)
        text = data.decode("utf-8")
        assert "id" in text
        assert "name" in text
        # Quality columns should be present when records have quality_axes
        assert "psnr" in text

    def test_export_csv_row_count(self, sample_records):
        exporter = CsvExporter()
        config = ExportConfig(format="csv")
        data = exporter.export(sample_records, config)
        lines = data.decode("utf-8").strip().split("\n")
        # 1 header + 3 data rows
        assert len(lines) == 4


class TestPngExporter:
    def test_supported_formats(self):
        exporter = PngExporter()
        assert "png" in exporter.supported_formats

    def test_export_produces_bytes(self, sample_records):
        exporter = PngExporter()
        config = ExportConfig(format="png", title="Test", width=800, height=600)
        data = exporter.export(sample_records, config)
        assert isinstance(data, bytes)
        assert len(data) > 0


# ═══════════════════════════════════════════════════════════════
#  DOMAIN MODEL TESTS
# ═══════════════════════════════════════════════════════════════


class TestMetricRecord:
    def test_display_name(self):
        r = MetricRecord(
            id="x", source=ResultSource.BENCHMARK_JSON,
            category=ResultCategory.COMPRESSION,
            name="strategy_a", tags={"version": "v2", "iteration": "1000"},
        )
        assert "strategy_a" in r.display_name
        assert "v2" in r.display_name

    def test_get_quality_fallback(self):
        q = QualityMetrics(psnr=30.0)
        r = MetricRecord(
            id="y", source=ResultSource.TRAINING_JSON,
            category=ResultCategory.TRAINING,
            name="train_1", quality_metrics=q,
        )
        assert r.get_quality() == q
        assert r.get_quality(ComparisonAxis.END_TO_END) == q

    def test_get_quality_axis(self):
        q1 = QualityMetrics(psnr=30.0)
        q2 = QualityMetrics(psnr=25.0)
        r = MetricRecord(
            id="z", source=ResultSource.BENCHMARK_JSON,
            category=ResultCategory.COMPRESSION,
            name="test",
            quality_metrics=q1,
            quality_axes={ComparisonAxis.END_TO_END: q2},
        )
        assert r.get_quality(ComparisonAxis.END_TO_END).psnr == 25.0
        assert r.get_quality().psnr == 30.0


class TestQualityMetrics:
    def test_to_dict(self):
        q = QualityMetrics(psnr=30.0, ssim=0.95)
        d = q.to_dict()
        assert d["psnr"] == 30.0
        assert d["ssim"] == 0.95
        assert d["vmaf"] is None

    def test_non_none_dict(self):
        q = QualityMetrics(psnr=30.0, ssim=0.95)
        d = q.non_none_dict()
        assert "vmaf" not in d
        assert len(d) == 2
