"""Enumerations for the metrics viewer domain."""

from enum import Enum


class ResultSource(Enum):
    """Where the metric data was loaded from."""
    BENCHMARK_JSON = "benchmark_json"
    BENCHMARK_CSV = "benchmark_csv"
    TRAINING_JSON = "training_json"
    VMAF_JSON = "vmaf_json"


class ResultCategory(Enum):
    """High-level category of the result."""
    COMPRESSION = "compression"
    DECOMPRESSION = "decompression"
    TRAINING = "training"
    END_TO_END = "end_to_end"


class MetricType(Enum):
    """Individual metric identifiers."""
    PSNR = "psnr"
    SSIM = "ssim"
    LPIPS_VGG = "lpips_vgg"
    LPIPS_ALEX = "lpips_alex"
    MS_SSIM = "ms_ssim"
    D_SSIM = "d_ssim"
    VMAF = "vmaf"
    COMPRESSION_RATIO = "compression_ratio"
    SAVINGS_PCT = "savings_pct"
    QOE_SCORE = "qoe_score"
    COMPRESS_TIME = "compress_time"
    DECOMPRESS_TIME = "decompress_time"
    TRAIN_TIME = "train_time"
    RENDER_TIME = "render_time"


class ComparisonAxis(Enum):
    """The three quality-comparison axes used in benchmarks."""
    COMPRESSION_FIDELITY = "compression_fidelity"   # decompressed vs original model
    END_TO_END = "end_to_end"                       # decompressed vs ground truth
    TRAINING_BASELINE = "training_baseline"          # original model vs ground truth
