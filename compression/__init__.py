"""
4DGS Compression Module
=======================
Modular compression pipeline for 4D Gaussian Splatting models.

Pipeline: Dataset → 4DGS → Compression → Transmission → Decodification → Render

Usage:
    from compression import CompressionPipeline, GaussianData, DeformationData
    from compression.strategies import QuantizationStrategy, PruningStrategy, ...
"""

from compression.base import (
    GaussianData,
    DeformationData,
    CompressedPayload,
    CompressionStrategy,
    CompressionStats,
)
from compression.pipeline import CompressionPipeline
from compression.chunker import ModelChunker, ModelAssembler
from compression.serializer import ModelSerializer
from compression.strategies.lightgaussian_pruning import LightGaussianPruningStrategy

__all__ = [
    "GaussianData",
    "DeformationData",
    "CompressedPayload",
    "CompressionStrategy",
    "CompressionStats",
    "CompressionPipeline",
    "ModelChunker",
    "ModelAssembler",
    "ModelSerializer",
    "LightGaussianPruningStrategy",
]

__version__ = "1.0.0"
