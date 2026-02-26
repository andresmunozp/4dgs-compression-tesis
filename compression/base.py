"""
Base classes and data structures for the 4DGS compression system.

Defines the abstract CompressionStrategy interface and the data containers
(GaussianData, DeformationData, CompressedPayload) that flow through the pipeline.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class GaussianData:
    """Container for the per-Gaussian parameters (canonical frame).

    All arrays are NumPy float32 unless noted.
    Shapes use N = number of Gaussians.
    """

    xyz: np.ndarray              # (N, 3)
    features_dc: np.ndarray      # (N, 1, 3)  — DC SH coeff (after transpose)
    features_rest: np.ndarray    # (N, K, 3)  — higher-order SH (K = (deg+1)²-1)
    opacity: np.ndarray          # (N, 1)
    scaling: np.ndarray          # (N, 3)
    rotation: np.ndarray         # (N, 4)
    sh_degree: int = 3
    active_sh_degree: int = 3

    # Optional auxiliary data (may or may not be present)
    deformation_table: Optional[np.ndarray] = None   # (N,) bool
    deformation_accum: Optional[np.ndarray] = None   # (N, 3)

    # ---------- helpers ----------
    @property
    def num_gaussians(self) -> int:
        return self.xyz.shape[0]

    @property
    def total_bytes(self) -> int:
        """Uncompressed size in bytes (all arrays)."""
        total = 0
        for arr in (self.xyz, self.features_dc, self.features_rest,
                    self.opacity, self.scaling, self.rotation):
            total += arr.nbytes
        if self.deformation_table is not None:
            total += self.deformation_table.nbytes
        if self.deformation_accum is not None:
            total += self.deformation_accum.nbytes
        return total

    def validate(self) -> None:
        """Raise if shapes are inconsistent."""
        N = self.num_gaussians
        assert self.xyz.shape == (N, 3), f"xyz: expected ({N},3), got {self.xyz.shape}"
        assert self.features_dc.shape[0] == N, f"features_dc first dim mismatch"
        assert self.features_rest.shape[0] == N, f"features_rest first dim mismatch"
        assert self.opacity.shape == (N, 1), f"opacity: expected ({N},1), got {self.opacity.shape}"
        assert self.scaling.shape == (N, 3), f"scaling: expected ({N},3), got {self.scaling.shape}"
        assert self.rotation.shape == (N, 4), f"rotation: expected ({N},4), got {self.rotation.shape}"
        if self.deformation_table is not None:
            assert self.deformation_table.shape == (N,), \
                f"deformation_table: expected ({N},), got {self.deformation_table.shape}"
        if self.deformation_accum is not None:
            assert self.deformation_accum.shape == (N, 3), \
                f"deformation_accum: expected ({N},3), got {self.deformation_accum.shape}"

    def copy(self) -> "GaussianData":
        """Deep copy."""
        return GaussianData(
            xyz=self.xyz.copy(),
            features_dc=self.features_dc.copy(),
            features_rest=self.features_rest.copy(),
            opacity=self.opacity.copy(),
            scaling=self.scaling.copy(),
            rotation=self.rotation.copy(),
            sh_degree=self.sh_degree,
            active_sh_degree=self.active_sh_degree,
            deformation_table=self.deformation_table.copy() if self.deformation_table is not None else None,
            deformation_accum=self.deformation_accum.copy() if self.deformation_accum is not None else None,
        )


@dataclass
class DeformationData:
    """Container for the deformation network state."""

    state_dict: Dict[str, Any]  # torch state_dict (kept as-is for serialization)

    # Optional metadata about the network architecture (for validation)
    net_width: Optional[int] = None
    defor_depth: Optional[int] = None
    bounds: Optional[float] = None

    @property
    def total_bytes(self) -> int:
        """Approximate uncompressed size."""
        total = 0
        for v in self.state_dict.values():
            if hasattr(v, "nbytes"):
                total += v.nbytes
            elif hasattr(v, "nelement"):
                total += v.nelement() * v.element_size()
        return total


@dataclass
class CompressionStats:
    """Metrics collected during a single compress/decompress operation."""

    strategy_name: str
    original_bytes: int = 0
    compressed_bytes: int = 0
    compression_time_s: float = 0.0
    decompression_time_s: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def ratio(self) -> float:
        if self.compressed_bytes == 0:
            return 0.0
        return self.original_bytes / self.compressed_bytes

    @property
    def savings_pct(self) -> float:
        if self.original_bytes == 0:
            return 0.0
        return (1 - self.compressed_bytes / self.original_bytes) * 100


@dataclass
class CompressedPayload:
    """Output of a compression pipeline: binary data + rich metadata."""

    data: bytes
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Automatically computed
    checksum: str = ""

    def compute_checksum(self) -> str:
        self.checksum = hashlib.sha256(self.data).hexdigest()
        return self.checksum

    def verify_checksum(self) -> bool:
        return hashlib.sha256(self.data).hexdigest() == self.checksum

    @property
    def size_bytes(self) -> int:
        return len(self.data)


# ---------------------------------------------------------------------------
# Abstract strategy
# ---------------------------------------------------------------------------

class CompressionStrategy(ABC):
    """Abstract base for all compression strategies.

    Subclasses must implement compress_gaussian / decompress_gaussian.
    Optionally override compress_deformation / decompress_deformation
    (default is pass-through).
    """

    def __init__(self, **kwargs):
        self._params = kwargs
        self._stats = CompressionStats(strategy_name=self.name)

    # ------ identity ------
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        ...

    @property
    def params(self) -> Dict[str, Any]:
        return dict(self._params)

    @property
    def stats(self) -> CompressionStats:
        return self._stats

    # ------ Gaussian data ------
    @abstractmethod
    def compress_gaussian(self, data: GaussianData) -> GaussianData:
        """Compress / transform Gaussian data (in-place or copy)."""
        ...

    @abstractmethod
    def decompress_gaussian(self, data: GaussianData, metadata: Dict[str, Any]) -> GaussianData:
        """Reverse the Gaussian compression using stored metadata."""
        ...

    # ------ Deformation data (optional, default = passthrough) ------
    def compress_deformation(self, data: DeformationData) -> DeformationData:
        return data

    def decompress_deformation(self, data: DeformationData, metadata: Dict[str, Any]) -> DeformationData:
        return data

    # ------ Metadata produced by this strategy ------
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata that must be stored alongside the compressed data."""
        return {"strategy": self.name, "params": self.params}
