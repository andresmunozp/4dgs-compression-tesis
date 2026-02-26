"""
Spherical harmonics reduction strategy.

Truncates higher-order SH coefficients to reduce per-Gaussian storage.
SH degree 3 → 2, 1, or 0 eliminates the dominant storage cost
(features_rest is ~76 % of per-Gaussian data at degree 3).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from compression.base import CompressionStrategy, GaussianData

# SH coefficients count per degree level:
#   degree 0 →  1 band  →  1 coeff   (DC only, stored in features_dc)
#   degree 1 →  4 bands →  3 extra
#   degree 2 →  9 bands →  8 extra   (5 new)
#   degree 3 → 16 bands → 15 extra   (7 new)
# features_rest shape = (N, (deg+1)²-1, 3)


def _sh_rest_count(degree: int) -> int:
    """Number of extra SH coefficients for a given max degree."""
    return (degree + 1) ** 2 - 1


class SHReductionStrategy(CompressionStrategy):
    """Truncate spherical harmonics to a lower degree.

    Parameters
    ----------
    target_sh_degree : int
        Desired SH degree (0, 1, or 2).  Must be < source degree.
    """

    def __init__(self, target_sh_degree: int = 1, **kwargs):
        super().__init__(target_sh_degree=target_sh_degree, **kwargs)
        self.target_sh_degree = target_sh_degree
        self._original_sh_degree: int = 3

    @property
    def name(self) -> str:
        return "sh_reduction"

    def compress_gaussian(self, data: GaussianData) -> GaussianData:
        self._original_sh_degree = data.sh_degree

        if self.target_sh_degree >= data.sh_degree:
            # Nothing to do
            return data

        target_rest = _sh_rest_count(self.target_sh_degree)
        orig_rest = _sh_rest_count(data.sh_degree)

        if self.target_sh_degree == 0:
            # Drop all higher-order SH
            data.features_rest = np.zeros(
                (data.num_gaussians, 0, 3), dtype=data.features_rest.dtype
            )
        else:
            # Keep only the first target_rest coefficients
            data.features_rest = data.features_rest[:, :target_rest, :].copy()

        data.sh_degree = self.target_sh_degree
        data.active_sh_degree = self.target_sh_degree

        self._stats.extra = {
            "original_sh_degree": self._original_sh_degree,
            "target_sh_degree": self.target_sh_degree,
            "original_rest_coeffs": orig_rest,
            "target_rest_coeffs": target_rest,
            "per_gaussian_savings_bytes": (orig_rest - target_rest) * 3 * 4,
        }

        return data

    def decompress_gaussian(
        self, data: GaussianData, metadata: Dict[str, Any]
    ) -> GaussianData:
        """SH reduction is lossy — decompression is a no-op."""
        return data

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "strategy": self.name,
            "params": self.params,
            "original_sh_degree": self._original_sh_degree,
        }
