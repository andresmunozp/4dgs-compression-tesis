"""
Pruning strategy — remove Gaussians with low impact.

Supports multiple pruning criteria (opacity, deformation accumulation,
spatial redundancy, max count) and produces an **index_map** that maps
original Gaussian indices → pruned indices (or -1 if removed) for
strict validation and traceability.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from compression.base import CompressionStrategy, DeformationData, GaussianData


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


class PruningStrategy(CompressionStrategy):
    """Remove low-impact Gaussians from the model.

    Parameters
    ----------
    opacity_threshold : float
        Pruning threshold on **activated** (sigmoid) opacity.  Gaussians
        whose sigmoid(opacity_raw) < threshold are removed.
    deformation_threshold : float or None
        If set, Gaussians whose mean deformation_accum magnitude is below
        this value are pruned (low contribution to dynamics).
    redundancy_radius : float or None
        If set, for each pair of Gaussians closer than this distance, the
        one with lower opacity is removed.
    max_gaussians : int or None
        Hard cap on number of Gaussians to keep (sorted by opacity,
        highest first).
    """

    def __init__(
        self,
        opacity_threshold: float = 0.005,
        deformation_threshold: Optional[float] = None,
        redundancy_radius: Optional[float] = None,
        max_gaussians: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            opacity_threshold=opacity_threshold,
            deformation_threshold=deformation_threshold,
            redundancy_radius=redundancy_radius,
            max_gaussians=max_gaussians,
            **kwargs,
        )
        self.opacity_threshold = opacity_threshold
        self.deformation_threshold = deformation_threshold
        self.redundancy_radius = redundancy_radius
        self.max_gaussians = max_gaussians

        self._index_map: Optional[np.ndarray] = None  # original→pruned
        self._original_count: int = 0

    @property
    def name(self) -> str:
        return "pruning"

    # ── Build the keep-mask ──────────────────────────────────────────

    def _build_mask(self, data: GaussianData) -> np.ndarray:
        """Return boolean mask (True = keep)."""
        N = data.num_gaussians
        keep = np.ones(N, dtype=bool)

        # 1. Opacity threshold
        activated_opacity = _sigmoid(data.opacity.squeeze(-1))
        keep &= activated_opacity >= self.opacity_threshold

        # 2. Deformation contribution threshold
        if self.deformation_threshold is not None and data.deformation_accum is not None:
            deform_mag = np.linalg.norm(data.deformation_accum, axis=-1)
            keep &= deform_mag >= self.deformation_threshold

        # 3. Spatial redundancy (greedy)
        if self.redundancy_radius is not None and self.redundancy_radius > 0:
            from scipy.spatial import cKDTree

            pts = data.xyz[keep]
            local_indices = np.where(keep)[0]
            tree = cKDTree(pts)
            pairs = tree.query_pairs(r=self.redundancy_radius)
            # Remove the lower-opacity member of each pair
            opac = activated_opacity
            for i, j in pairs:
                gi, gj = local_indices[i], local_indices[j]
                if keep[gi] and keep[gj]:
                    if opac[gi] < opac[gj]:
                        keep[gi] = False
                    else:
                        keep[gj] = False

        # 4. Hard cap
        if self.max_gaussians is not None and keep.sum() > self.max_gaussians:
            kept_indices = np.where(keep)[0]
            opac_kept = activated_opacity[kept_indices]
            sorted_idx = np.argsort(-opac_kept)  # highest opacity first
            # Mark excess as not-kept
            excess = kept_indices[sorted_idx[self.max_gaussians :]]
            keep[excess] = False

        return keep

    # ── Build index map ──────────────────────────────────────────────

    @staticmethod
    def _build_index_map(keep_mask: np.ndarray) -> np.ndarray:
        """Create mapping  original_idx → new_idx  (or -1 if pruned).

        Returns int32 array of shape (N_original,).
        """
        index_map = np.full(len(keep_mask), -1, dtype=np.int32)
        new_idx = 0
        for orig_idx, kept in enumerate(keep_mask):
            if kept:
                index_map[orig_idx] = new_idx
                new_idx += 1
        return index_map

    # ── Compress ─────────────────────────────────────────────────────

    def compress_gaussian(self, data: GaussianData) -> GaussianData:
        self._original_count = data.num_gaussians
        keep = self._build_mask(data)
        self._index_map = self._build_index_map(keep)

        n_kept = int(keep.sum())
        n_pruned = self._original_count - n_kept

        # Validate before pruning
        assert n_kept > 0, "Pruning would remove ALL Gaussians — aborting"

        data.xyz = data.xyz[keep]
        data.features_dc = data.features_dc[keep]
        data.features_rest = data.features_rest[keep]
        data.opacity = data.opacity[keep]
        data.scaling = data.scaling[keep]
        data.rotation = data.rotation[keep]

        if data.deformation_table is not None:
            data.deformation_table = data.deformation_table[keep]
        if data.deformation_accum is not None:
            data.deformation_accum = data.deformation_accum[keep]

        # Strict validation: shapes must be consistent
        data.validate()

        # Cross-check index_map
        assert (self._index_map >= 0).sum() == n_kept, \
            f"index_map inconsistency: {(self._index_map >= 0).sum()} vs {n_kept}"

        self._stats.extra = {
            "original_gaussians": self._original_count,
            "kept_gaussians": n_kept,
            "pruned_gaussians": n_pruned,
            "pruned_pct": round(n_pruned / self._original_count * 100, 2),
        }

        return data

    def decompress_gaussian(
        self, data: GaussianData, metadata: Dict[str, Any]
    ) -> GaussianData:
        """Pruning is lossy — decompression is a no-op (data stays pruned)."""
        return data

    # ── Metadata ─────────────────────────────────────────────────────

    def get_metadata(self) -> Dict[str, Any]:
        meta = {
            "strategy": self.name,
            "params": self.params,
            "original_count": self._original_count,
            "kept_count": int((self._index_map >= 0).sum()) if self._index_map is not None else 0,
        }
        if self._index_map is not None:
            meta["index_map"] = self._index_map.tolist()
        return meta

    # ── Accessors ────────────────────────────────────────────────────

    @property
    def index_map(self) -> Optional[np.ndarray]:
        """``original_idx → new_idx`` or ``-1`` if pruned."""
        return self._index_map

    def validate_index_map(self, original_count: int, pruned_count: int) -> bool:
        """Verify the index_map is consistent with the before/after counts."""
        if self._index_map is None:
            return False
        if len(self._index_map) != original_count:
            return False
        kept = (self._index_map >= 0).sum()
        if kept != pruned_count:
            return False
        # New indices should be contiguous 0..kept-1
        new_indices = self._index_map[self._index_map >= 0]
        expected = np.arange(kept, dtype=np.int32)
        return np.array_equal(new_indices, expected)
