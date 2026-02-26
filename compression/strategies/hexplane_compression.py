"""
HexPlane grid compression — safe baseline + experimental methods.

The 4DGS deformation network stores a HexPlane (six 2-D factorised grids
per resolution scale).  These grids dominate the deformation state-dict
(~9 MB at default config).  This strategy compresses them.

Methods
-------
``quantize`` (baseline, safe)
    Cast grid parameters from float32 → float16.  Negligible quality loss
    and guaranteed 50 % reduction.

``svd`` (experimental)
    Truncated SVD per grid plane to rank *r*.  Stores U, S, Vt factors
    instead of the full grid.  Aggressive compression but may degrade
    deformation quality for low ranks.

``downsample`` (experimental)
    Spatially downsample grid planes by a given factor.  At decompression
    the grid is upsampled back via bilinear interpolation.  Only applied
    to spatial dimensions; temporal dimension is left unchanged.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

import numpy as np

from compression.base import CompressionStrategy, DeformationData, GaussianData

# Keys that match HexPlane grid parameters in the state_dict
_GRID_KEY_PATTERN = "deformation_net.grid.grids"

# In a 4D HexPlane (x, y, z, t), itertools.combinations(range(4), 2) yields:
#   idx 0 → (0,1) XY   idx 1 → (0,2) XZ   idx 2 → (0,3) XT
#   idx 3 → (1,2) YZ   idx 4 → (1,3) YT   idx 5 → (2,3) ZT
# Planes containing the time axis (dim 3) are at indices 2, 4, 5.
_TEMPORAL_PLANE_INDICES = {2, 4, 5}


def _is_grid_key(key: str) -> bool:
    return _GRID_KEY_PATTERN in key


def _is_temporal_plane(key: str) -> bool:
    """Return True if *key* points to a temporal grid plane (XT, YT, or ZT).

    Keys look like ``deformation_net.grid.grids.<scale>.<plane_idx>``.
    Plane indices 2, 4, 5 contain the time dimension.
    """
    import re
    m = re.search(r"grids\.(\d+)\.(\d+)$", key)
    if m is None:
        return False
    plane_idx = int(m.group(2))
    return plane_idx in _TEMPORAL_PLANE_INDICES


# ── Per-method helpers ────────────────────────────────────────────────────

def _quantize_grids(state_dict: dict) -> dict:
    """Cast all grid tensors to float16 (safe baseline)."""
    import torch

    new_sd = {}
    for k, v in state_dict.items():
        if _is_grid_key(k) and isinstance(v, torch.Tensor) and v.dtype == torch.float32:
            new_sd[k] = v.half()
        else:
            new_sd[k] = v
    return new_sd


def _dequantize_grids(state_dict: dict) -> dict:
    """Cast grid tensors back to float32."""
    import torch

    new_sd = {}
    for k, v in state_dict.items():
        if _is_grid_key(k) and isinstance(v, torch.Tensor) and v.dtype == torch.float16:
            new_sd[k] = v.float()
        else:
            new_sd[k] = v
    return new_sd


def _svd_compress_grids(
    state_dict: dict,
    rank: int,
    temporal_rank_multiplier: float = 4.0,
    energy_threshold: float = 0.999,
) -> tuple:
    """Truncated SVD on each grid tensor  [1, C, H, W] → U, S, Vt factors.

    Temporal planes (XT, YT, ZT) use a higher rank to preserve the motion
    signal.  An energy-based threshold ensures we keep enough singular
    values to reconstruct the plane faithfully.

    Parameters
    ----------
    rank : int
        Base rank for spatial-only planes.
    temporal_rank_multiplier : float
        Multiplier applied to *rank* for temporal planes (default 4×).
    energy_threshold : float
        Minimum fraction of Frobenius-norm energy to retain (0–1).
        The effective rank is the *maximum* of the requested rank and
        the energy-based rank, capped by min(H, W).

    Returns (new_state_dict, svd_info).
    """
    import torch

    new_sd = {}
    svd_info: Dict[str, Dict[str, Any]] = {}

    for k, v in state_dict.items():
        if _is_grid_key(k) and isinstance(v, torch.Tensor) and v.ndim == 4:
            # v shape: [1, C, H, W]
            C, H, W = v.shape[1], v.shape[2], v.shape[3]
            is_temporal = _is_temporal_plane(k)

            # Higher base rank for temporal planes so motion is preserved
            base_rank = int(rank * temporal_rank_multiplier) if is_temporal else rank

            # Reshape to [C, H, W], process each channel independently
            mat = v.squeeze(0).float()  # (C, H, W)
            U_list, S_list, Vt_list = [], [], []

            # Determine energy-aware rank across all channels
            channel_ranks = []
            all_svd = []
            for c in range(C):
                U, S, Vt = torch.linalg.svd(mat[c], full_matrices=False)
                all_svd.append((U, S, Vt))
                # Energy-based rank: cumulative energy ≥ threshold
                energy = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
                energy_rank = int((energy < energy_threshold).sum().item()) + 1
                channel_ranks.append(energy_rank)

            # Use the max of base_rank and median energy rank, capped by min(H,W)
            median_energy_rank = sorted(channel_ranks)[len(channel_ranks) // 2]
            effective_rank = min(max(base_rank, median_energy_rank), H, W)

            for c in range(C):
                U, S, Vt = all_svd[c]
                U_list.append(U[:, :effective_rank])
                S_list.append(S[:effective_rank])
                Vt_list.append(Vt[:effective_rank, :])

            # Store as stacked tensors — keep float32 to avoid precision
            # loss during U @ diag(S) @ Vt reconstruction
            U_all = torch.stack(U_list)    # (C, H, rank)
            S_all = torch.stack(S_list)    # (C, rank)
            Vt_all = torch.stack(Vt_list)  # (C, rank, W)

            new_sd[k + ".__svd_U"] = U_all     # float32
            new_sd[k + ".__svd_S"] = S_all     # float32
            new_sd[k + ".__svd_Vt"] = Vt_all   # float32

            svd_info[k] = {
                "original_shape": list(v.shape),
                "rank": effective_rank,
                "is_temporal": is_temporal,
            }
        else:
            new_sd[k] = v

    return new_sd, svd_info


def _svd_decompress_grids(state_dict: dict, svd_info: dict) -> dict:
    """Reconstruct grids from U, S, Vt factors."""
    import torch

    new_sd = {}
    reconstructed = set()

    for k in svd_info:
        info = svd_info[k]
        U = state_dict[k + ".__svd_U"].float()
        S = state_dict[k + ".__svd_S"].float()
        Vt = state_dict[k + ".__svd_Vt"].float()
        C = U.shape[0]

        mats = []
        for c in range(C):
            mats.append(U[c] @ torch.diag(S[c]) @ Vt[c])
        grid = torch.stack(mats).unsqueeze(0)  # (1, C, H, W)
        new_sd[k] = grid
        reconstructed.add(k + ".__svd_U")
        reconstructed.add(k + ".__svd_S")
        reconstructed.add(k + ".__svd_Vt")

    for k, v in state_dict.items():
        if k not in new_sd and k not in reconstructed:
            new_sd[k] = v

    return new_sd


def _downsample_grids(state_dict: dict, factor: float) -> tuple:
    """Spatially downsample grid planes by *factor*.

    **Spatial-only planes (XY, XZ, YZ)** are *not* downsampled.  They
    contain high-frequency learned features and the MLP is very sensitive
    to even small perturbations in these grids; a 2× bilinear down/up
    cycle introduces ~30-50 % relative error which pushes MLP inputs
    out-of-distribution and collapses deformations to near-zero.

    **Temporal planes (XT, YT, ZT):** only the spatial axis (W) is
    downsampled; the temporal axis (H = reso_t) is kept intact so that
    the motion signal is preserved.  Temporal planes are smooth
    (initialised at 1.0, small trained deltas) and tolerate downsampling
    well.
    """
    import torch
    import torch.nn.functional as F

    new_sd = {}
    ds_info: Dict[str, Dict[str, Any]] = {}

    for k, v in state_dict.items():
        if _is_grid_key(k) and isinstance(v, torch.Tensor) and v.ndim == 4:
            original_shape = list(v.shape)
            _, C, H, W = v.shape
            is_temporal = _is_temporal_plane(k)

            if is_temporal:
                # Only downsample W (spatial); keep H (temporal) intact
                new_H = H
                new_W = max(1, int(W / factor))

                downsampled = F.interpolate(
                    v.float(), size=(new_H, new_W), mode="bilinear", align_corners=True
                )
                new_sd[k] = downsampled.half()
                ds_info[k] = {
                    "original_shape": original_shape,
                    "is_temporal": is_temporal,
                }
            else:
                # Spatial-only plane — keep untouched (only quantise to fp16)
                new_sd[k] = v.half() if v.dtype == torch.float32 else v
                ds_info[k] = {
                    "original_shape": original_shape,
                    "is_temporal": False,
                }
        else:
            new_sd[k] = v

    return new_sd, ds_info


def _upsample_grids(state_dict: dict, ds_info: dict) -> dict:
    """Restore grids to original spatial resolution.

    Temporal planes are bilinear-upsampled back to their original W.
    Spatial planes were only quantised to fp16, so just cast back to fp32.
    """
    import torch
    import torch.nn.functional as F

    new_sd = {}
    for k, v in state_dict.items():
        if k in ds_info:
            orig = ds_info[k]["original_shape"]
            is_temporal = ds_info[k].get("is_temporal", False)

            if is_temporal:
                # Bilinear upsample spatial axis back to original W
                restored = F.interpolate(
                    v.float(), size=(orig[2], orig[3]), mode="bilinear", align_corners=True
                )
                new_sd[k] = restored
            else:
                # Spatial plane was only fp16-quantised; cast back
                new_sd[k] = v.float() if v.dtype == torch.float16 else v
        else:
            new_sd[k] = v
    return new_sd


# ── Strategy ──────────────────────────────────────────────────────────────

class HexPlaneCompressionStrategy(CompressionStrategy):
    """Compress the HexPlane grids in the deformation network.

    Parameters
    ----------
    method : str
        ``"quantize"`` (safe baseline), ``"svd"`` (experimental),
        or ``"downsample"`` (experimental).
    svd_rank : int
        Rank for truncated SVD (only used when ``method="svd"``).
    downsample_factor : float
        Spatial downsampling factor (only used when ``method="downsample"``).
    """

    VALID_METHODS = ("quantize", "svd", "downsample")

    def __init__(
        self,
        method: str = "quantize",
        svd_rank: int = 16,
        svd_temporal_rank_multiplier: float = 4.0,
        svd_energy_threshold: float = 0.999,
        downsample_factor: float = 2.0,
        **kwargs,
    ):
        if method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}, got '{method}'")

        # Set before super().__init__() because base class accesses self.name
        # which depends on self.method
        self.method = method
        self.svd_rank = svd_rank
        self.svd_temporal_rank_multiplier = svd_temporal_rank_multiplier
        self.svd_energy_threshold = svd_energy_threshold
        self.downsample_factor = downsample_factor

        super().__init__(
            method=method,
            svd_rank=svd_rank,
            svd_temporal_rank_multiplier=svd_temporal_rank_multiplier,
            svd_energy_threshold=svd_energy_threshold,
            downsample_factor=downsample_factor,
            **kwargs,
        )
        self._extra_info: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return f"hexplane_{self.method}"

    # ── Gaussian (passthrough) ────────────────────────────────────────

    def compress_gaussian(self, data: GaussianData) -> GaussianData:
        return data

    def decompress_gaussian(self, data: GaussianData, metadata: Dict[str, Any]) -> GaussianData:
        return data

    # ── Deformation ───────────────────────────────────────────────────

    def compress_deformation(self, data: DeformationData) -> DeformationData:
        sd = data.state_dict

        if self.method == "quantize":
            data.state_dict = _quantize_grids(sd)
            self._extra_info = {"method": "quantize"}

        elif self.method == "svd":
            new_sd, svd_info = _svd_compress_grids(
                sd,
                self.svd_rank,
                temporal_rank_multiplier=self.svd_temporal_rank_multiplier,
                energy_threshold=self.svd_energy_threshold,
            )
            data.state_dict = new_sd
            self._extra_info = {"method": "svd", "svd_info": svd_info}

        elif self.method == "downsample":
            new_sd, ds_info = _downsample_grids(sd, self.downsample_factor)
            data.state_dict = new_sd
            self._extra_info = {"method": "downsample", "ds_info": ds_info}

        return data

    def decompress_deformation(
        self, data: DeformationData, metadata: Dict[str, Any]
    ) -> DeformationData:
        method = metadata.get("hexplane_info", {}).get("method", "quantize")

        if method == "quantize":
            data.state_dict = _dequantize_grids(data.state_dict)

        elif method == "svd":
            svd_info = metadata["hexplane_info"]["svd_info"]
            data.state_dict = _svd_decompress_grids(data.state_dict, svd_info)

        elif method == "downsample":
            ds_info = metadata["hexplane_info"]["ds_info"]
            data.state_dict = _upsample_grids(data.state_dict, ds_info)

        return data

    # ── Metadata ──────────────────────────────────────────────────────

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "strategy": self.name,
            "params": self.params,
            "hexplane_info": self._extra_info,
        }
