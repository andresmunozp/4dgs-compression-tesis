"""
Quantization strategy — reduce numeric precision of Gaussian and/or
deformation network parameters.

Supports per-attribute dtype selection (float16, int8, int16) with
min/max scaling for integer types.  Deformation network weights can
optionally be quantized to float16.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from compression.base import CompressionStrategy, DeformationData, GaussianData


# ── Helpers ───────────────────────────────────────────────────────────────

def _quantize_array(arr: np.ndarray, target_dtype: str) -> tuple:
    """Quantize *arr* to *target_dtype*, return (quantized, scale, zero_point, orig_dtype).

    For float targets (float16): simple cast.
    For integer targets (int8, int16, uint8): affine quantization with
    per-channel min/max.
    """
    orig_dtype = str(arr.dtype)
    target = np.dtype(target_dtype)

    if target.kind == "f":
        # Float→Float cast (e.g. float32→float16)
        return arr.astype(target), None, None, orig_dtype

    # Integer quantization (affine per-column)
    flat = arr.reshape(arr.shape[0], -1).astype(np.float64)
    col_min = flat.min(axis=0, keepdims=True)
    col_max = flat.max(axis=0, keepdims=True)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0  # avoid div-by-zero

    info = np.iinfo(target)
    scale = col_range / (info.max - info.min)
    zero_point = info.min - col_min / scale

    quantized = np.clip(
        np.round(flat / scale + zero_point), info.min, info.max
    ).astype(target).reshape(arr.shape)

    return quantized, scale.astype(np.float32), zero_point.astype(np.float32), orig_dtype


def _dequantize_array(
    arr: np.ndarray,
    scale: Optional[np.ndarray],
    zero_point: Optional[np.ndarray],
    orig_dtype: str,
) -> np.ndarray:
    """Reverse of ``_quantize_array``."""
    target = np.dtype(orig_dtype)

    if scale is None:
        # Was a float cast
        return arr.astype(target)

    flat = arr.reshape(arr.shape[0], -1).astype(np.float64)
    restored = (flat - zero_point) * scale
    return restored.astype(target).reshape(arr.shape)


# ── Strategy ──────────────────────────────────────────────────────────────

class QuantizationStrategy(CompressionStrategy):
    """Reduce numeric precision of Gaussian attributes.

    Parameters
    ----------
    attribute_dtypes : dict
        Mapping ``{attribute_name: target_dtype_str}``.
        Valid attribute names: ``xyz``, ``features_dc``, ``features_rest``,
        ``opacity``, ``scaling``, ``rotation``.
        Valid dtypes: ``float16``, ``int8``, ``int16``, ``uint8``.
        Attributes not listed keep their original dtype.
    quantize_deformation : bool
        If True, cast deformation network weights to float16.
    """

    def __init__(
        self,
        attribute_dtypes: Optional[Dict[str, str]] = None,
        quantize_deformation: bool = False,
        **kwargs,
    ):
        super().__init__(
            attribute_dtypes=attribute_dtypes or {},
            quantize_deformation=quantize_deformation,
            **kwargs,
        )
        self.attribute_dtypes: Dict[str, str] = attribute_dtypes or {}
        self.quantize_deformation = quantize_deformation

        # Stored during compression for decompression
        self._quant_params: Dict[str, dict] = {}

    @property
    def name(self) -> str:
        return "quantization"

    # ── Gaussian ──────────────────────────────────────────────────────

    def compress_gaussian(self, data: GaussianData) -> GaussianData:
        self._quant_params = {}
        ATTR_NAMES = ["xyz", "features_dc", "features_rest", "opacity", "scaling", "rotation"]

        for attr in ATTR_NAMES:
            if attr not in self.attribute_dtypes:
                continue
            target = self.attribute_dtypes[attr]
            arr = getattr(data, attr)
            quantized, scale, zp, orig = _quantize_array(arr, target)
            setattr(data, attr, quantized)
            self._quant_params[attr] = {
                "orig_dtype": orig,
                "target_dtype": target,
                "has_scale": scale is not None,
            }
            if scale is not None:
                self._quant_params[attr]["scale"] = scale.tolist()
                self._quant_params[attr]["zero_point"] = zp.tolist()

        return data

    def decompress_gaussian(self, data: GaussianData, metadata: Dict[str, Any]) -> GaussianData:
        qp = metadata.get("quant_params", {})
        for attr, info in qp.items():
            arr = getattr(data, attr)
            scale = np.array(info["scale"], dtype=np.float32) if info.get("has_scale") else None
            zp = np.array(info["zero_point"], dtype=np.float32) if info.get("has_scale") else None
            restored = _dequantize_array(arr, scale, zp, info["orig_dtype"])
            setattr(data, attr, restored)
        return data

    # ── Deformation ───────────────────────────────────────────────────

    def compress_deformation(self, data: DeformationData) -> DeformationData:
        if not self.quantize_deformation:
            return data

        import torch

        new_sd = {}
        for k, v in data.state_dict.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                new_sd[k] = v.half()
            else:
                new_sd[k] = v
        data.state_dict = new_sd
        return data

    def decompress_deformation(self, data: DeformationData, metadata: Dict[str, Any]) -> DeformationData:
        if not metadata.get("params", {}).get("quantize_deformation", False):
            return data

        import torch

        new_sd = {}
        for k, v in data.state_dict.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float16:
                new_sd[k] = v.float()
            else:
                new_sd[k] = v
        data.state_dict = new_sd
        return data

    # ── Metadata ──────────────────────────────────────────────────────

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "strategy": self.name,
            "params": self.params,
            "quant_params": self._quant_params,
        }
