"""
Entropy-coding strategy — lossless compression on the final byte stream.

Applied as the **last** step in a compression pipeline, after lossy
strategies (pruning, quantization, SH reduction) have already reduced
the data.

Supported algorithms:
  - ``zlib``  — standard deflate, widely available
  - ``zstd``  — Facebook's Zstandard, better ratio + speed
  - ``lz4``   — extremely fast, moderate ratio
  - ``gzip``  — gzip wrapper around deflate
"""

from __future__ import annotations

import io
import zlib
from typing import Any, Dict, Optional

import numpy as np

from compression.base import CompressionStrategy, DeformationData, GaussianData

# Try optional high-performance codecs
try:
    import zstandard as _zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

try:
    import lz4.frame as _lz4
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False


# ── Codec wrappers ────────────────────────────────────────────────────────

def _compress_bytes(data: bytes, algorithm: str, level: int) -> bytes:
    if algorithm == "zlib":
        return zlib.compress(data, level)
    elif algorithm == "gzip":
        import gzip
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=level) as f:
            f.write(data)
        return buf.getvalue()
    elif algorithm == "zstd":
        if not HAS_ZSTD:
            raise ImportError("zstandard package not installed. pip install zstandard")
        ctx = _zstd.ZstdCompressor(level=level)
        return ctx.compress(data)
    elif algorithm == "lz4":
        if not HAS_LZ4:
            raise ImportError("lz4 package not installed. pip install lz4")
        return _lz4.compress(data, compression_level=level)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def _decompress_bytes(data: bytes, algorithm: str) -> bytes:
    if algorithm == "zlib":
        return zlib.decompress(data)
    elif algorithm == "gzip":
        import gzip
        buf = io.BytesIO(data)
        with gzip.GzipFile(fileobj=buf, mode="rb") as f:
            return f.read()
    elif algorithm == "zstd":
        if not HAS_ZSTD:
            raise ImportError("zstandard package not installed. pip install zstandard")
        ctx = _zstd.ZstdDecompressor()
        return ctx.decompress(data)
    elif algorithm == "lz4":
        if not HAS_LZ4:
            raise ImportError("lz4 package not installed. pip install lz4")
        return _lz4.decompress(data)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


# ── Strategy ──────────────────────────────────────────────────────────────

class EntropyCodingStrategy(CompressionStrategy):
    """Apply lossless entropy coding on the serialised byte stream.

    This strategy works differently from the others: it packs/unpacks the
    Gaussian arrays into a flat byte blob, compresses it, and stores the
    compressed blob *inside* the GaussianData arrays (as a single uint8
    array + metadata for reconstruction).

    Parameters
    ----------
    algorithm : str
        One of ``"zlib"``, ``"gzip"``, ``"zstd"``, ``"lz4"``.
    level : int
        Codec-specific compression level.
    """

    VALID_ALGORITHMS = ("zlib", "gzip", "zstd", "lz4")

    def __init__(self, algorithm: str = "zlib", level: int = 6, **kwargs):
        if algorithm not in self.VALID_ALGORITHMS:
            raise ValueError(f"algorithm must be one of {self.VALID_ALGORITHMS}")
        # Set before super().__init__() because base class accesses self.name
        # which depends on self.algorithm
        self.algorithm = algorithm
        self.level = level
        super().__init__(algorithm=algorithm, level=level, **kwargs)
        self._encode_info: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return f"entropy_{self.algorithm}"

    # ── Gaussian ──────────────────────────────────────────────────────

    def compress_gaussian(self, data: GaussianData) -> GaussianData:
        """Entropy-code each Gaussian array individually."""
        self._encode_info = {}
        ATTRS = ["xyz", "features_dc", "features_rest", "opacity", "scaling", "rotation"]

        for attr in ATTRS:
            arr = getattr(data, attr)
            raw = arr.tobytes()
            compressed = _compress_bytes(raw, self.algorithm, self.level)
            # Store compressed data as uint8 array
            comp_arr = np.frombuffer(compressed, dtype=np.uint8).copy()
            self._encode_info[attr] = {
                "original_dtype": str(arr.dtype),
                "original_shape": list(arr.shape),
                "original_size": len(raw),
                "compressed_size": len(compressed),
            }
            setattr(data, attr, comp_arr.reshape(1, -1) if comp_arr.ndim == 1 else comp_arr)

        # Also compress auxiliary arrays
        for attr in ("deformation_table", "deformation_accum"):
            arr = getattr(data, attr, None)
            if arr is not None:
                raw = arr.tobytes()
                compressed = _compress_bytes(raw, self.algorithm, self.level)
                comp_arr = np.frombuffer(compressed, dtype=np.uint8).copy()
                self._encode_info[attr] = {
                    "original_dtype": str(arr.dtype),
                    "original_shape": list(arr.shape),
                    "original_size": len(raw),
                    "compressed_size": len(compressed),
                }
                setattr(data, attr, comp_arr.reshape(1, -1) if comp_arr.ndim == 1 else comp_arr)

        return data

    def decompress_gaussian(
        self, data: GaussianData, metadata: Dict[str, Any]
    ) -> GaussianData:
        encode_info = metadata.get("encode_info", {})
        algorithm = metadata.get("params", {}).get("algorithm", self.algorithm)

        ATTRS = ["xyz", "features_dc", "features_rest", "opacity", "scaling", "rotation",
                 "deformation_table", "deformation_accum"]

        for attr in ATTRS:
            if attr not in encode_info:
                continue
            info = encode_info[attr]
            comp_arr = getattr(data, attr, None)
            if comp_arr is None:
                continue

            compressed = comp_arr.tobytes()
            raw = _decompress_bytes(compressed, algorithm)
            dtype = np.dtype(info["original_dtype"])
            shape = tuple(info["original_shape"])
            restored = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
            setattr(data, attr, restored)

        return data

    # ── Deformation ───────────────────────────────────────────────────

    def compress_deformation(self, data: DeformationData) -> DeformationData:
        # Deformation is serialized separately via torch.save; we compress
        # that byte blob in the serializer layer, not here.
        return data

    def decompress_deformation(
        self, data: DeformationData, metadata: Dict[str, Any]
    ) -> DeformationData:
        return data

    # ── Metadata ──────────────────────────────────────────────────────

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "strategy": self.name,
            "params": self.params,
            "encode_info": self._encode_info,
        }
