"""
Model serializer with manifest, versioning, and checksums.

Handles the conversion between in-memory data structures (GaussianData,
DeformationData) and a self-describing binary archive with integrity
verification.

Binary archive layout (.4dgs):
  [HEADER]
    - magic (4 bytes): b'4DGS'
    - version (2 bytes): uint16
    - manifest_size (4 bytes): uint32  (size of JSON manifest blob)
  [MANIFEST]
    - JSON blob describing every section, shapes, dtypes, checksums
  [SECTIONS]
    - sequential binary blobs referenced by the manifest
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from compression.base import (
    CompressedPayload,
    CompressionStats,
    DeformationData,
    GaussianData,
)

# ── Constants ──────────────────────────────────────────────────────────────
MAGIC = b"4DGS"
FORMAT_VERSION = 1  # bump when breaking the binary layout
HEADER_FMT = "<4sHI"  # magic(4) + version(2) + manifest_size(4) = 10 bytes
HEADER_SIZE = struct.calcsize(HEADER_FMT)


# ── Manifest builder ─────────────────────────────────────────────────────
def _array_meta(name: str, arr: np.ndarray, offset: int) -> Dict[str, Any]:
    """Create a manifest entry for a single numpy array."""
    data = arr.tobytes()
    return {
        "name": name,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "offset": offset,
        "size": len(data),
        "checksum": hashlib.sha256(data).hexdigest(),
    }


# ── Serializer ────────────────────────────────────────────────────────────

class ModelSerializer:
    """Serialize / deserialize a compressed 4DGS model to a single binary blob.

    The archive is fully self-describing via the embedded JSON manifest.
    Every section has an independent SHA-256 checksum for integrity
    verification, and the manifest itself records the pipeline config
    and version info.
    """

    @staticmethod
    def serialize(
        gaussian: GaussianData,
        deformation_bytes: bytes,
        pipeline_config: Dict[str, Any],
        strategy_metadata: List[Dict[str, Any]],
        stats: List[CompressionStats],
    ) -> bytes:
        """Pack everything into a single .4dgs binary archive.

        Parameters
        ----------
        gaussian : GaussianData
            The (possibly compressed/quantized) Gaussian parameters.
        deformation_bytes : bytes
            Already-serialized deformation network (e.g. via torch.save to BytesIO).
        pipeline_config : dict
            The YAML / dict config that was used for compression.
        strategy_metadata : list[dict]
            Per-strategy metadata needed for decompression.
        stats : list[CompressionStats]
            Compression statistics to embed for traceability.

        Returns
        -------
        bytes
            Complete binary archive.
        """
        sections: List[bytes] = []
        manifest_sections: List[Dict[str, Any]] = []
        current_offset = 0  # offset relative to start-of-sections

        # ── Gaussian arrays ──
        gaussian_arrays = [
            ("xyz", gaussian.xyz),
            ("features_dc", gaussian.features_dc),
            ("features_rest", gaussian.features_rest),
            ("opacity", gaussian.opacity),
            ("scaling", gaussian.scaling),
            ("rotation", gaussian.rotation),
        ]
        if gaussian.deformation_table is not None:
            gaussian_arrays.append(("deformation_table", gaussian.deformation_table))
        if gaussian.deformation_accum is not None:
            gaussian_arrays.append(("deformation_accum", gaussian.deformation_accum))

        for name, arr in gaussian_arrays:
            raw = arr.tobytes()
            meta = _array_meta(name, arr, current_offset)
            manifest_sections.append(meta)
            sections.append(raw)
            current_offset += len(raw)

        # ── Deformation network blob ──
        deform_checksum = hashlib.sha256(deformation_bytes).hexdigest()
        manifest_sections.append({
            "name": "deformation_network",
            "dtype": "torch_state_dict",
            "shape": [],
            "offset": current_offset,
            "size": len(deformation_bytes),
            "checksum": deform_checksum,
        })
        sections.append(deformation_bytes)
        current_offset += len(deformation_bytes)

        # ── Build manifest ──
        manifest = {
            "format_version": FORMAT_VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "num_gaussians": gaussian.num_gaussians,
            "sh_degree": gaussian.sh_degree,
            "active_sh_degree": gaussian.active_sh_degree,
            "pipeline_config": pipeline_config,
            "strategy_metadata": strategy_metadata,
            "compression_stats": [
                {
                    "strategy_name": s.strategy_name,
                    "original_bytes": s.original_bytes,
                    "compressed_bytes": s.compressed_bytes,
                    "compression_time_s": round(s.compression_time_s, 6),
                    "decompression_time_s": round(s.decompression_time_s, 6),
                    "ratio": round(s.ratio, 4),
                    "savings_pct": round(s.savings_pct, 2),
                    "extra": s.extra,
                }
                for s in stats
            ],
            "sections": manifest_sections,
        }

        manifest_bytes = json.dumps(manifest, indent=None, separators=(",", ":")).encode("utf-8")

        # ── Assemble archive ──
        header = struct.pack(HEADER_FMT, MAGIC, FORMAT_VERSION, len(manifest_bytes))
        archive = io.BytesIO()
        archive.write(header)
        archive.write(manifest_bytes)
        for section in sections:
            archive.write(section)

        # Compute whole-archive checksum and append
        archive_bytes = archive.getvalue()
        archive_checksum = hashlib.sha256(archive_bytes).hexdigest()

        # Re-inject archive checksum into manifest (update in-place)
        manifest["archive_checksum"] = archive_checksum
        manifest_bytes = json.dumps(manifest, indent=None, separators=(",", ":")).encode("utf-8")
        header = struct.pack(HEADER_FMT, MAGIC, FORMAT_VERSION, len(manifest_bytes))

        final = io.BytesIO()
        final.write(header)
        final.write(manifest_bytes)
        for section in sections:
            final.write(section)

        return final.getvalue()

    @staticmethod
    def deserialize(archive: bytes, verify_checksums: bool = True) -> Tuple[
        GaussianData, bytes, Dict[str, Any]
    ]:
        """Unpack a .4dgs archive back into constituent data.

        Parameters
        ----------
        archive : bytes
            The complete binary archive.
        verify_checksums : bool
            If True, verify SHA-256 checksum of every section.

        Returns
        -------
        tuple of (GaussianData, deformation_bytes, manifest)

        Raises
        ------
        ValueError
            On magic/version mismatch or checksum failure.
        """
        if len(archive) < HEADER_SIZE:
            raise ValueError(f"Archive too small ({len(archive)} bytes)")

        magic, version, manifest_size = struct.unpack_from(HEADER_FMT, archive, 0)

        if magic != MAGIC:
            raise ValueError(f"Bad magic: {magic!r} (expected {MAGIC!r})")
        if version > FORMAT_VERSION:
            raise ValueError(
                f"Archive version {version} not supported (max {FORMAT_VERSION})"
            )

        manifest_start = HEADER_SIZE
        manifest_end = manifest_start + manifest_size
        manifest = json.loads(archive[manifest_start:manifest_end].decode("utf-8"))

        sections_base = manifest_end  # all section offsets are relative to here

        # ── Verify & extract sections ──
        arrays: Dict[str, np.ndarray] = {}
        deformation_bytes = b""

        for sec in manifest["sections"]:
            start = sections_base + sec["offset"]
            end = start + sec["size"]
            raw = archive[start:end]

            if verify_checksums:
                actual = hashlib.sha256(raw).hexdigest()
                if actual != sec["checksum"]:
                    raise ValueError(
                        f"Checksum mismatch for section '{sec['name']}': "
                        f"expected {sec['checksum']}, got {actual}"
                    )

            if sec["name"] == "deformation_network":
                deformation_bytes = raw
            else:
                dtype = np.dtype(sec["dtype"])
                arr = np.frombuffer(raw, dtype=dtype).reshape(sec["shape"]).copy()
                arrays[sec["name"]] = arr

        gaussian = GaussianData(
            xyz=arrays["xyz"],
            features_dc=arrays["features_dc"],
            features_rest=arrays["features_rest"],
            opacity=arrays["opacity"],
            scaling=arrays["scaling"],
            rotation=arrays["rotation"],
            sh_degree=manifest["sh_degree"],
            active_sh_degree=manifest["active_sh_degree"],
            deformation_table=arrays.get("deformation_table"),
            deformation_accum=arrays.get("deformation_accum"),
        )

        return gaussian, deformation_bytes, manifest

    @staticmethod
    def read_manifest_only(archive: bytes) -> Dict[str, Any]:
        """Read just the manifest without deserializing data sections."""
        if len(archive) < HEADER_SIZE:
            raise ValueError(f"Archive too small ({len(archive)} bytes)")

        magic, version, manifest_size = struct.unpack_from(HEADER_FMT, archive, 0)
        if magic != MAGIC:
            raise ValueError(f"Bad magic: {magic!r}")

        manifest_start = HEADER_SIZE
        manifest_end = manifest_start + manifest_size
        return json.loads(archive[manifest_start:manifest_end].decode("utf-8"))
