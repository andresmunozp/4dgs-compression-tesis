"""
Composable compression pipeline.

Chains multiple CompressionStrategy instances sequentially during
compression and in reverse order during decompression.
Tracks per-strategy statistics and aggregates them.
"""

from __future__ import annotations

import io
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

from compression.base import (
    CompressionStats,
    CompressionStrategy,
    CompressedPayload,
    DeformationData,
    GaussianData,
)
from compression.serializer import ModelSerializer


# ── Strategy registry ─────────────────────────────────────────────────────

_STRATEGY_REGISTRY: Dict[str, type] = {}


def register_strategy(cls: type) -> type:
    """Decorator — register a strategy class by its ``name`` property."""
    # Instantiate temporarily just to get the name
    _STRATEGY_REGISTRY[cls.__name__] = cls
    return cls


def get_strategy_class(name: str) -> type:
    """Look up a strategy class by class name."""
    if name in _STRATEGY_REGISTRY:
        return _STRATEGY_REGISTRY[name]
    raise KeyError(f"Unknown strategy: '{name}'.  Available: {list(_STRATEGY_REGISTRY)}")


# ── Pipeline ──────────────────────────────────────────────────────────────

class CompressionPipeline:
    """Orchestrates ordered application of compression strategies.

    Parameters
    ----------
    strategies : list[CompressionStrategy]
        Ordered list of strategies to apply (first → last on compress,
        last → first on decompress).
    config : dict, optional
        Full YAML config dict for embedding in the archive manifest.
    """

    def __init__(
        self,
        strategies: List[CompressionStrategy],
        config: Optional[Dict[str, Any]] = None,
    ):
        self.strategies = strategies
        self.config = config or {}
        self._all_stats: List[CompressionStats] = []
        self._all_metadata: List[Dict[str, Any]] = []

    # ── Compress ──────────────────────────────────────────────────────
    def compress(
        self,
        gaussian: GaussianData,
        deformation: DeformationData,
    ) -> Tuple[GaussianData, DeformationData, List[CompressionStats], List[Dict[str, Any]]]:
        """Apply all strategies sequentially and return transformed data.

        Returns
        -------
        (gaussian, deformation, stats, metadata)
        """
        gaussian.validate()
        stats_list: List[CompressionStats] = []
        metadata_list: List[Dict[str, Any]] = []

        for strategy in self.strategies:
            orig_bytes = gaussian.total_bytes + deformation.total_bytes

            t0 = time.perf_counter()
            gaussian = strategy.compress_gaussian(gaussian)
            deformation = strategy.compress_deformation(deformation)
            elapsed = time.perf_counter() - t0

            new_bytes = gaussian.total_bytes + deformation.total_bytes
            st = CompressionStats(
                strategy_name=strategy.name,
                original_bytes=orig_bytes,
                compressed_bytes=new_bytes,
                compression_time_s=elapsed,
            )
            stats_list.append(st)
            metadata_list.append(strategy.get_metadata())

        self._all_stats = stats_list
        self._all_metadata = metadata_list
        return gaussian, deformation, stats_list, metadata_list

    # ── Decompress ────────────────────────────────────────────────────
    def decompress(
        self,
        gaussian: GaussianData,
        deformation: DeformationData,
        metadata_list: List[Dict[str, Any]],
    ) -> Tuple[GaussianData, DeformationData]:
        """Reverse all strategies in reverse order.

        Parameters
        ----------
        metadata_list : list[dict]
            The per-strategy metadata produced during compression.
        """
        for strategy, meta in zip(reversed(self.strategies), reversed(metadata_list)):
            t0 = time.perf_counter()
            gaussian = strategy.decompress_gaussian(gaussian, meta)
            deformation = strategy.decompress_deformation(deformation, meta)
            elapsed = time.perf_counter() - t0

            # update decompression time in the matching stats entry
            for st in self._all_stats:
                if st.strategy_name == strategy.name:
                    st.decompression_time_s = elapsed
                    break

        return gaussian, deformation

    # ── Full round-trip to binary archive ─────────────────────────────
    def compress_to_archive(
        self,
        gaussian: GaussianData,
        deformation: DeformationData,
    ) -> bytes:
        """Compress → serialize to binary .4dgs archive."""
        g, d, stats, meta = self.compress(gaussian, deformation)

        # Serialize deformation state_dict to bytes
        buf = io.BytesIO()
        torch.save(d.state_dict, buf)
        deform_bytes = buf.getvalue()

        return ModelSerializer.serialize(
            gaussian=g,
            deformation_bytes=deform_bytes,
            pipeline_config=self.config,
            strategy_metadata=meta,
            stats=stats,
        )

    def decompress_from_archive(
        self,
        archive: bytes,
        verify_checksums: bool = True,
    ) -> Tuple[GaussianData, DeformationData, Dict[str, Any]]:
        """Deserialize binary .4dgs archive → decompress."""
        gaussian, deform_bytes, manifest = ModelSerializer.deserialize(
            archive, verify_checksums=verify_checksums
        )

        # Reconstruct deformation
        buf = io.BytesIO(deform_bytes)
        state_dict = torch.load(buf, map_location="cpu", weights_only=False)
        deformation = DeformationData(state_dict=state_dict)

        metadata_list = manifest.get("strategy_metadata", [])
        gaussian, deformation = self.decompress(gaussian, deformation, metadata_list)

        return gaussian, deformation, manifest

    # ── Helpers ───────────────────────────────────────────────────────
    @property
    def stats(self) -> List[CompressionStats]:
        return self._all_stats

    def print_stats(self) -> None:
        print("\n" + "=" * 70)
        print("Compression Pipeline Statistics")
        print("=" * 70)
        for s in self._all_stats:
            print(
                f"  {s.strategy_name:30s} | "
                f"ratio {s.ratio:6.2f}x | "
                f"savings {s.savings_pct:5.1f}% | "
                f"compress {s.compression_time_s:.3f}s | "
                f"decompress {s.decompression_time_s:.3f}s"
            )
        if self._all_stats:
            total_orig = self._all_stats[0].original_bytes
            total_comp = self._all_stats[-1].compressed_bytes
            total_time = sum(s.compression_time_s for s in self._all_stats)
            if total_comp > 0:
                print(f"  {'TOTAL':30s} | "
                      f"ratio {total_orig / total_comp:6.2f}x | "
                      f"savings {(1 - total_comp / total_orig) * 100:5.1f}% | "
                      f"time {total_time:.3f}s")
        print("=" * 70)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CompressionPipeline":
        """Build a pipeline from a YAML-style dict config.

        Expected format::

            strategies:
              - name: PruningStrategy
                params:
                  opacity_threshold: 0.005
              - name: SHReductionStrategy
                params:
                  target_sh_degree: 1
              ...
        """
        from compression.strategies import STRATEGY_CLASSES  # late import to avoid circular

        strats = []
        for entry in config.get("strategies", []):
            cls_name = entry["name"]
            params = entry.get("params", {})
            if cls_name in STRATEGY_CLASSES:
                strats.append(STRATEGY_CLASSES[cls_name](**params))
            else:
                # Try the registry
                klass = get_strategy_class(cls_name)
                strats.append(klass(**params))
        return cls(strategies=strats, config=config)

    @classmethod
    def from_yaml(cls, path: str) -> "CompressionPipeline":
        """Load pipeline config from a YAML file."""
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_config(config)
