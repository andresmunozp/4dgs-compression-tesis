"""Reader for results_json/*.json (training evaluation) files."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import List, Optional

from ...domain.enums import ResultCategory, ResultSource
from ...domain.models import (
    MetricRecord,
    ModelInfo,
    QualityMetrics,
    TimingMetrics,
)


class TrainingJsonReader:
    """Adapter that reads training-evaluation JSON files from ``results_json/``.

    These files are produced by ``train_eval_json.sh`` and contain a single
    JSON object with scene info, timing, ``metrics_full``, VMAF, and PLY
    export metadata.
    """

    @property
    def source_type(self) -> ResultSource:
        return ResultSource.TRAINING_JSON

    def supports(self, path: Path) -> bool:
        if not path.is_file():
            return False
        if not path.name.endswith(".json"):
            return False
        # Heuristic: training jsons live in results_json/ and start with "4dgs_"
        return (
            "results_json" in path.parts
            or path.name.startswith("4dgs_")
        )

    def read(self, path: Path) -> List[MetricRecord]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        record = self._parse(data, path)
        return [record]

    # ── Internals ───────────────────────────────────────────────

    @staticmethod
    def _extract_iteration(data: dict, path: Path) -> Optional[int]:
        """Try to determine iteration from metrics_full keys or filename."""
        for key in data.get("metrics_full", {}).keys():
            m = re.search(r"(\d+)$", key)
            if m:
                return int(m.group(1))
        m = re.search(r"it(\d+)", path.stem)
        if m:
            return int(m.group(1))
        return None

    @staticmethod
    def _make_id(scene: str, iteration: Optional[int], path: Path) -> str:
        raw = f"training_json:{scene}:it{iteration}:{path.stem}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _parse(self, data: dict, path: Path) -> MetricRecord:
        scene = data.get("scene", "unknown")
        iteration = self._extract_iteration(data, path)
        record_id = self._make_id(scene, iteration, path)

        # Parse quality from metrics_full
        quality = self._parse_quality(data, iteration)

        # Parse timing
        timing = TimingMetrics(
            train_time_s=data.get("train_time_sec"),
            render_time_s=data.get("render_time_sec"),
            metrics_eval_time_s=data.get("metrics_time_sec"),
            ply_export_time_s=data.get("ply_perframe_time_sec"),
        )

        # Parse model info from ply_pertimestamp
        ply = data.get("ply_pertimestamp", {})
        model_info = ModelInfo(
            iteration=iteration or ply.get("iteration"),
            num_ply_files=ply.get("num_ply_files"),
            total_ply_size_bytes=ply.get("total_size_bytes"),
        )

        # Build tags
        tags = {
            "source_path": str(path),
            "expname": data.get("expname", ""),
        }
        if iteration is not None:
            tags["iteration"] = str(iteration)

        return MetricRecord(
            id=record_id,
            source=ResultSource.TRAINING_JSON,
            category=ResultCategory.TRAINING,
            name=f"{scene}_it{iteration}" if iteration else scene,
            scene=scene,
            tags=tags,
            quality_metrics=quality,
            timing_metrics=timing,
            model_info=model_info,
        )

    @staticmethod
    def _parse_quality(data: dict, iteration: Optional[int]) -> QualityMetrics:
        mf = data.get("metrics_full", {})
        # Pick the right key: "ours_4000", "ours_2000", etc.
        block = {}
        if iteration is not None:
            block = mf.get(f"ours_{iteration}", {})
        if not block:
            # Fallback: pick the first (or only) entry
            for key in mf:
                block = mf[key]
                break

        return QualityMetrics(
            psnr=block.get("PSNR"),
            ssim=block.get("SSIM"),
            lpips_vgg=block.get("LPIPS-vgg"),
            lpips_alex=block.get("LPIPS-alex"),
            ms_ssim=block.get("MS-SSIM"),
            d_ssim=block.get("D-SSIM"),
            vmaf=data.get("VMAF"),
        )
