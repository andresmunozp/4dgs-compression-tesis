"""
LightGaussian Pruning Strategy — Volume-weighted Importance Score.

Implements the pruning approach from:
    "LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS"
    Fan et al., NeurIPS 2024  (arXiv:2311.17245)

Key idea — **Global Significance Score**:
    score_i = (V_i / V_kth)^v_pow  ×  importance_i

where V_i = prod(exp(scaling_raw_i)) is the volume of the i-th Gaussian,
V_kth is the 90th-percentile volume, and importance_i can be:

  • **parameter mode** (default, fast, no GPU/cameras needed):
        importance_i = sigmoid(opacity_raw_i)

  • **render mode** (more faithful to the original paper, needs cameras + GPU):
        importance_i = visibility_count_i / total_renders
    accumulated over forward render passes on training cameras.

Optionally incorporates a **4DGS-aware deformation weight** that boosts the
score of Gaussians with high accumulated deformation, preserving dynamic parts.

This strategy is completely independent of ``PruningStrategy`` and can coexist
with it in the same pipeline YAML.
"""

from __future__ import annotations

import json
import math
import os
import sys
from argparse import Namespace
from typing import Any, Dict, List, Optional

import numpy as np

from compression.base import CompressionStrategy, DeformationData, GaussianData


# ── Helpers ───────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x.astype(np.float64), -500, 500)))


class LightGaussianPruningStrategy(CompressionStrategy):
    """Prune Gaussians by Volume-weighted Importance Score (LightGaussian).

    Parameters
    ----------
    prune_percent : float
        Fraction of Gaussians to **remove** (0.0–1.0).  Bottom X% by
        significance score are pruned.
    prune_decay : float
        Multiplicative decay applied to ``prune_percent`` for iterative
        chaining (default 1.0 = no decay).
    v_pow : float
        Exponent for the normalised volume ratio.  Controls how much
        larger Gaussians are favoured.  Original paper default is 0.1.
    importance_mode : str
        ``"parameter"`` — fast, CPU-only, uses sigmoid(opacity) as proxy.
        ``"render"`` — runs forward render passes to count per-Gaussian
        visibility (requires ``source_path`` and GPU).
    deformation_weight : float
        Weight of the deformation-accumulation bonus (4DGS-specific).
        0 disables it (pure LightGaussian).  Values > 0 boost the score
        of Gaussians that undergo large deformations, penalising static
        or barely-moving ones less aggressively.
    source_path : str or None
        Path to the dataset source directory (only for ``render`` mode).
    model_path : str or None
        Path to the trained model root (only for ``render`` mode, to load
        deformation weights and cameras.json).
    iteration : int
        Model iteration to load deformation weights from (render mode).
    configs : str or None
        Path to hyper-parameter config file (e.g. arguments/dynerf/coffee_martini.py).
    num_views : int
        Number of training views to sample for visibility scoring
        (render mode only).
    temporal_samples : int
        Number of timestamps uniformly sampled per camera for the
        visibility pass (render mode only, 4DGS-specific).
    """

    def __init__(
        self,
        prune_percent: float = 0.3,
        prune_decay: float = 1.0,
        v_pow: float = 0.1,
        importance_mode: str = "parameter",
        deformation_weight: float = 0.0,
        source_path: Optional[str] = None,
        model_path: Optional[str] = None,
        iteration: int = -1,
        configs: Optional[str] = None,
        num_views: int = 50,
        temporal_samples: int = 5,
        **kwargs,
    ):
        super().__init__(
            prune_percent=prune_percent,
            prune_decay=prune_decay,
            v_pow=v_pow,
            importance_mode=importance_mode,
            deformation_weight=deformation_weight,
            source_path=source_path,
            model_path=model_path,
            iteration=iteration,
            configs=configs,
            num_views=num_views,
            temporal_samples=temporal_samples,
            **kwargs,
        )
        self.prune_percent = prune_percent
        self.prune_decay = prune_decay
        self.v_pow = v_pow
        self.importance_mode = importance_mode
        self.deformation_weight = deformation_weight
        self.source_path = source_path
        self.model_path = model_path
        self.iteration = iteration
        self.configs = configs
        self.num_views = num_views
        self.temporal_samples = temporal_samples

        self._index_map: Optional[np.ndarray] = None
        self._original_count: int = 0

    @property
    def name(self) -> str:
        return "lightgaussian_pruning"

    # ──────────────────────────────────────────────────────────────────
    #  Importance computation — PARAMETER mode  (fast, CPU-only)
    # ──────────────────────────────────────────────────────────────────

    def _compute_importance_parameter(self, data: GaussianData) -> np.ndarray:
        """Importance = sigmoid(opacity), optionally boosted by deformation.

        Returns shape (N,).
        """
        importance = _sigmoid(data.opacity.squeeze(-1))  # (N,)

        if self.deformation_weight > 0 and data.deformation_accum is not None:
            deform_mag = np.linalg.norm(data.deformation_accum, axis=-1)  # (N,)
            # Normalise to [0, 1] relative to max
            dmax = deform_mag.max()
            if dmax > 0:
                deform_norm = deform_mag / dmax
            else:
                deform_norm = deform_mag
            importance = importance + self.deformation_weight * deform_norm

        return importance

    # ──────────────────────────────────────────────────────────────────
    #  Importance computation — RENDER mode  (GPU, cameras needed)
    # ──────────────────────────────────────────────────────────────────

    def _compute_importance_render(self, data: GaussianData) -> np.ndarray:
        """Accumulate per-Gaussian visibility over training views.

        For each sampled camera × timestamp, run a forward render pass
        and count how many times each Gaussian lands on screen
        (``radii > 0``).  The normalised count is the importance.

        Returns shape (N,).
        """
        import torch
        import torch.nn as nn

        # ----------------------------------------------------------
        #  1. Load cameras
        # ----------------------------------------------------------
        cameras = self._load_cameras()
        if not cameras:
            print("  [LightGaussian] No cameras found — falling back to parameter mode")
            return self._compute_importance_parameter(data)

        # ----------------------------------------------------------
        #  2. Build a GaussianModel on GPU from GaussianData
        # ----------------------------------------------------------
        gaussians = self._build_gaussian_model(data)

        # ----------------------------------------------------------
        #  3. Render and accumulate visibility
        # ----------------------------------------------------------
        from gaussian_renderer import render as gs_render

        pipe = Namespace(
            convert_SHs_python=False,
            compute_cov3D_python=False,
            debug=False,
        )
        bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        N = data.num_gaussians
        visibility_count = torch.zeros(N, dtype=torch.float32, device="cuda")
        total_renders = 0

        # Sample cameras uniformly
        n_cams = min(self.num_views, len(cameras))
        indices = np.linspace(0, len(cameras) - 1, n_cams, dtype=int)

        timestamps = np.linspace(0.0, 1.0, self.temporal_samples)

        with torch.no_grad():
            for cam_idx in indices:
                cam_info = cameras[cam_idx]
                for t in timestamps:
                    mini_cam = self._caminfo_to_minicam(cam_info, float(t))
                    result = gs_render(
                        mini_cam, gaussians, pipe, bg,
                        stage="fine", cam_type=None,
                    )
                    vis = result["visibility_filter"]  # (N,) bool
                    visibility_count += vis.float()
                    total_renders += 1

        importance = (visibility_count / max(total_renders, 1)).cpu().numpy()

        # Optionally add deformation bonus
        if self.deformation_weight > 0 and data.deformation_accum is not None:
            deform_mag = np.linalg.norm(data.deformation_accum, axis=-1)
            dmax = deform_mag.max()
            if dmax > 0:
                deform_norm = deform_mag / dmax
            else:
                deform_norm = deform_mag
            importance = importance + self.deformation_weight * deform_norm

        # Cleanup GPU
        del gaussians, visibility_count
        torch.cuda.empty_cache()

        return importance

    # ──────────────────────────────────────────────────────────────────
    #  Volume-weighted score  (core of LightGaussian)
    # ──────────────────────────────────────────────────────────────────

    def _calculate_v_imp_score(
        self, data: GaussianData, importance: np.ndarray
    ) -> np.ndarray:
        """Compute ``v_list = (volume / V_kth)^v_pow  *  importance``.

        Faithfully reproduces the ``calculate_v_imp_score`` function from
        the LightGaussian codebase (prune.py), adapted for numpy.

        Parameters
        ----------
        data : GaussianData
            Need ``data.scaling`` in **raw** (log) space.
        importance : np.ndarray, shape (N,)
            Per-Gaussian importance score.

        Returns
        -------
        np.ndarray, shape (N,)
            Final significance score (higher = more important).
        """
        # Volume = product of scales in activated space
        # scaling is stored as log-scale → exp → real scale; volume = prod
        volume = np.prod(np.exp(data.scaling.astype(np.float64)), axis=1)  # (N,)

        # 90th-percentile volume
        sorted_vol = np.sort(volume)[::-1]  # descending
        kth_idx = int(len(volume) * 0.9)
        kth_idx = min(kth_idx, len(sorted_vol) - 1)
        kth_volume = sorted_vol[kth_idx]

        # Normalise and raise to v_pow
        if kth_volume > 0:
            v_ratio = np.power(volume / kth_volume, self.v_pow)
        else:
            v_ratio = np.ones_like(volume)

        score = v_ratio * importance
        return score.astype(np.float64)

    # ──────────────────────────────────────────────────────────────────
    #  Mask & index map
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_mask(scores: np.ndarray, prune_percent: float) -> np.ndarray:
        """Return boolean keep-mask (True = keep).

        The bottom ``prune_percent`` by score are pruned.
        """
        N = len(scores)
        n_prune = int(N * prune_percent)
        n_prune = min(n_prune, N - 1)  # always keep at least 1

        # Indices sorted ascending by score → lowest first
        sorted_idx = np.argsort(scores)
        prune_set = set(sorted_idx[:n_prune].tolist())

        keep = np.ones(N, dtype=bool)
        for idx in prune_set:
            keep[idx] = False
        return keep

    @staticmethod
    def _build_index_map(keep_mask: np.ndarray) -> np.ndarray:
        """Map original_idx → new_idx (or -1 if pruned)."""
        index_map = np.full(len(keep_mask), -1, dtype=np.int32)
        new_idx = 0
        for orig_idx, kept in enumerate(keep_mask):
            if kept:
                index_map[orig_idx] = new_idx
                new_idx += 1
        return index_map

    # ──────────────────────────────────────────────────────────────────
    #  CompressionStrategy interface
    # ──────────────────────────────────────────────────────────────────

    def compress_gaussian(self, data: GaussianData) -> GaussianData:
        self._original_count = data.num_gaussians
        N = self._original_count

        # 1. Compute importance
        if self.importance_mode == "render":
            importance = self._compute_importance_render(data)
        else:
            importance = self._compute_importance_parameter(data)

        # 2. Volume-weighted significance score
        scores = self._calculate_v_imp_score(data, importance)

        # 3. Build mask
        effective_prune = self.prune_percent * self.prune_decay
        keep = self._build_mask(scores, effective_prune)
        self._index_map = self._build_index_map(keep)

        n_kept = int(keep.sum())
        n_pruned = N - n_kept

        assert n_kept > 0, (
            "LightGaussian pruning would remove ALL Gaussians — "
            "try lowering prune_percent"
        )

        # 4. Apply mask to all arrays
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

        data.validate()

        # Consistent index map
        assert (self._index_map >= 0).sum() == n_kept, (
            f"index_map inconsistency: {(self._index_map >= 0).sum()} vs {n_kept}"
        )

        self._stats.extra = {
            "original_gaussians": self._original_count,
            "kept_gaussians": n_kept,
            "pruned_gaussians": n_pruned,
            "pruned_pct": round(n_pruned / self._original_count * 100, 2),
            "importance_mode": self.importance_mode,
            "v_pow": self.v_pow,
            "prune_percent": self.prune_percent,
            "deformation_weight": self.deformation_weight,
        }

        return data

    def decompress_gaussian(
        self, data: GaussianData, metadata: Dict[str, Any]
    ) -> GaussianData:
        """Pruning is lossy — decompression is a no-op."""
        return data

    # ──────────────────────────────────────────────────────────────────
    #  Metadata
    # ──────────────────────────────────────────────────────────────────

    def get_metadata(self) -> Dict[str, Any]:
        meta = {
            "strategy": self.name,
            "params": self.params,
            "original_count": self._original_count,
            "kept_count": (
                int((self._index_map >= 0).sum())
                if self._index_map is not None
                else 0
            ),
        }
        if self._index_map is not None:
            meta["index_map"] = self._index_map.tolist()
        return meta

    @property
    def index_map(self) -> Optional[np.ndarray]:
        """``original_idx → new_idx`` or ``-1`` if pruned."""
        return self._index_map

    def validate_index_map(
        self, original_count: int, pruned_count: int
    ) -> bool:
        """Verify index_map consistency."""
        if self._index_map is None:
            return False
        if len(self._index_map) != original_count:
            return False
        kept = (self._index_map >= 0).sum()
        if kept != pruned_count:
            return False
        new_indices = self._index_map[self._index_map >= 0]
        expected = np.arange(kept, dtype=np.int32)
        return np.array_equal(new_indices, expected)

    # ══════════════════════════════════════════════════════════════════
    #  Private helpers — camera loading & model building (render mode)
    # ══════════════════════════════════════════════════════════════════

    def _load_cameras(self) -> list:
        """Load camera metadata without GT images.

        Tries (in order):
        1. ``cameras.json`` in ``model_path``
        2. Dataset loading via ``sceneLoadTypeCallbacks`` with ``source_path``

        Returns a list of dicts or CameraInfo-like objects with at
        minimum: R, T, FovX, FovY, width, height, time.
        """
        cameras: list = []

        # Strategy A: cameras.json  (written during training)
        if self.model_path:
            cj_path = os.path.join(self.model_path, "cameras.json")
            if os.path.exists(cj_path):
                cameras = self._parse_cameras_json(cj_path)
                if cameras:
                    return cameras

        # Strategy B: load from source_path via dataset readers
        if self.source_path:
            cameras = self._load_cameras_from_source()

        return cameras

    @staticmethod
    def _parse_cameras_json(path: str) -> list:
        """Parse a cameras.json written by the training script.

        Each entry has: id, img_name, width, height, position [3],
        rotation [4x4-ish or 3x3], fx, fy, FovX, FovY.
        """
        import torch
        from utils.graphics_utils import getWorld2View2, getProjectionMatrix

        with open(path, "r") as f:
            cam_list = json.load(f)

        cameras = []
        for entry in cam_list:
            try:
                R = np.array(entry["rotation"])
                T = np.array(entry["position"])
                FovX = entry.get("FovX") or entry.get("fovx") or entry.get("FOV_x")
                FovY = entry.get("FovY") or entry.get("fovy") or entry.get("FOV_y")
                width = entry.get("width", 800)
                height = entry.get("height", 800)
                time_val = entry.get("time", 0.0)

                if FovX is None or FovY is None:
                    # Try to derive from fx/fy
                    fx = entry.get("fx", None)
                    fy = entry.get("fy", None)
                    if fx and fy:
                        FovX = 2.0 * math.atan(width / (2.0 * fx))
                        FovY = 2.0 * math.atan(height / (2.0 * fy))
                    else:
                        continue

                # Compute transforms
                wvt = torch.tensor(
                    getWorld2View2(R, T)
                ).transpose(0, 1).float()
                proj = getProjectionMatrix(
                    znear=0.01, zfar=100.0, fovX=FovX, fovY=FovY
                ).transpose(0, 1).float()
                fpt = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)

                cameras.append({
                    "R": R, "T": T,
                    "FovX": FovX, "FovY": FovY,
                    "width": width, "height": height,
                    "time": time_val,
                    "world_view_transform": wvt,
                    "full_proj_transform": fpt,
                })
            except Exception:
                continue

        return cameras

    def _load_cameras_from_source(self) -> list:
        """Load camera info from source_path via dataset readers."""
        try:
            import torch
            from scene.dataset_readers import sceneLoadTypeCallbacks
            from utils.graphics_utils import getWorld2View2, getProjectionMatrix

            # Detect dataset type
            source = self.source_path
            if os.path.exists(os.path.join(source, "sparse")):
                scene_info = sceneLoadTypeCallbacks["Colmap"](
                    source, None, False
                )
            elif os.path.exists(os.path.join(source, "transforms_train.json")):
                scene_info = sceneLoadTypeCallbacks["Blender"](
                    source, False, 1
                )
            elif os.path.exists(os.path.join(source, "dataset.json")):
                scene_info = sceneLoadTypeCallbacks["nerfies"](
                    source, False, 1
                )
            elif os.path.exists(os.path.join(source, "poses_bounds.npy")):
                scene_info = sceneLoadTypeCallbacks["dynerf"](
                    source, False, 1
                )
            else:
                return []

            cameras = []
            cam_infos = scene_info.train_cameras
            for ci in cam_infos:
                R = ci.R
                T = ci.T
                FovX = ci.FovX
                FovY = ci.FovY
                width = ci.width
                height = ci.height
                time_val = getattr(ci, "time", 0.0)

                wvt = torch.tensor(
                    getWorld2View2(R, T)
                ).transpose(0, 1).float()
                proj = getProjectionMatrix(
                    znear=0.01, zfar=100.0, fovX=FovX, fovY=FovY
                ).transpose(0, 1).float()
                fpt = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)

                cameras.append({
                    "R": R, "T": T,
                    "FovX": FovX, "FovY": FovY,
                    "width": width, "height": height,
                    "time": time_val,
                    "world_view_transform": wvt,
                    "full_proj_transform": fpt,
                })

            return cameras

        except Exception as e:
            print(f"  [LightGaussian] Failed to load cameras from source: {e}")
            return []

    @staticmethod
    def _caminfo_to_minicam(cam_info: dict, time_override: float):
        """Build a MiniCam from a camera info dict."""
        from scene.cameras import MiniCam

        return MiniCam(
            width=cam_info["width"],
            height=cam_info["height"],
            fovy=cam_info["FovY"],
            fovx=cam_info["FovX"],
            znear=0.01,
            zfar=100.0,
            world_view_transform=cam_info["world_view_transform"],
            full_proj_transform=cam_info["full_proj_transform"],
            time=time_override,
        )

    def _build_gaussian_model(self, data: GaussianData):
        """Create a GaussianModel on GPU from GaussianData (numpy).

        Also loads the deformation network from disk.
        """
        import torch
        import torch.nn as nn
        from scene.gaussian_model import GaussianModel
        from scene.deformation import deform_network

        # Build hyper_args (same pattern as decompress.py)
        hyper_args = self._make_hyper_args()

        # Infer architecture from state_dict if model_path available
        deform_state = None
        if self.model_path:
            iteration = self.iteration
            if iteration == -1:
                from utils.system_utils import searchForMaxIteration
                iteration = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            iter_dir = os.path.join(
                self.model_path, "point_cloud", f"iteration_{iteration}"
            )
            deform_path = os.path.join(iter_dir, "deformation.pth")
            if os.path.exists(deform_path):
                deform_state = torch.load(
                    deform_path, map_location="cpu", weights_only=False
                )

        if deform_state is not None:
            self._infer_hyper_from_state_dict(deform_state, hyper_args)

        gaussians = GaussianModel(sh_degree=data.sh_degree, args=hyper_args)
        gaussians.active_sh_degree = data.active_sh_degree

        # Populate parameters
        gaussians._xyz = nn.Parameter(
            torch.tensor(data.xyz, dtype=torch.float32, device="cuda")
        )
        gaussians._features_dc = nn.Parameter(
            torch.tensor(data.features_dc, dtype=torch.float32, device="cuda")
        )
        gaussians._features_rest = nn.Parameter(
            torch.tensor(data.features_rest, dtype=torch.float32, device="cuda")
        )
        gaussians._opacity = nn.Parameter(
            torch.tensor(data.opacity, dtype=torch.float32, device="cuda")
        )
        gaussians._scaling = nn.Parameter(
            torch.tensor(data.scaling, dtype=torch.float32, device="cuda")
        )
        gaussians._rotation = nn.Parameter(
            torch.tensor(data.rotation, dtype=torch.float32, device="cuda")
        )

        # Deformation network
        if deform_state is not None:
            gaussians._deformation.load_state_dict(deform_state)
        gaussians._deformation = gaussians._deformation.to("cuda")
        gaussians._deformation.eval()

        N = data.num_gaussians
        gaussians._deformation_table = torch.ones(N, dtype=torch.bool, device="cuda")

        return gaussians

    def _make_hyper_args(self) -> Namespace:
        """Build hyper-parameter namespace (same defaults as decompress.py)."""
        defaults = Namespace(
            net_width=64,
            timebase_pe=4,
            defor_depth=1,
            posebase_pe=10,
            scale_rotation_pe=2,
            opacity_pe=2,
            timenet_width=64,
            timenet_output=32,
            bounds=1.6,
            plane_tv_weight=0.0001,
            time_smoothness_weight=0.01,
            l1_time_planes=0.0001,
            kplanes_config={
                "grid_dimensions": 2,
                "input_coordinate_dim": 4,
                "output_coordinate_dim": 16,
                "resolution": [64, 64, 64, 25],
            },
            multires=[1, 2],
            no_grid=False,
            no_dx=False,
            no_ds=False,
            no_dr=False,
            no_do=False,
            no_dshs=False,
            apply_rotation=False,
            empty_voxel=False,
            grid_pe=0,
            static_mlp=False,
        )

        if self.configs:
            try:
                from decompress import _read_py_config
                cfg = _read_py_config(self.configs)
                hidden = cfg.get("ModelHiddenParams", {})
                for key, val in hidden.items():
                    setattr(defaults, key, val)
            except Exception:
                pass

        return defaults

    @staticmethod
    def _infer_hyper_from_state_dict(state_dict: dict, hyper_args) -> None:
        """Infer deformation net architecture from state_dict shapes.

        Mirrors the logic in decompress.py ``_infer_hyperparams_from_state_dict``.
        """
        import re

        sd = state_dict

        # net_width
        key = "deformation_net.feature_out.0.weight"
        if key in sd:
            inferred_w = sd[key].shape[0]
            if inferred_w != hyper_args.net_width:
                hyper_args.net_width = inferred_w

        # defor_depth
        feat_out_keys = sorted(
            [k for k in sd if k.startswith("deformation_net.feature_out.") and k.endswith(".weight")]
        )
        n_linears = len(feat_out_keys)
        if n_linears > 1:
            hyper_args.defor_depth = n_linears

        # kplanes time resolution
        for k, v in sd.items():
            m = re.match(r"deformation_net\.grid\.grids\.(\d+)\.(\d+)", k)
            if m and int(m.group(2)) in (2, 4, 5):
                time_res = v.shape[2]
                current_res = hyper_args.kplanes_config.get(
                    "resolution", [64, 64, 64, 25]
                )
                if current_res[3] != time_res:
                    current_res[3] = time_res
                    hyper_args.kplanes_config["resolution"] = current_res
                break

        # multires
        res_indices = set()
        for k in sd:
            m = re.match(r"deformation_net\.grid\.grids\.(\d+)\.", k)
            if m:
                res_indices.add(int(m.group(1)))
        if res_indices:
            n_res = len(res_indices)
            if n_res != len(hyper_args.multires):
                hyper_args.multires = list(range(1, n_res + 1))

        # deformation heads
        has_opacity = any("opacity_deform" in k for k in sd)
        has_scales = any("scales_deform" in k for k in sd)
        has_rotations = any("rotations_deform" in k for k in sd)
        has_shs = any("shs_deform" in k for k in sd)

        for attr, has_head in [
            ("no_do", not has_opacity),
            ("no_ds", not has_scales),
            ("no_dr", not has_rotations),
            ("no_dshs", not has_shs),
        ]:
            current = getattr(hyper_args, attr, False)
            if current != has_head:
                setattr(hyper_args, attr, has_head)
