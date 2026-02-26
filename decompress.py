#!/usr/bin/env python3
"""
decompress.py — Decode a compressed 4DGS archive and export PLY sequences.

Separates **decode time** (decompression of the binary archive) from
**export time** (running the deformation network + writing per-frame PLYs).

Usage
-----
    python decompress.py \
        --input compressed_output/ \
        --output decompressed_output/ \
        --source_path data/dynerf/coffee_martini \
        --configs arguments/dynerf/coffee_martini.py \
        --num_frames 300 \
        --format ply
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import time
from argparse import ArgumentParser

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compression.base import DeformationData, GaussianData
from compression.chunker import ModelAssembler
from compression.pipeline import CompressionPipeline
from compression.serializer import ModelSerializer


# ── PLY export helpers (from export_perframe_3DGS.py) ─────────────────────

def _construct_ply_attributes(features_dc_shape, features_rest_shape, scaling_shape, rotation_shape):
    l = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(features_dc_shape[1] * features_dc_shape[2]):
        l.append(f"f_dc_{i}")
    for i in range(features_rest_shape[1] * features_rest_shape[2]):
        l.append(f"f_rest_{i}")
    l.append("opacity")
    for i in range(scaling_shape[1]):
        l.append(f"scale_{i}")
    for i in range(rotation_shape[1]):
        l.append(f"rot_{i}")
    return l


def _write_ply(path, xyz, features_dc, features_rest, opacity, scaling, rotation):
    """Write a standard 3DGS-compatible PLY file."""
    from plyfile import PlyData, PlyElement

    normals = np.zeros_like(xyz)

    # features_dc: (N, 1, 3) → transpose → (N, 3, 1) → flatten → (N, 3)
    f_dc = features_dc.transpose(0, 2, 1).reshape(xyz.shape[0], -1)
    f_rest = features_rest.transpose(0, 2, 1).reshape(xyz.shape[0], -1)

    attrs = _construct_ply_attributes(
        features_dc.shape, features_rest.shape, scaling.shape, rotation.shape
    )
    dtype_full = [(a, "f4") for a in attrs]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    data = np.concatenate([xyz, normals, f_dc, f_rest, opacity, scaling, rotation], axis=1)
    elements[:] = list(map(tuple, data))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser(description="Decompress a 4DGS archive and export PLY sequences")
    parser.add_argument("--input", required=True,
                        help="Directory with .4dgsc chunks or path to single .4dgs file")
    parser.add_argument("--output", default="decompressed_output",
                        help="Output directory for PLY files")
    parser.add_argument("--format", choices=["ply"], default="ply",
                        help="Export format")
    parser.add_argument("--num_frames", type=int, default=300,
                        help="Number of frames to export")

    # Model hyperparams needed to rebuild the deformation network
    parser.add_argument("--source_path", type=str, default=None,
                        help="Source data path (needed only for scene/camera loading)")
    parser.add_argument("--configs", type=str, default=None,
                        help="Config file for model hidden params (e.g. arguments/dynerf/default.py)")
    parser.add_argument("--sh_degree", type=int, default=3,
                        help="SH degree (override if not in manifest)")
    parser.add_argument("--no_verify", action="store_true",
                        help="Skip checksum verification")
    parser.add_argument("--compression_config", type=str, default=None,
                        help="Compression pipeline YAML (if not embedded in archive)")

    args = parser.parse_args()

    # ── 1. Reassemble chunks ──────────────────────────────────────────
    print("=" * 60)
    print("PHASE 1: Chunk reassembly")
    print("=" * 60)
    t_assemble_start = time.perf_counter()

    if os.path.isfile(args.input) and args.input.endswith(".4dgs"):
        with open(args.input, "rb") as f:
            archive = f.read()
        print(f"  Loaded single archive: {len(archive) / 1e6:.2f} MB")
    else:
        archive = ModelAssembler.assemble_from_dir(
            args.input, verify=not args.no_verify
        )
        print(f"  Assembled from chunks: {len(archive) / 1e6:.2f} MB")

    t_assemble = time.perf_counter() - t_assemble_start
    print(f"  Assembly time: {t_assemble:.3f}s")

    # ── 2. Decode (decompress) ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: Decode (decompression)")
    print("=" * 60)
    t_decode_start = time.perf_counter()

    # Read manifest first
    manifest = ModelSerializer.read_manifest_only(archive)
    pipeline_config = manifest.get("pipeline_config", {})

    # If user provides an external config, use it, else use the embedded one
    if args.compression_config:
        with open(args.compression_config, "r") as f:
            pipeline_config = yaml.safe_load(f)

    # Build pipeline
    pipeline = CompressionPipeline.from_config(pipeline_config)

    # Decompress
    gaussian, deformation, manifest = pipeline.decompress_from_archive(
        archive, verify_checksums=not args.no_verify
    )

    t_decode = time.perf_counter() - t_decode_start
    print(f"  Decoded {gaussian.num_gaussians} Gaussians")
    print(f"  SH degree: {gaussian.sh_degree}")
    print(f"  Decode time: {t_decode:.3f}s")

    # ── 3. Export per-frame PLYs ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3: Export per-frame PLYs (deformation bake-out)")
    print("=" * 60)
    t_export_start = time.perf_counter()

    os.makedirs(args.output, exist_ok=True)

    # Rebuild deformation network on GPU
    _export_with_deformation(
        gaussian, deformation, args, manifest
    )

    t_export = time.perf_counter() - t_export_start
    print(f"  Export time: {t_export:.3f}s  ({args.num_frames} frames)")

    # ── 4. Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    print(f"  Chunk assembly:  {t_assemble:.3f}s")
    print(f"  Decode time:     {t_decode:.3f}s")
    print(f"  Export time:     {t_export:.3f}s")
    print(f"  Total time:      {t_assemble + t_decode + t_export:.3f}s")
    print(f"  Decode-only:     {t_decode:.3f}s  (network-relevant latency)")
    print(f"  Decode+Export:   {t_decode + t_export:.3f}s  (end-to-end reconstruction)")

    # Write timing report
    timing_report = {
        "assemble_time_s": round(t_assemble, 4),
        "decode_time_s": round(t_decode, 4),
        "export_time_s": round(t_export, 4),
        "total_time_s": round(t_assemble + t_decode + t_export, 4),
        "num_gaussians": gaussian.num_gaussians,
        "num_frames": args.num_frames,
        "sh_degree": gaussian.sh_degree,
        "archive_size_bytes": len(archive),
    }
    report_path = os.path.join(args.output, "decompression_report.json")
    with open(report_path, "w") as f:
        json.dump(timing_report, f, indent=2)
    print(f"\n  Report saved to {report_path}")


def _export_with_deformation(
    gaussian: GaussianData,
    deformation: DeformationData,
    args,
    manifest: dict,
):
    """Rebuild the deformation network, run it for each frame, write PLYs."""
    from scene.deformation import deform_network
    from tqdm import tqdm

    # Load model hyperparams from config, then override with state_dict inference
    hyper_args = _load_hyperparams(args)
    _infer_hyperparams_from_state_dict(deformation.state_dict, hyper_args)

    # Build deformation network with same arch
    deform_net = deform_network(hyper_args)
    deform_net.load_state_dict(deformation.state_dict)
    # NOTE: do NOT call set_aabb here — the training AABB is already
    # stored in the state_dict (deformation_net.grid.aabb) and was
    # restored by load_state_dict.  Overwriting it with the decompressed
    # Gaussian bounds would break grid coordinate normalisation.
    deform_net = deform_net.to("cuda")
    deform_net.eval()

    # Convert Gaussian data to torch tensors on GPU
    means3D = torch.tensor(gaussian.xyz, dtype=torch.float32, device="cuda")
    opacity = torch.tensor(gaussian.opacity, dtype=torch.float32, device="cuda")
    scaling = torch.tensor(gaussian.scaling, dtype=torch.float32, device="cuda")
    rotation = torch.tensor(gaussian.rotation, dtype=torch.float32, device="cuda")

    # Reconstruct SH features: concat dc + rest → (N, total_coeffs, 3)
    features_dc = torch.tensor(gaussian.features_dc, dtype=torch.float32, device="cuda")
    features_rest = torch.tensor(gaussian.features_rest, dtype=torch.float32, device="cuda")
    shs = torch.cat([features_dc, features_rest], dim=1)  # (N, K, 3)

    N = gaussian.num_gaussians
    n_dc = gaussian.features_dc.shape[1]

    ply_dir = os.path.join(args.output, "gaussian_pertimestamp")
    os.makedirs(ply_dir, exist_ok=True)

    with torch.no_grad():
        for frame_idx in tqdm(range(args.num_frames), desc="Exporting frames"):
            t = frame_idx / max(args.num_frames - 1, 1)
            time_tensor = torch.tensor([[t]], dtype=torch.float32, device="cuda").expand(N, 1)

            means3D_d, scales_d, rots_d, opacity_d, shs_d = deform_net(
                means3D, scaling, rotation, opacity, shs, time_tensor
            )

            # Write PLY with deformed params
            _write_ply(
                path=os.path.join(ply_dir, f"time_{frame_idx:05d}.ply"),
                xyz=means3D_d.cpu().numpy(),
                features_dc=shs_d[:, :n_dc, :].cpu().numpy(),
                features_rest=shs_d[:, n_dc:, :].cpu().numpy(),
                opacity=opacity.cpu().numpy(),
                scaling=scales_d.cpu().numpy(),
                rotation=rots_d.cpu().numpy(),
            )

    print(f"  Exported {args.num_frames} PLY files to {ply_dir}/")


def _infer_hyperparams_from_state_dict(state_dict: dict, hyper_args) -> None:
    """Override hyper_args with values inferred directly from state_dict shapes.

    This is the most reliable method — the weight shapes tell us exactly
    what architecture was used during training, regardless of config parsing.
    """
    import re

    sd = state_dict

    # ── net_width from feature_out.0.weight ──
    key = "deformation_net.feature_out.0.weight"
    if key in sd:
        inferred_w = sd[key].shape[0]
        if inferred_w != hyper_args.net_width:
            print(f"  [infer] net_width: {hyper_args.net_width} → {inferred_w}")
            hyper_args.net_width = inferred_w

    # ── defor_depth from counting feature_out layers ──
    # feature_out is Sequential: Linear, [ReLU, Linear] * (D-1)
    # So weight keys are feature_out.0.weight, feature_out.2.weight, feature_out.4.weight, ...
    feat_out_keys = sorted(
        [k for k in sd if k.startswith("deformation_net.feature_out.") and k.endswith(".weight")]
    )
    n_linears = len(feat_out_keys)  # = D (first linear + D-1 extra)
    if n_linears >= 1:
        inferred_depth = n_linears - 1  # D-1 extra layers
        # But defor_depth in code is actually D (passed as D to Deformation, then D-1 loop iterations)
        # deform_network passes defor_depth to Deformation(D=defor_depth, ...)
        # create_net does: for i in range(self.D - 1): => D-1 extra layers
        # So n_linears = 1 + (D-1) = D => defor_depth = n_linears - 1 when D>=1, or 0 means 1 linear only
        # But when defor_depth=0, D=0, range(D-1)=range(-1)=empty => only 1 linear. n_linears=1 => defor_depth=0 ✓
        # When defor_depth=1, D=1, range(0)=empty => only 1 linear. n_linears=1 => inferred=0. Hmm.
        # Let me check: defor_depth is passed as D to Deformation. for i in range(self.D-1): adds D-1 extra.
        # So total linears in feature_out = 1 + max(0, D-1) = D when D>=1, or 1 when D=0.
        # So defor_depth = max(n_linears, 1) if n_linears > 1, else n_linears-1... 
        # Actually: n_linears = 1 + max(0, D-1). So D = n_linears if n_linears > 1, else could be 0 or 1.
        # For n_linears=1: D is 0 or 1 (both produce 1 linear). Check feature_out input dim to distinguish.
        if n_linears == 1:
            # Could be D=0 or D=1; both produce exactly 1 linear.
            # D=0 in code: range(-1) = empty, so feature_out = [Linear(in, W)] = 1 linear
            # D=1 in code: range(0) = empty, so feature_out = [Linear(in, W)] = 1 linear
            # Can't distinguish, but defor_depth=0 is the dynerf default. Check current value.
            # If config already set it, keep it; otherwise use 0 (dynerf default).
            pass
        else:
            inferred_depth = n_linears  # D = n_linears when n_linears > 1
            if inferred_depth != hyper_args.defor_depth:
                print(f"  [infer] defor_depth: {hyper_args.defor_depth} → {inferred_depth}")
                hyper_args.defor_depth = inferred_depth

    # ── kplanes resolution (time dimension) from grid shapes ──
    # Grid keys like deformation_net.grid.grids.{res_idx}.{plane_idx}
    # Planes involving time (indices 2, 4, 5) have shape [1, C, T, S]
    # where T is the time resolution.
    time_res = None
    for k, v in sd.items():
        m = re.match(r"deformation_net\.grid\.grids\.(\d+)\.(\d+)", k)
        if m and int(m.group(2)) in (2, 4, 5):  # time-involving planes
            time_res = v.shape[2]  # T dimension
            break

    if time_res is not None:
        current_res = hyper_args.kplanes_config.get("resolution", [64, 64, 64, 25])
        if current_res[3] != time_res:
            print(f"  [infer] kplanes resolution time: {current_res[3]} → {time_res}")
            current_res[3] = time_res
            hyper_args.kplanes_config["resolution"] = current_res

    # ── multires from number of resolution groups ──
    res_indices = set()
    for k in sd:
        m = re.match(r"deformation_net\.grid\.grids\.(\d+)\.", k)
        if m:
            res_indices.add(int(m.group(1)))
    if res_indices:
        n_res = len(res_indices)
        if n_res != len(hyper_args.multires):
            inferred_multires = list(range(1, n_res + 1))
            print(f"  [infer] multires: {hyper_args.multires} → {inferred_multires}")
            hyper_args.multires = inferred_multires

    # ── no_do, no_ds, no_dr, no_dshs from presence of deform heads ──
    has_opacity = any("opacity_deform" in k for k in sd)
    has_scales = any("scales_deform" in k for k in sd)
    has_rotations = any("rotations_deform" in k for k in sd)
    has_shs = any("shs_deform" in k for k in sd)

    for attr, has_head, label in [
        ("no_do", not has_opacity, "opacity"),
        ("no_ds", not has_scales, "scales"),
        ("no_dr", not has_rotations, "rotations"),
        ("no_dshs", not has_shs, "shs"),
    ]:
        current = getattr(hyper_args, attr, False)
        if current != (not has_head):
            inferred = not has_head
            print(f"  [infer] {attr}: {current} → {inferred} ({label} head {'absent' if inferred else 'present'})")
            setattr(hyper_args, attr, inferred)


def _read_py_config(config_path: str) -> dict:
    """Read a .py config file with _base_ inheritance (no mmcv needed).

    Supports the pattern used in arguments/dynerf/*.py:
        _base_ = './default.py'
        ModelHiddenParams = dict(...)
    """
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        return {}

    # Execute the file in a namespace
    ns: dict = {}
    with open(config_path, "r") as f:
        code = f.read()

    # Resolve _base_ first
    base_ns: dict = {}
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("_base_"):
            # e.g.  _base_ = './default.py'
            try:
                _, rhs = stripped.split("=", 1)
                base_rel = rhs.strip().strip("'\"")
                base_path = os.path.normpath(
                    os.path.join(os.path.dirname(config_path), base_rel)
                )
                base_ns = _read_py_config(base_path)
            except Exception:
                pass
            break

    try:
        exec(compile(code, config_path, "exec"), ns)
    except Exception:
        return base_ns

    # Merge: base values overridden by current file
    merged: dict = {}
    for key in ("ModelHiddenParams", "OptimizationParams"):
        base_dict = base_ns.get(key, {})
        cur_dict = ns.get(key, {})
        if isinstance(base_dict, dict) and isinstance(cur_dict, dict):
            merged[key] = {**base_dict, **cur_dict}
        elif cur_dict:
            merged[key] = cur_dict
        elif base_dict:
            merged[key] = base_dict

    return merged


def _load_hyperparams(args):
    """Build a namespace of hyperparams for deform_network construction."""
    from argparse import Namespace

    # Defaults matching arguments/__init__.py ModelHiddenParams
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

    if args.configs:
        loaded = False
        # Try mmcv first (handles advanced inheritance)
        try:
            import mmcv
            config = mmcv.Config.fromfile(args.configs)
            for key, val in config.items():
                if hasattr(defaults, key):
                    setattr(defaults, key, val)
            loaded = True
        except Exception:
            pass

        # Fallback: parse .py config directly
        if not loaded:
            try:
                cfg = _read_py_config(args.configs)
                hidden = cfg.get("ModelHiddenParams", {})
                for key, val in hidden.items():
                    setattr(defaults, key, val)
                if hidden:
                    loaded = True
                    print(f"  Loaded config from {args.configs} (direct parse)")
            except Exception as e:
                print(f"  Warning: could not load config file '{args.configs}': {e}")

        if not loaded:
            print(f"  Warning: could not load '{args.configs}', using defaults")

    return defaults


if __name__ == "__main__":
    main()
