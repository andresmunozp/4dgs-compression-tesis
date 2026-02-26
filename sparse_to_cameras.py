#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convierte COLMAP sparse/0 (TXT) -> camera.json compatible con SIBR/NeRF.
- Lee cameras.txt e images.txt
- Convierte poses (COLMAP world->cam) a cam->world
- Escribe camera.json con intrínsecos y frames

Uso:
python colmap_sparse_to_camera_json.py \
  --sparse_dir "C:/.../sparse/0_txt" \
  --out_json   "C:/.../coffee_martini/camera.json" \
  --image_prefix "images"    # opcional
"""

import os, json, math
import argparse
import numpy as np

def qvec2rotmat(qw, qx, qy, qz):
    # Normaliza y crea matriz de rotación (convención COLMAP)
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=np.float64)
    return R

def parse_cameras_txt(path):
    cams = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): 
                continue
            # CAM_ID, MODEL, WIDTH, HEIGHT, PARAMS...
            toks = line.split()
            cam_id = int(toks[0])
            model = toks[1]
            w = int(toks[2]); h = int(toks[3])
            params = list(map(float, toks[4:]))

            # Deriva intrínsecos
            if model in ('PINHOLE', 'OPENCV', 'OPENCV_FISHEYE', 'FULL_OPENCV'):
                # fx, fy, cx, cy primeros 4
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            elif model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL'):
                # f, cx, cy
                f, cx, cy = params[0], params[1], params[2]
                fx, fy = f, f
            else:
                # fallback: intenta al menos algo razonable
                if len(params) >= 4:
                    fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                elif len(params) >= 3:
                    fx, fy, cx, cy = params[0], params[0], params[1], params[2]
                else:
                    fx = fy = max(w, h) * 1.2
                    cx, cy = w/2.0, h/2.0

            cams[cam_id] = {
                'model': model, 'w': w, 'h': h,
                'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy
            }
    return cams

def parse_images_txt(path):
    # Devuelve lista de (image_id, qw,qx,qy,qz, tx,ty,tz, cam_id, name)
    imgs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            toks = line.split()
            if len(toks) < 10:
                # La siguiente línea suele ser features 2D; la saltamos
                continue
            image_id = int(toks[0])
            qw, qx, qy, qz = map(float, toks[1:5])
            tx, ty, tz = map(float, toks[5:8])
            cam_id = int(toks[8])
            name = " ".join(toks[9:])  # por si hay espacios (raro pero posible)
            imgs.append((image_id, qw,qx,qy,qz, tx,ty,tz, cam_id, name))
    return imgs

def pose_cam_to_world_from_colmap(qw,qx,qy,qz, tx,ty,tz):
    """
    COLMAP en images.txt entrega R,t que llevan puntos de MUNDO a CÁMARA:
        X_cam = R * X_world + t
    Para obtener CÁMARA->MUNDO:
        R_c2w = R^T
        t_c2w = -R^T * t
    """
    R = qvec2rotmat(qw,qx,qy,qz)
    t = np.array([tx,ty,tz], dtype=np.float64).reshape(3,1)
    R_c2w = R.T
    t_c2w = (-R.T @ t).reshape(3)
    # 4x4
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R_c2w
    T[:3, 3] = t_c2w
    return T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sparse_dir', required=True,
                    help='Carpeta con cameras.txt, images.txt (TXT export de COLMAP).')
    ap.add_argument('--out_json', required=True, help='Ruta de salida camera.json')
    ap.add_argument('--image_prefix', default='', 
                    help='Prefijo para file_path (p.ej. "images"). Se ignora si vacío.')
    args = ap.parse_args()

    cam_path = os.path.join(args.sparse_dir, 'cameras.txt')
    img_path = os.path.join(args.sparse_dir, 'images.txt')

    if not os.path.isfile(cam_path) or not os.path.isfile(img_path):
        raise FileNotFoundError('No encuentro cameras.txt o images.txt en: ' + args.sparse_dir)

    cams = parse_cameras_txt(cam_path)
    imgs = parse_images_txt(img_path)

    frames = []
    flx = fly = cx = cy = w = h = None

    for (image_id, qw,qx,qy,qz, tx,ty,tz, cam_id, name) in imgs:
        if cam_id not in cams:
            continue
        C = cams[cam_id]
        T_c2w = pose_cam_to_world_from_colmap(qw,qx,qy,qz, tx,ty,tz)

        # Guarda intrínsecos del último (o del primero; SIBR/NeRF suelen asumir 1 cámara o varias iguales)
        flx, fly, cx, cy, w, h = C['fx'], C['fy'], C['cx'], C['cy'], C['w'], C['h']

        # file_path relativo (opcionalmente con prefijo)
        if args.image_prefix:
            fp = os.path.join(args.image_prefix, name).replace('\\','/')
        else:
            fp = name.replace('\\','/')

        frames.append({
            "file_path": fp,
            "transform_matrix": T_c2w.tolist()
        })

    if not frames:
        raise RuntimeError("No se generaron frames. ¿images.txt vacío?")

    out = {
        "w": w, "h": h,
        "fl_x": flx, "fl_y": fly,
        "cx": cx, "cy": cy,
        "frames": frames
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"Listo -> {args.out_json}  (frames: {len(frames)})")

if __name__ == '__main__':
    main()
