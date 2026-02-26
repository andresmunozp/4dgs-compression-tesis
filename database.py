# This script is based on an original implementation by True Price.
# Created by liminghao — patched for NumPy 2.x / Python 3 by you :)
import sys
import os
import argparse
import numpy as np
import sqlite3

def array_to_blob(array: np.ndarray) -> memoryview:
    # NumPy 2.x compatible: tobytes() y envolver en memoryview para SQLite
    return memoryview(array.tobytes())

def blob_to_array(blob: bytes, dtype, shape=(-1,)) -> np.ndarray:
    # frombuffer no copia, es la alternativa recomendada a fromstring
    return np.frombuffer(blob, dtype=dtype).reshape(*shape)

class COLMAPDatabase(sqlite3.Connection):
    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        # prior_focal_length: usar 1/0 para SQLite en lugar de True/False
        cursor = self.execute(
            """
            UPDATE cameras
            SET model=?, width=?, height=?, params=?, prior_focal_length=1
            WHERE camera_id=?
            """,
            (model, width, height, array_to_blob(params), camera_id),
        )
        return cursor.lastrowid

def camTodatabase():
    camModelDict = {
        'SIMPLE_PINHOLE': 0, 'PINHOLE': 1,
        'SIMPLE_RADIAL': 2, 'RADIAL': 3, 'OPENCV': 4,
        'FULL_OPENCV': 5, 'SIMPLE_RADIAL_FISHEYE': 6,
        'RADIAL_FISHEYE': 7, 'OPENCV_FISHEYE': 8,
        'FOV': 9, 'THIN_PRISM_FISHEYE': 10
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", type=str, default="database.db")
    parser.add_argument("--txt_path", type=str, default="colmap/sparse_cameras.txt")
    args = parser.parse_args()

    if not os.path.exists(args.database_path):
        print("ERROR: database path doesn't exist — please check database.db.")
        return
    if not os.path.exists(args.txt_path):
        print("ERROR: txt_path doesn't exist — please check cameras.txt.")
        return

    db = COLMAPDatabase.connect(args.database_path)

    idList, modelList, widthList, heightList, paramsList = [], [], [], [], []

    with open(args.txt_path, "r", encoding="utf-8") as cam:
        for line in cam:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Formato COLMAP cameras.txt:
            # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS...
            cameraId = int(parts[0])
            cameraModel = camModelDict[parts[1]]
            width = int(parts[2])
            height = int(parts[3])

            # El resto son los parámetros (k), no asumas 8 exactamente:
            # SIMPLE_PINHOLE: 3; PINHOLE: 4; SIMPLE_RADIAL: 4; RADIAL: 5; OPENCV: 8; etc.
            param_values = np.array(parts[4:], dtype=np.float64)

            idList.append(cameraId)
            modelList.append(cameraModel)
            widthList.append(width)
            heightList.append(height)
            paramsList.append(param_values)

            db.update_camera(cameraModel, width, height, param_values, cameraId)

    db.commit()

    # Verificación rápida
    rows = db.execute("SELECT camera_id, model, width, height, params FROM cameras ORDER BY camera_id")
    rows = list(rows)
    id2row = {r[0]: r for r in rows}

    for i, cam_id in enumerate(idList):
        camera_id, model, width, height, params_blob = id2row[cam_id]
        params = blob_to_array(params_blob, np.float64)
        assert camera_id == idList[i]
        assert model == modelList[i] and width == widthList[i] and height == heightList[i]
        # allclose permite pequeñas diferencias de precisión
        assert np.allclose(params, paramsList[i]), f"Params mismatch for camera {cam_id}"

    db.close()

if __name__ == "__main__":
    camTodatabase()
