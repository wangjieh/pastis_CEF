import json
import os
import numpy as np
import torch

DATA_DIR = r"G:\PASTIE_elev\PASTIS-R\DATA_S2"
META_PATH = r"C:\Users\23948\Desktop\pastis_data\metadata.geojson"
OUT_DIR = r"G:\PASTIE_elev\PASTIS-R\DATA_S2_PT"
os.makedirs(OUT_DIR, exist_ok=True)

with open(META_PATH, encoding="utf-8") as f:
    meta = json.load(f)

def impute(img):
    """10波段 -> 13标准波段"""
    return torch.stack([
        img[0],  # B1 <- B2
        img[0],  # B2
        img[1],  # B3
        img[2],  # B4
        img[3],  # B5
        img[4],  # B6
        img[5],  # B7
        img[6],  # B8
        img[7],  # B8A
        img[7],  # B9 <- B8A
        img[8],  # B10 <- B11
        img[8],  # B11
        img[9],  # B12
    ])

for feat in meta["features"]:
    props = feat["properties"]
    pid = props["ID_PATCH"]
    dates = props["dates-S2"]

    npy_path = os.path.join(DATA_DIR, f"S2_{pid}.npy")
    if not os.path.exists(npy_path):
        print(f"Skip {pid}: file not found")
        continue

    imgs = np.load(npy_path)  # (T, 10, 128, 128)

    # 按月聚合
    month_groups = {}
    for idx, date in dates.items():
        ym = date // 100
        month_groups.setdefault(ym, []).append(int(idx))

    tensors = []
    for ym in sorted(month_groups):
        indices = month_groups[ym]
        avg = torch.tensor(imgs[indices].mean(axis=0), dtype=torch.float32)  # (10, 128, 128)
        tensors.append(impute(avg))

    result = torch.stack(tensors)  # (T_months, 13, 128, 128)
    torch.save(result, os.path.join(OUT_DIR, f"S2_{pid}.pt"))
    print(f"{pid}: {result.shape}")

print("Done.")
