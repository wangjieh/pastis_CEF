import json
from pyproj import Transformer

with open(r"C:\Users\23948\Desktop\pastis_data\metadata.geojson", encoding="utf-8") as f:
    data = json.load(f)

transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
results = []

def extract_points(coords):
    """递归提取所有 [x, y] 坐标点"""
    if isinstance(coords[0], (int, float)):
        return [coords]  # [x, y]
    return [pt for sub in coords for pt in extract_points(sub)]

for feat in data["features"]:
    props = feat["properties"]
    pid = props["ID_PATCH"]

    # 提取所有坐标点并取平均
    points = extract_points(feat["geometry"]["coordinates"])
    avg_x = sum(p[0] for p in points) / len(points)
    avg_y = sum(p[1] for p in points) / len(points)
    lon, lat = transformer.transform(avg_x, avg_y)

    # 对 dates-S2 按年月分组，同月内取日均值
    dates = sorted(props["dates-S2"].values())
    month_groups = {}
    for d in dates:
        ym = d // 100
        month_groups.setdefault(ym, []).append(d % 100)
    avg_dates = []
    for ym, days in sorted(month_groups.items()):
        avg_day = int(round(sum(days) / len(days)))
        avg_dates.append(ym * 100 + avg_day)

    results.append({
        "ID_PATCH": pid,
        "Fold": props["Fold"],
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "avg_dates_S2": avg_dates,
        "num_dates": len(avg_dates)
    })

with open(r"C:\Users\23948\Desktop\pastis_data\mytools\metadata_summary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Done, {len(results)} patches saved.")
