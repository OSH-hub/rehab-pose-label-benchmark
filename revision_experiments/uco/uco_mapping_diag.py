import json
import sys
from pathlib import Path

import numpy as np

MAP = {"lower/left": [16, 17, 18], "lower/right": [21, 22, 23], "upper/left": [7, 8, 9], "upper/right": [12, 13, 14]}
MIRROR = {"lower/left": "lower/right", "lower/right": "lower/left", "upper/left": "upper/right", "upper/right": "upper/left"}
rows = [json.loads(l) for l in open(Path(sys.argv[1]) / "manifest.jsonl", encoding="utf-8")]
z = np.load(Path(sys.argv[2]) / "predictions.npz", allow_pickle=False)
p26 = z["pred26"]
det = z["detected_primary_t025"]
gt = z["gt3"].astype(np.float64)
res = {}
for i, r in enumerate(rows):
    if not det[i]:
        continue
    key = r["body"] + "/" + r["side"]
    d = res.setdefault(key, {"n": 0, "mapped": [], "mirrored": [], "clavicle": [], "hip_center_alt": []})
    d["n"] += 1
    d["mapped"].append(np.linalg.norm(p26[i, MAP[key], :2] - gt[i], axis=1))
    d["mirrored"].append(np.linalg.norm(p26[i, MAP[MIRROR[key]], :2] - gt[i], axis=1))
    if r["body"] == "upper":
        alt = [6 if r["side"] == "left" else 11] + MAP[key][1:]
        d["clavicle"].append(np.linalg.norm(p26[i, alt, :2] - gt[i], axis=1))
for key, d in res.items():
    out = {"n_detected_frames": d["n"]}
    for k in ("mapped", "mirrored", "clavicle"):
        if d[k]:
            arr = np.asarray(d[k])
            out[k + "_px_per_joint"] = np.round(np.nanmean(arr, axis=0), 1).tolist()
            out[k + "_px_mean"] = round(float(np.nanmean(arr)), 1)
    print(key, json.dumps(out))
bc = z["box_conf"]
print("box_conf quantiles (all frames with any box):", np.round(np.nanpercentile(bc[np.isfinite(bc)], [10, 25, 50, 75, 90]), 3).tolist(),
      "frames_any_box", int(np.isfinite(bc).sum()), "of", len(bc))
