import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ev_code, prepared, ev_dir, reg_path, out = (Path(a) for a in sys.argv[1:6])
sys.path.insert(0, str(ev_code))
import penn_evaluate as pe  # noqa: E402

out.mkdir(parents=True, exist_ok=True)
with np.load(prepared / "gt_test.npz", allow_pickle=False) as z:
    gt = {k: z[k].copy() for k in z.files}
with open(reg_path, "r", encoding="utf-8-sig") as f:
    registry = json.load(f)["models"]
rows = []
for m in registry:
    folder = ev_dir / m["model_id"]
    if not (folder / "metrics.json").exists():
        continue
    with np.load(folder / "predictions.npz", allow_pickle=False) as z:
        xy, det, kconf, bconf = z["xy"].astype(np.float64), z["detected"].astype(bool), z["keypoint_confidence"], z["box_confidence"]
    sc = pe.score_predictions(gt["gt_xy"], gt["visibility"], gt["bbox_xyxy"], xy, det)
    mask = sc["mask"]
    den_all = int(mask.sum())
    in_det = mask & det[:, None]
    den_det = int(in_det.sum())
    rec = {"model_id": m["model_id"], "method": m["method"], "seed": m["seed"], "budget": m["budget"],
           "frames": int(len(det)), "frames_detected": int(det.sum()), "detection_rate": float(det.mean()),
           "evaluable_joints": den_all, "evaluable_joints_in_detected_frames": den_det,
           "share_evaluable_joints_in_detected_frames": den_det / den_all,
           "mean_box_conf_detected": float(np.nanmean(bconf[det])) if det.any() else None,
           "mean_kpt_conf_detected_12": float(np.nanmean(kconf[det])) if det.any() else None}
    for tag in ("020", "010"):
        corr = sc["correct"][tag]
        rec["micro_pck_" + tag + "_all"] = float(corr.sum() / den_all)
        rec["micro_pck_" + tag + "_given_detected"] = float((corr & in_det).sum() / den_det) if den_det else None
        rec["micro_pck_" + tag + "_upper_bound_if_localization_perfect"] = den_det / den_all
    clip_det = np.zeros(len(gt["clip_ids"]), dtype=bool)
    for i in range(len(gt["clip_ids"])):
        clip_det[i] = det[gt["clip_index"] == i].any()
    rec["clips_with_any_detection"] = int(clip_det.sum())
    rec["clips_with_zero_detection"] = int((~clip_det).sum())
    rows.append(rec)
with open(out / "detection_decomposition.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
groups = defaultdict(list)
for r in rows:
    groups[(r["method"], r["budget"])].append(r)
summary = []
keys = ["detection_rate", "share_evaluable_joints_in_detected_frames", "micro_pck_020_all", "micro_pck_020_given_detected",
        "micro_pck_010_all", "micro_pck_010_given_detected", "clips_with_zero_detection", "mean_box_conf_detected", "mean_kpt_conf_detected_12"]
for (meth, bud), rs in sorted(groups.items(), key=lambda kv: (-kv[0][1], kv[0][0])):
    rec = {"method": meth, "budget": bud, "n_seeds": len(rs)}
    for k in keys:
        v = np.array([r[k] for r in rs if r[k] is not None], dtype=float)
        rec[k + "_mean"] = float(v.mean()); rec[k + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else None
        rec[k + "_min"] = float(v.min()); rec[k + "_max"] = float(v.max())
    summary.append(rec)
    print("%-12s b=%.2f det=%.3f (%.3f-%.3f) jointshare=%.3f PCK02 all=%.4f | given det=%.4f (%.3f-%.3f) PCK01 given det=%.4f zero-det clips=%.0f" % (
        meth, bud, rec["detection_rate_mean"], rec["detection_rate_min"], rec["detection_rate_max"],
        rec["share_evaluable_joints_in_detected_frames_mean"], rec["micro_pck_020_all_mean"], rec["micro_pck_020_given_detected_mean"],
        rec["micro_pck_020_given_detected_min"], rec["micro_pck_020_given_detected_max"], rec["micro_pck_010_given_detected_mean"], rec["clips_with_zero_detection_mean"]))
with open(out / "detection_decomposition_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
