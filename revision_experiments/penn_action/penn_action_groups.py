r"""Rehab-like Penn actions (squat, pushup, jumping_jacks) vs all other actions, per method.
Clip-macro PCK@0.2 / @0.1 and clip-macro detection rate inside each group, mean +- SD over seeds,
plus seed-paired difference vs A within each group (clip-level bootstrap 95% CI).
Usage: python penn_action_groups.py <evaluations_dir> <registry.json> <out_csv>"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ev_dir, reg_path, out_csv = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
REHAB = {"squat", "pushup", "jumping_jacks"}
rng = np.random.default_rng(20261013)
with open(reg_path, "r", encoding="utf-8-sig") as f:
    registry = json.load(f)["models"]
tables = {}
for m in registry:
    p = ev_dir / m["model_id"] / "per_clip_raw.csv"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            tables[m["model_id"]] = list(csv.DictReader(f))
meta = {m["model_id"]: m for m in registry}
clip_act = np.array([r["action"] for r in tables[next(iter(tables))]])
groups = {"rehab_like(squat,pushup,jumping_jacks)": np.isin(clip_act, list(REHAB)), "other_12_actions": ~np.isin(clip_act, list(REHAB))}
def arr(mid, key):
    return np.array([float(r[key]) if r[key] != "" else np.nan for r in tables[mid]])
def det(mid):
    return np.array([int(r["n_frames_with_detection"]) / int(r["n_frames"]) for r in tables[mid]])
by = defaultdict(list)
for mid in tables:
    by[(meta[mid]["method"], meta[mid]["budget"])].append(mid)
rows = []
for gname, sel in groups.items():
    idx = np.where(sel)[0]
    boot = rng.integers(0, len(idx), size=(5000, len(idx)))
    for (meth, bud), mids in sorted(by.items(), key=lambda kv: (-kv[0][1], kv[0][0])):
        mids = sorted(mids, key=lambda x: meta[x]["seed"])
        p2 = [np.nanmean(arr(mid, "pck_020")[sel]) for mid in mids]
        p1 = [np.nanmean(arr(mid, "pck_010")[sel]) for mid in mids]
        dd = [det(mid)[sel].mean() for mid in mids]
        rec = {"group": gname, "n_clips": int(sel.sum()), "method": meth, "budget": bud, "n_seeds": len(mids),
               "pck_020_mean": np.mean(p2), "pck_020_std": np.std(p2, ddof=1), "pck_010_mean": np.mean(p1), "pck_010_std": np.std(p1, ddof=1),
               "detection_rate_mean": np.mean(dd), "detection_rate_std": np.std(dd, ddof=1), "diff_vs_A_pck_020": "", "ci95": ""}
        if meth != "A" and ("A", bud) in by:
            A = {meta[x]["seed"]: x for x in by[("A", bud)]}
            pairs = [(A[meta[mid]["seed"]], mid) for mid in mids if meta[mid]["seed"] in A]
            D = np.stack([arr(mid, "pck_020")[idx] - arr(aid, "pck_020")[idx] for aid, mid in pairs]).mean(axis=0)
            b = np.nanmean(D[boot], axis=1)
            rec["diff_vs_A_pck_020"] = "%+.4f" % np.nanmean(D)
            rec["ci95"] = "[%+.4f, %+.4f]" % tuple(np.percentile(b, [2.5, 97.5]))
        rows.append(rec)
        print("%-38s %-12s b=%.2f PCK02=%.4f+-%.4f PCK01=%.4f+-%.4f det=%.3f+-%.3f  dA=%s %s" % (gname, meth, bud, rec["pck_020_mean"], rec["pck_020_std"], rec["pck_010_mean"], rec["pck_010_std"], rec["detection_rate_mean"], rec["detection_rate_std"], rec["diff_vs_A_pck_020"], rec["ci95"]))
with open(out_csv, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
