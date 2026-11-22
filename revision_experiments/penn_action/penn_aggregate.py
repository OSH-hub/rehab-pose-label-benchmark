r"""Aggregate completed Penn Action evaluations (protocol v1.1) into method-level tables.

Usage: python -I penn_aggregate.py <evaluations_dir> <registry.json> <out_dir>

Produces in <out_dir>:
  per_model.csv            one row per completed model (metrics.json values)
  per_method.csv           mean +- std over the 6 training seeds for each (method, budget)
  paired_vs_A.csv          seed-paired differences vs A (same seed, same budget) with clip-level
                           bootstrap 95% CI (clips resampled with replacement, seeds kept paired)
  per_action.csv           clip-macro PCK@0.2 by Penn action class, per method (mean over seeds)
  aggregate_summary.json   everything above plus counts/caveats for the markdown write-up
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

B = 5000
RNG = np.random.default_rng(20261013)


def read_csv(p):
    with open(p, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ev_dir, reg_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    with open(reg_path, "r", encoding="utf-8-sig") as f:
        registry = json.load(f)["models"]
    models = {}
    clip_tables = {}
    for m in registry:
        mp = ev_dir / m["model_id"] / "metrics.json"
        if not mp.exists():
            print("MISSING", m["model_id"])
            continue
        with open(mp, "r", encoding="utf-8") as f:
            met = json.load(f)
        if met.get("status") != "complete":
            print("INCOMPLETE", m["model_id"])
            continue
        if met["official_test_frames_processed"] != 76298 or met["official_test_clips_processed"] != 1068:
            raise RuntimeError("frame/clip count mismatch " + m["model_id"])
        models[m["model_id"]] = {"model_id": m["model_id"], "method": m["method"], "seed": int(m["seed"]),
                                 "budget": float(m["budget"]), "provenance": m.get("provenance", ""),
                                 "checkpoint_sha256": met["checkpoint_sha256"], "fingerprint": met["fingerprint"],
                                 **{k: met["metrics"][k] for k in ("clip_macro_pck_020", "clip_macro_pck_010",
                                                                   "micro_pck_020", "micro_pck_010", "detection_rate",
                                                                   "evaluable_joints", "zero_evaluable_clips",
                                                                   "invalid_bbox_frames")}}
        clip_tables[m["model_id"]] = read_csv(ev_dir / m["model_id"] / "per_clip_raw.csv")
    ids = list(models)
    # common clip order
    clip_ids = [r["clip_id"] for r in clip_tables[ids[0]]]
    actions = {r["clip_id"]: r["action"] for r in clip_tables[ids[0]]}
    for mid in ids:
        if [r["clip_id"] for r in clip_tables[mid]] != clip_ids:
            raise RuntimeError("clip order differs " + mid)
        if [r["n_evaluable_joints"] for r in clip_tables[mid]] != [r["n_evaluable_joints"] for r in clip_tables[ids[0]]]:
            raise RuntimeError("GT denominators differ " + mid)
    pck = {tag: {mid: np.array([float(r["pck_" + tag]) if r["pck_" + tag] != "" else np.nan for r in clip_tables[mid]])
                 for mid in ids} for tag in ("020", "010")}
    detrate_clip = {mid: np.array([int(r["n_frames_with_detection"]) / int(r["n_frames"]) for r in clip_tables[mid]]) for mid in ids}
    # per_model.csv
    rows = sorted(models.values(), key=lambda r: (r["budget"], r["method"], r["seed"]))
    with open(out / "per_model.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # per_method.csv
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["budget"])].append(r)
    keys = ["clip_macro_pck_020", "clip_macro_pck_010", "micro_pck_020", "micro_pck_010", "detection_rate"]
    method_rows = []
    for (meth, bud), rs in sorted(groups.items(), key=lambda kv: (-kv[0][1], kv[0][0])):
        rec = {"method": meth, "budget": bud, "n_seeds": len(rs), "seeds": " ".join(str(r["seed"]) for r in sorted(rs, key=lambda r: r["seed"]))}
        for k in keys:
            v = np.array([r[k] for r in rs], dtype=float)
            rec[k + "_mean"] = float(v.mean())
            rec[k + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else None
            rec[k + "_min"] = float(v.min())
            rec[k + "_max"] = float(v.max())
        method_rows.append(rec)
    with open(out / "per_method.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(method_rows[0]))
        w.writeheader()
        w.writerows(method_rows)
    # paired differences vs A (same seed, same budget), clip-level bootstrap with seeds paired
    paired = []
    n_clips = len(clip_ids)
    boot_idx = RNG.integers(0, n_clips, size=(B, n_clips))
    for (meth, bud), rs in groups.items():
        if meth == "A":
            continue
        pairs = []
        for r in rs:
            a = [x for x in groups.get(("A", bud), []) if x["seed"] == r["seed"]]
            if a:
                pairs.append((a[0]["model_id"], r["model_id"], r["seed"]))
        if not pairs:
            continue
        for tag in ("020", "010"):
            diffs_seed = []
            # per-seed clip-macro difference (observed)
            for aid, mid, seed in pairs:
                diffs_seed.append(float(np.nanmean(pck[tag][mid]) - np.nanmean(pck[tag][aid])))
            obs = float(np.mean(diffs_seed))
            # bootstrap over clips, seeds paired
            A_mat = np.stack([pck[tag][aid] for aid, _, _ in pairs])   # [S, C]
            M_mat = np.stack([pck[tag][mid] for _, mid, _ in pairs])
            D = M_mat - A_mat                                             # [S, C], same clips, same denominators
            d_clip = D.mean(axis=0)                                       # seed-averaged clip difference
            boot = np.nanmean(d_clip[boot_idx], axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            paired.append({"method": meth, "budget": bud, "vs": "A", "threshold": tag, "n_seed_pairs": len(pairs),
                           "seeds": " ".join(str(s) for _, _, s in pairs),
                           "mean_paired_diff": obs, "per_seed_diffs": " ".join("%.5f" % d for d in diffs_seed),
                           "seed_sd_of_diff": float(np.std(diffs_seed, ddof=1)) if len(diffs_seed) > 1 else None,
                           "boot_ci95_low": float(lo), "boot_ci95_high": float(hi), "bootstrap_resamples": B,
                           "resampling_unit": "clip (1068), seeds kept paired"})
        # detection rate paired diff (frame-level rate, clip bootstrap of clip-level detection rate)
        A_d = np.stack([detrate_clip[aid] for aid, _, _ in pairs]); M_d = np.stack([detrate_clip[mid] for _, mid, _ in pairs])
        d_clip = (M_d - A_d).mean(axis=0)
        boot = d_clip[boot_idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        paired.append({"method": meth, "budget": bud, "vs": "A", "threshold": "detection_rate_clip_macro", "n_seed_pairs": len(pairs),
                       "seeds": " ".join(str(s) for _, _, s in pairs), "mean_paired_diff": float(d_clip.mean()),
                       "per_seed_diffs": " ".join("%.5f" % (M_d[i].mean() - A_d[i].mean()) for i in range(len(pairs))),
                       "seed_sd_of_diff": float(np.std([M_d[i].mean() - A_d[i].mean() for i in range(len(pairs))], ddof=1)) if len(pairs) > 1 else None,
                       "boot_ci95_low": float(lo), "boot_ci95_high": float(hi), "bootstrap_resamples": B,
                       "resampling_unit": "clip (1068), seeds kept paired"})
    # A budget 0.05 vs A budget 0.40 (same seed)
    if ("A", 0.05) in groups and ("A", 0.4) in groups:
        pairs = []
        for r in groups[("A", 0.05)]:
            a = [x for x in groups[("A", 0.4)] if x["seed"] == r["seed"]]
            if a:
                pairs.append((a[0]["model_id"], r["model_id"], r["seed"]))
        for tag in ("020", "010"):
            A_mat = np.stack([pck[tag][aid] for aid, _, _ in pairs]); M_mat = np.stack([pck[tag][mid] for _, mid, _ in pairs])
            d_clip = (M_mat - A_mat).mean(axis=0)
            boot = np.nanmean(d_clip[boot_idx], axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            ds = [float(np.nanmean(M_mat[i]) - np.nanmean(A_mat[i])) for i in range(len(pairs))]
            paired.append({"method": "A(b005)", "budget": 0.05, "vs": "A(b040)", "threshold": tag, "n_seed_pairs": len(pairs),
                           "seeds": " ".join(str(s) for _, _, s in pairs), "mean_paired_diff": float(np.mean(ds)),
                           "per_seed_diffs": " ".join("%.5f" % d for d in ds),
                           "seed_sd_of_diff": float(np.std(ds, ddof=1)) if len(ds) > 1 else None,
                           "boot_ci95_low": float(lo), "boot_ci95_high": float(hi), "bootstrap_resamples": B,
                           "resampling_unit": "clip (1068), seeds kept paired"})
    with open(out / "paired_vs_A.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0]))
        w.writeheader()
        w.writerows(paired)
    # per action
    act_list = sorted(set(actions.values()))
    clip_act = np.array([actions[c] for c in clip_ids])
    action_rows = []
    for (meth, bud), rs in sorted(groups.items(), key=lambda kv: (-kv[0][1], kv[0][0])):
        for act in act_list:
            sel = clip_act == act
            vals = [float(np.nanmean(pck["020"][r["model_id"]][sel])) for r in rs]
            dets = [float(detrate_clip[r["model_id"]][sel].mean()) for r in rs]
            action_rows.append({"method": meth, "budget": bud, "action": act, "n_clips": int(sel.sum()),
                                "clip_macro_pck_020_mean": float(np.mean(vals)), "clip_macro_pck_020_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                                "detection_rate_mean": float(np.mean(dets))})
    with open(out / "per_action.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(action_rows[0]))
        w.writeheader()
        w.writerows(action_rows)
    summary = {"n_models_complete": len(ids), "n_clips": n_clips, "evaluable_joints": rows[0]["evaluable_joints"],
               "invalid_bbox_frames": rows[0]["invalid_bbox_frames"], "zero_evaluable_clips": rows[0]["zero_evaluable_clips"],
               "per_method": method_rows, "paired": paired, "per_action": action_rows,
               "bootstrap": {"resamples": B, "seed": 20261013, "unit": "clip"}}
    with open(out / "aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    for r in method_rows:
        print("%-12s b=%.2f n=%d  PCK0.2=%.4f+-%.4f  PCK0.1=%.4f+-%.4f  det=%.4f+-%.4f" % (
            r["method"], r["budget"], r["n_seeds"], r["clip_macro_pck_020_mean"], r["clip_macro_pck_020_std"] or 0,
            r["clip_macro_pck_010_mean"], r["clip_macro_pck_010_std"] or 0, r["detection_rate_mean"], r["detection_rate_std"] or 0))
    for p in paired:
        print("%-10s vs %-8s %-28s diff=%+.4f  CI95=[%+.4f, %+.4f]" % (p["method"], p["vs"], p["threshold"], p["mean_paired_diff"], p["boot_ci95_low"], p["boot_ci95_high"]))


if __name__ == "__main__":
    main()
