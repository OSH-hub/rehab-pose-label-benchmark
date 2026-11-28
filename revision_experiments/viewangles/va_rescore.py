import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import va_evaluate as V  # noqa: E402
import uco_evaluate as U  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--evaluations", required=True)
    a = ap.parse_args()
    meta, rows, ref = V.load_manifest(a.prepared)
    w = np.asarray([r["width"] for r in rows], dtype=np.float64)[:, None]
    h = np.asarray([r["height"] for r in rows], dtype=np.float64)[:, None]
    mp = np.array(ref["mp_xy"], dtype=np.float64)
    if np.nanmax(mp) > 2.0:
        raise SystemExit("reference does not look normalized; refusing")
    mp[:, :, 0] *= w
    mp[:, :, 1] *= h
    ref = dict(ref)
    ref["mp_xy"] = mp
    diag_rows = []
    for folder in sorted(Path(a.evaluations).iterdir()):
        mfile = folder / "metrics.json"
        if not mfile.exists():
            continue
        m = json.load(open(mfile, encoding="utf-8"))
        if m.get("reference_unit_correction"):
            print("already rescored", folder.name)
            continue
        z = np.load(folder / "predictions.npz", allow_pickle=False)
        if list(z["frame_keys"]) != [r["frame_key"] for r in rows]:
            raise RuntimeError("frame key mismatch " + folder.name)
        pred26 = z["pred26"]
        det_any = z["detected_any_t001"]
        bconf = z["box_conf"]
        bc = np.where(np.isfinite(bconf), bconf, -1.0)
        detected = det_any & (bc >= U.PRIMARY_T)
        s, seg, metrics = V.build(rows, ref, pred26, detected)
        sens = {}
        for t in U.SENS_T:
            _, _, m_t = V.build(rows, ref, pred26, det_any & (bc >= t))
            sens[str(t)] = {"person_macro": {k: m_t["person_macro"][k] for k in ("overall",) + tuple(f"bin_{b}" for b in V.BINS)}}
        for name in ("metrics.json", "per_segment.csv"):
            src = folder / name
            dst = folder / (Path(name).stem + "_v1_unscaled" + Path(name).suffix)
            if not dst.exists():
                shutil.copy2(src, dst)
        np.savez_compressed(folder / "predictions.npz", frame_keys=z["frame_keys"], pred26=pred26, detected_any_t001=det_any, box_conf=bconf,
                            detected_primary_t025=detected, mp_xy12_px=mp[:, V.MP_IDX, :].astype(np.float32), mp_vis12=z["mp_vis12"],
                            mp_valid=s["valid"], scale_px=s["scale"].astype(np.float32), px_dist=s["px"].astype(np.float32),
                            agree020=s["agree"]["020"], agree010=s["agree"]["010"], meta=z["meta"], meta_columns=z["meta_columns"],
                            persons=z["persons"], fingerprint=z["fingerprint"])
        for r in seg:
            r.update({"model_id": m["model_id"], "method": m["method"], "seed": m["seed"], "budget": m["budget"]})
        U.write_csv(folder / "per_segment.csv", seg)
        m2 = dict(m)
        m2.update({"metrics": metrics, "sensitivity_box_conf": sens, "predictions_sha256": U.sha256_file(folder / "predictions.npz"),
                   "per_segment_sha256": U.sha256_file(folder / "per_segment.csv"),
                   "reference_unit_correction": {"applied": True, "rule": "mp_x*width, mp_y*height (normalized MediaPipe coords -> pixels)",
                                                 "rescore_code_sha256": U.sha256_file(Path(__file__)), "original_kept_as": "metrics_v1_unscaled.json",
                                                 "note": "post-hoc unit correction of the reference, identical for all models; no inference rerun"},
                   "rescored_at_utc": U.now()})
        tmp = folder / "metrics.tmp.json"
        json.dump(m2, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        os.replace(tmp, mfile)
        ov = metrics["person_macro"]["overall"]
        print(folder.name, "det", round(ov["detection_rate"], 3), "agree020_all", round(ov["agree020_all"], 3), "agree020_det", round(ov["agree020_det"], 3), "px", round(ov["px_dist_det"], 1), flush=True)
        if m["method"] in ("B", "ref"):
            by = {}
            for i, r in enumerate(rows):
                by.setdefault((r["video_id"], r["camera"]), []).append(i)
            mismatch = {t: (d, mpf) for t, d, mpf in meta.get("decoded_vs_mediapipe_frames", [])}
            for (vid, cam), idx in by.items():
                idx = np.asarray(idx)
                tag = f"{vid}_cam{cam}"
                d = s["detected"][idx]
                vd = s["valid"][idx] & d[:, None]
                ag = float((s["agree"]["020"][idx] & vd).sum() / vd.sum()) if vd.sum() else float("nan")
                dm = mismatch.get(tag, (None, None))
                diag_rows.append({"model_id": m["model_id"], "recording": tag, "n_frames": len(idx), "detection_rate": float(d.mean()),
                                  "agree020_det": ag, "px_dist_det": U._mean(s["px"][idx]), "decoded_frames": dm[0], "mediapipe_frames": dm[1],
                                  "decoded_minus_mp": (dm[0] - dm[1]) if dm[0] is not None else 0})
    if diag_rows:
        U.write_csv(Path(a.evaluations).parent / "alignment_diagnostic_per_recording.csv", diag_rows)
        big = [r for r in diag_rows if r["decoded_minus_mp"] > 100]
        small = [r for r in diag_rows if r["decoded_minus_mp"] <= 100]
        print("alignment diag: agree020_det mean | decoded-mp>100:", round(U._mean([r["agree020_det"] for r in big]), 3), "n", len(big),
              "| <=100:", round(U._mean([r["agree020_det"] for r in small]), 3), "n", len(small))


if __name__ == "__main__":
    main()
