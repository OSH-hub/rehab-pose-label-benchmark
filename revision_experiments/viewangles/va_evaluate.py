r"""REHAB26-ViewAngles: full-frame inference of rehab26 models and agreement with the MediaPipe
reference (NOT ground truth) on 12 mappable joints, by viewing-angle bin.

Inference machinery is imported from uco_evaluate.py (same locked settings: imgsz 640, iou 0.70,
max_det 300, person class, highest-box-confidence instance, record down to conf 0.01, primary
detection threshold box conf >= 0.25). Agreement metrics per frame: pixel distance per joint on
frames where the model detected a person and the MediaPipe joint has visibility >= 0.5; agreement
@0.1 and @0.2 of the MediaPipe keypoint bounding-box max side (misses = failures, denominator =
all MediaPipe-visible joints).

Usage: python va_evaluate.py --prepared <dir> --registry <json> --protocol <txt> --out <dir>
                             --device cpu|0 [--batch 32] [--chunk-size 512] [--models ...]
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uco_evaluate as U  # noqa: E402

# MediaPipe index -> rehab26 index
MP2R = {11: 7, 12: 12, 13: 8, 14: 13, 15: 9, 16: 14, 23: 16, 24: 21, 25: 17, 26: 22, 27: 18, 28: 23}
MP_IDX = list(MP2R.keys())
R_IDX = [MP2R[k] for k in MP_IDX]
JOINT_NAMES = ["LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand", "LeftUpLeg", "RightUpLeg", "LeftLeg",
               "RightLeg", "LeftFoot", "RightFoot"]
VIS_T = 0.5
THR = {"010": 0.1, "020": 0.2}
BINS = ["0", "10-30", "40-60", "70-90"]


def load_manifest(prepared):
    meta = json.load(open(Path(prepared) / "prepare_meta.json", encoding="utf-8"))
    if meta.get("status") != "complete":
        raise RuntimeError("prepared data incomplete")
    for name, key in [("manifest.jsonl", "manifest_sha256"), ("reference.npz", "reference_npz_sha256")]:
        if U.sha256_file(Path(prepared) / name) != meta[key]:
            raise RuntimeError("prepared checksum mismatch: " + name)
    rows = [json.loads(l) for l in open(Path(prepared) / "manifest.jsonl", encoding="utf-8")]
    z = np.load(Path(prepared) / "reference.npz", allow_pickle=False)
    if list(z["frame_keys"]) != [r["frame_key"] for r in rows]:
        raise RuntimeError("manifest/reference key order mismatch")
    return meta, rows, {k: z[k] for k in z.files}


def score(rows, ref, pred26, detected):
    n = len(rows)
    mp_xy = ref["mp_xy"][:, MP_IDX, :]
    mp_vis = ref["mp_vis"][:, MP_IDX]
    allv = ref["mp_vis"] >= VIS_T
    scale = np.full(n, np.nan)
    for i in range(n):
        pts = ref["mp_xy"][i][allv[i]]
        if len(pts) >= 2:
            w = pts[:, 0].max() - pts[:, 0].min()
            h = pts[:, 1].max() - pts[:, 1].min()
            scale[i] = max(w, h)
    valid = (mp_vis >= VIS_T) & np.isfinite(mp_xy).all(axis=2) & np.isfinite(scale)[:, None] & (scale > 0)[:, None]
    pred = pred26[:, R_IDX, :2].astype(np.float64)
    pred = np.where(detected[:, None, None], pred, np.nan)
    finite = np.isfinite(pred).all(axis=2) & detected[:, None]
    dist = np.linalg.norm(pred - mp_xy, axis=2)
    dist = np.where(finite, dist, np.inf)
    nerr = dist / np.where(np.isfinite(scale) & (scale > 0), scale, np.nan)[:, None]
    agree = {k: valid & (nerr <= t) for k, t in THR.items()}
    px = np.where(finite & valid, dist, np.nan)
    return {"valid": valid, "scale": scale, "dist": dist, "nerr": nerr, "agree": agree, "px": px, "detected": detected}


def segment_rows(rows, s):
    """Unit = (video_id, camera, person_orientation) segment."""
    groups = {}
    info = {}
    for i, r in enumerate(rows):
        key = (r["video_id"], r["camera"], r["person_orientation"])
        groups.setdefault(key, []).append(i)
        info[key] = r
    out = []
    for key, idx in groups.items():
        idx = np.asarray(idx)
        r = info[key]
        det = s["detected"][idx]
        v = s["valid"][idx]
        row = {"video_id": r["video_id"], "exercise_id": r["exercise_id"], "camera": r["camera"], "person": r["person"],
               "person_orientation": r["person_orientation"], "view_angle": r["view_angle"], "angle_bin": r["angle_bin"],
               "n_frames": int(len(idx)), "n_detected": int(det.sum()), "detection_rate": float(det.mean()),
               "n_mp_valid_joints": int(v.sum())}
        for k in THR:
            a = s["agree"][k][idx]
            row[f"agree{k}_all"] = float(a.sum() / v.sum()) if v.sum() else float("nan")
            vd = v & det[:, None]
            row[f"agree{k}_det"] = float((a & vd).sum() / vd.sum()) if vd.sum() else float("nan")
        px = s["px"][idx]
        row["px_dist_det"] = U._mean(px)
        row["nerr_det"] = U._mean(np.where(np.isfinite(px), s["nerr"][idx], np.nan))
        for j, nm in enumerate(JOINT_NAMES):
            row[f"agree020_all_{nm}"] = float(s["agree"]["020"][idx][:, j].sum() / v[:, j].sum()) if v[:, j].sum() else float("nan")
        out.append(row)
    return out


KEYS = ["detection_rate", "agree010_all", "agree020_all", "agree010_det", "agree020_det", "px_dist_det", "nerr_det"]


def person_macro(seg_rows, keys=KEYS):
    by = {}
    for r in seg_rows:
        by.setdefault(r["person"], []).append(r)
    res = {"n_segments": len(seg_rows), "n_persons": len(by)}
    for k in keys:
        pm = [U._mean([r[k] for r in rs]) for rs in by.values()]
        res[k] = U._mean(pm)
        res[k + "_person_sd"] = float(np.nanstd(pm, ddof=1)) if len(pm) > 1 else float("nan")
    return res


def build(rows, ref, pred26, detected):
    s = score(rows, ref, pred26, detected)
    seg = segment_rows(rows, s)
    strata = {"overall": seg}
    for b in BINS:
        strata[f"bin_{b}"] = [r for r in seg if r["angle_bin"] == b]
    for ex in (1, 6, 7):
        strata[f"exercise_{ex}"] = [r for r in seg if r["exercise_id"] == ex]
        for b in BINS:
            strata[f"exercise_{ex}_bin_{b}"] = [r for r in seg if r["exercise_id"] == ex and r["angle_bin"] == b]
    for cam in (1, 2, 3):
        strata[f"camera_{cam}"] = [r for r in seg if r["camera"] == cam]
    for va in sorted({r["view_angle"] for r in seg}):
        strata[f"view_angle_{va:+d}"] = [r for r in seg if r["view_angle"] == va]
    metrics = {"person_macro": {k: person_macro(v) for k, v in strata.items()},
               "frame_micro": {"overall": {"n_frames": len(rows), "detection_rate": float(s["detected"].mean()),
                                           "agree020_all": float(s["agree"]["020"].sum() / s["valid"].sum()),
                                           "agree010_all": float(s["agree"]["010"].sum() / s["valid"].sum()),
                                           "px_dist_det": U._mean(s["px"])}},
               "mp_valid_joint_fraction": float(s["valid"].mean())}
    return s, seg, metrics


def run_model(entry, meta, rows, ref, out_root, protocol_hash, code_hash, device, batch, chunk_size, threads, versions):
    from ultralytics import YOLO
    folder = Path(out_root) / entry["model_id"]
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "metrics.json").exists():
        U.log(model=entry["model_id"], status="already_complete")
        return
    ckpt = Path(entry["checkpoint"])
    ck_hash = U.sha256_file(ckpt)
    if entry.get("checkpoint_sha256", ck_hash) != ck_hash:
        raise RuntimeError("checkpoint SHA256 mismatch")
    identity = {"schema": "va1", "model_entry": entry, "checkpoint_sha256": ck_hash, "protocol_sha256": protocol_hash,
                "manifest_sha256": meta["manifest_sha256"], "reference_npz_sha256": meta["reference_npz_sha256"], "stride": meta["stride"],
                "evaluator_sha256": code_hash, "uco_evaluator_sha256": U.sha256_file(Path(U.__file__)), "inference": U.INFERENCE_CONFIG,
                "mp2r": MP2R, "vis_threshold": VIS_T, "thresholds": THR}
    fp = hashlib.sha256(U.canonical(identity).encode()).hexdigest()
    idp = folder / "identity.json"
    if idp.exists():
        if json.load(open(idp, encoding="utf-8"))["fingerprint"] != fp:
            raise RuntimeError("fingerprint mismatch; refusing to overwrite " + entry["model_id"])
    else:
        json.dump({"fingerprint": fp, "identity": identity, "created_at_utc": U.now()}, open(idp, "w", encoding="utf-8"), indent=2)
    lock = folder / "prediction.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        model = YOLO(str(ckpt))
        shape = U.infer_model_shape(model)
        if shape != U.EXPECTED_SHAPE:
            raise RuntimeError("kpt_shape mismatch %s" % shape)
        n = len(rows)
        chunks_dir = folder / "prediction_chunks"
        chunks_dir.mkdir(exist_ok=True)
        starts = list(range(0, n, chunk_size))
        loader = VALoader(rows, threads)
        pending = None
        t0 = time.time()
        done = 0
        for ci, start in enumerate(starts):
            end = min(n, start + chunk_size)
            cp = chunks_dir / f"chunk_{ci:05d}.npz"
            if cp.exists():
                z = np.load(cp, allow_pickle=False)
                if str(z["fingerprint"]) != fp or int(z["start"]) != start or int(z["end"]) != end:
                    raise RuntimeError("chunk identity mismatch " + str(cp))
                continue
            if pending is None or pending[0] != start:
                pending = (start, loader.submit(start, end))
            imgs = [f.result() for f in pending[1]]
            nxt = ci + 1
            if nxt < len(starts) and not (chunks_dir / f"chunk_{nxt:05d}.npz").exists():
                ns, ne = starts[nxt], min(n, starts[nxt] + chunk_size)
                pending = (ns, loader.submit(ns, ne))
            else:
                pending = None
            pred, det, bconf, ninst, ninst025 = U.predict_chunk(model, imgs, device, batch, shape)
            tmp = cp.with_suffix(".tmp.npz")
            np.savez(tmp, fingerprint=np.asarray(fp), start=np.asarray(start), end=np.asarray(end), pred26=pred, detected_any=det,
                     box_conf=bconf, n_instances=ninst, n_instances_ge025=ninst025, frame_keys=np.asarray([r["frame_key"] for r in rows[start:end]]))
            os.replace(tmp, cp)
            done += end - start
            U.log(model=entry["model_id"], chunk=ci, of=len(starts), frames=end, fps=round(done / (time.time() - t0), 1),
                  det_rate_chunk_t025=round(float((det & (bconf >= U.PRIMARY_T)).mean()), 3))
        pred26 = np.full((n, 26, 3), np.nan, dtype=np.float32)
        det_any = np.zeros(n, dtype=bool)
        bconf = np.full(n, np.nan, dtype=np.float32)
        for ci, start in enumerate(starts):
            end = min(n, start + chunk_size)
            z = np.load(chunks_dir / f"chunk_{ci:05d}.npz", allow_pickle=False)
            if str(z["fingerprint"]) != fp or list(z["frame_keys"]) != [r["frame_key"] for r in rows[start:end]]:
                raise RuntimeError("chunk mismatch on assembly")
            pred26[start:end] = z["pred26"]; det_any[start:end] = z["detected_any"]; bconf[start:end] = z["box_conf"]
        bc = np.where(np.isfinite(bconf), bconf, -1.0)
        detected = det_any & (bc >= U.PRIMARY_T)
        s, seg, metrics = build(rows, ref, pred26, detected)
        sens = {}
        for t in U.SENS_T:
            _, _, m_t = build(rows, ref, pred26, det_any & (bc >= t))
            sens[str(t)] = {"person_macro": {k: m_t["person_macro"][k] for k in ("overall",) + tuple(f"bin_{b}" for b in BINS)}}
        pf = folder / "predictions.npz"
        np.savez_compressed(pf, frame_keys=np.asarray([r["frame_key"] for r in rows]), pred26=pred26, detected_any_t001=det_any, box_conf=bconf,
                            detected_primary_t025=detected, mp_xy12=ref["mp_xy"][:, MP_IDX, :].astype(np.float32), mp_vis12=ref["mp_vis"][:, MP_IDX].astype(np.float32),
                            mp_valid=s["valid"], scale=s["scale"].astype(np.float32), px_dist=s["px"].astype(np.float32),
                            agree020=s["agree"]["020"], agree010=s["agree"]["010"], meta=ref["meta"], meta_columns=ref["meta_columns"],
                            persons=ref["persons"], fingerprint=np.asarray(fp))
        for r in seg:
            r.update({"model_id": entry["model_id"], "method": entry["method"], "seed": entry["seed"], "budget": entry["budget"]})
        U.write_csv(folder / "per_segment.csv", seg)
        out = {"model_id": entry["model_id"], "method": entry["method"], "seed": entry["seed"], "budget": entry["budget"], "checkpoint": str(ckpt),
               "checkpoint_sha256": ck_hash, "fingerprint": fp, "protocol_sha256": protocol_hash, "manifest_sha256": meta["manifest_sha256"],
               "reference_npz_sha256": meta["reference_npz_sha256"], "evaluator_sha256": code_hash, "actual_kpt_shape": shape, "mp2r": MP2R,
               "inference_config": U.INFERENCE_CONFIG, "device": device, "frames": n, "segments": len(seg), "stride": meta["stride"],
               "reference_is_ground_truth": False, "metrics": metrics, "sensitivity_box_conf": sens, "predictions_sha256": U.sha256_file(pf),
               "per_segment_sha256": U.sha256_file(folder / "per_segment.csv"), "package_versions": versions, "completed_at_utc": U.now(),
               "elapsed_s": round(time.time() - t0, 1)}
        tmp = folder / "metrics.tmp.json"
        json.dump(out, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        os.replace(tmp, folder / "metrics.json")
        ov = metrics["person_macro"]["overall"]
        U.log(model=entry["model_id"], status="complete", detection_rate=round(ov["detection_rate"], 4), agree020_all=round(ov["agree020_all"], 4),
              px_dist_det=round(ov["px_dist_det"], 2), elapsed_s=round(time.time() - t0, 1))
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


class VALoader(U.Loader):
    def _load(self, r):
        import cv2
        b = open(r["image_path"], "rb").read()
        if U.sha256_bytes(b) != r["image_sha256"]:
            raise RuntimeError("frame cache checksum mismatch: " + r["frame_key"])
        img = cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.shape[0] != r["height"] or img.shape[1] != r["width"]:
            raise RuntimeError("frame decode failed: " + r["frame_key"])
        return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--loader-threads", type=int, default=8)
    ap.add_argument("--models", default="")
    a = ap.parse_args()
    import torch, ultralytics, cv2
    if a.device != "cpu" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise SystemExit("expected exactly one visible CUDA device")
    versions = {"python": sys.version.split()[0], "torch": torch.__version__, "ultralytics": ultralytics.__version__, "numpy": np.__version__,
                "opencv": cv2.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    protocol_hash = U.sha256_file(a.protocol)
    code_hash = U.sha256_file(Path(__file__))
    meta, rows, ref = load_manifest(a.prepared)
    reg = json.load(open(a.registry, encoding="utf-8-sig"))["models"]
    if a.models:
        want = a.models.split(",")
        reg = [e for e in reg if e["model_id"] in want]
        reg.sort(key=lambda e: want.index(e["model_id"]))
    U.log(stage="start", models=[e["model_id"] for e in reg], frames=len(rows), device=a.device, versions=versions)
    for e in reg:
        run_model(e, meta, rows, ref, a.out, protocol_hash, code_hash, a.device, a.batch, a.chunk_size, a.loader_threads, versions)
    U.log(stage="all_done")


if __name__ == "__main__":
    main()
