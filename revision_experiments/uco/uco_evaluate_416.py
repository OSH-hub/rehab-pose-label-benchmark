r"""Locked UCO Physical Rehabilitation RGB/2D full-frame external evaluation (rehab26 models).

Per model in the registry: verify checkpoint SHA256 and pose-head shape, predict every
manifest frame (full frame, imgsz 640, NMS iou 0.70, max_det 300, person class, no GT crop,
no keypoint-confidence filtering). Inference records candidates down to box conf 0.01 and
keeps the highest-box-confidence instance with its confidence; the locked primary scoring
threshold is box conf >= 0.25 (identical to predicting at conf=0.25: NMS only suppresses a box
by a higher-confidence box, so the surviving boxes >= 0.25 are the same set). Box conf >= 0.05
and >= 0.01 are pre-registered sensitivity analyses only. Raw 26x3 predictions are stored in
resumable chunks; predictions.npz, per_clip.csv and metrics.json are written only when every
manifest frame has a prediction or an explicit miss.

Usage: python uco_evaluate.py --prepared <dir> --registry <json> --protocol <txt> --out <dir>
            --device cpu|0 [--batch 32] [--chunk-size 512] [--models id1,id2] [--loader-threads 8]
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

INFERENCE_CONFIG = {"imgsz": 416, "rect": False, "conf": 0.01, "iou": 0.70, "max_det": 300, "augment": False,
                    "half": False, "classes": [0], "person_selection": "highest_box_confidence_first_on_tie",
                    "scoring_box_conf_primary": 0.25, "scoring_box_conf_sensitivity": [0.05, 0.01],
                    "gt_crop": False, "keypoint_confidence_filter": False, "input": "full_frame_1280x720_png"}
API_KEYS = {"imgsz", "rect", "conf", "iou", "max_det", "augment", "half", "classes"}
PRIMARY_T = 0.25
SENS_T = [0.05, 0.01]
# rehab26 0-based indices. UCO joint order lower: hip,knee,ankle ; upper: shoulder,elbow,wrist.
JOINT_MAP = {"lower/left": [16, 17, 18], "lower/right": [21, 22, 23], "upper/left": [7, 8, 9], "upper/right": [12, 13, 14]}
JOINT_MAP_NAMES = {"lower/left": ["LeftUpLeg", "LeftLeg", "LeftFoot"], "lower/right": ["RightUpLeg", "RightLeg", "RightFoot"],
                   "upper/left": ["LeftArm", "LeftForeArm", "LeftHand"], "upper/right": ["RightArm", "RightForeArm", "RightHand"]}
PCK_ALPHA = 0.2
ANGLE_TOL = [10.0, 15.0]
EXPECTED_SHAPE = [26, 3]


def now():
    return datetime.now(timezone.utc).isoformat()


def log(**v):
    print(json.dumps(v, ensure_ascii=True), flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def angle_deg(p0, p1, p2):
    a = p0 - p1
    b = p2 - p1
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (a * b).sum(-1) / (na * nb)
    out = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
    return np.where((na > 0) & (nb > 0) & np.isfinite(c), out, np.nan)


def choose_instance(box_conf, kp):
    if box_conf is None or kp is None:
        return None
    boxes = np.asarray(box_conf).reshape(-1)
    kp = np.asarray(kp)
    if not len(boxes):
        return None
    if kp.ndim != 3 or kp.shape[0] != len(boxes):
        raise ValueError("instance dimension mismatch")
    valid = np.isfinite(boxes)
    if not valid.any():
        return None
    i = int(np.argmax(np.where(valid, boxes, -np.inf)))
    return i, kp[i], float(boxes[i])


def load_manifest(prepared):
    meta = json.load(open(Path(prepared) / "prepare_meta.json", encoding="utf-8"))
    if meta.get("status") != "complete":
        raise RuntimeError("prepared data incomplete")
    for name, key in [("manifest.jsonl", "manifest_sha256"), ("gt.npz", "gt_npz_sha256")]:
        if sha256_file(Path(prepared) / name) != meta[key]:
            raise RuntimeError("prepared checksum mismatch: " + name)
    rows = [json.loads(l) for l in open(Path(prepared) / "manifest.jsonl", encoding="utf-8")]
    z = np.load(Path(prepared) / "gt.npz", allow_pickle=False)
    if list(z["frame_keys"]) != [r["frame_key"] for r in rows]:
        raise RuntimeError("manifest/gt key order mismatch")
    return meta, rows, {k: z[k] for k in z.files}


def score_frames(rows, gt, pred26, detected):
    """Per-frame arrays. Misses: detected False -> pck False, angle/px NaN."""
    n = len(rows)
    gt_xy = gt["gt_xy"]
    pred3 = np.full((n, 3, 2), np.nan, dtype=np.float64)
    kconf3 = np.full((n, 3), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        jm = JOINT_MAP[r["body"] + "/" + r["side"]]
        pred3[i] = pred26[i, jm, :2]
        kconf3[i] = pred26[i, jm, 2]
    pred3 = np.where(detected[:, None, None], pred3, np.nan)
    finite = np.isfinite(pred3).all(axis=2) & detected[:, None]
    dist = np.linalg.norm(pred3 - gt_xy, axis=2)
    dist = np.where(finite, dist, np.inf)
    limb = np.linalg.norm(gt_xy[:, 0] - gt_xy[:, 1], axis=1) + np.linalg.norm(gt_xy[:, 1] - gt_xy[:, 2], axis=1)
    thr = PCK_ALPHA * limb
    correct = dist <= thr[:, None]
    gt_ang = angle_deg(gt_xy[:, 0], gt_xy[:, 1], gt_xy[:, 2])
    pr_ang = angle_deg(pred3[:, 0], pred3[:, 1], pred3[:, 2])
    ang_err = np.abs(pr_ang - gt_ang)
    ang_err = np.where(detected & np.isfinite(ang_err), ang_err, np.nan)
    px = np.where(finite, dist, np.nan)
    return {"pred3": pred3, "kconf3": kconf3, "dist": dist, "limb": limb, "correct": correct, "gt_angle": gt_ang,
            "pred_angle": pr_ang, "angle_err": ang_err, "px_err": px, "detected": detected}


def _mean(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


def clip_metrics(idx, s, oob_any):
    """Metrics for a set of frame indices, plus an _xoob variant excluding out-of-image GT frames."""
    out = {}
    for suffix, sel in (("", idx), ("_xoob", idx[~oob_any[idx]])):
        n = len(sel)
        det = s["detected"][sel]
        nd = int(det.sum())
        out["n_frames" + suffix] = n
        out["n_detected" + suffix] = nd
        out["detection_rate" + suffix] = nd / n if n else float("nan")
        ae = s["angle_err"][sel]
        out["angle_mae_det" + suffix] = _mean(ae)
        out["angle_median_det" + suffix] = float(np.nanmedian(ae)) if np.isfinite(ae).any() else float("nan")
        for t in ANGLE_TOL:
            ok = np.isfinite(ae) & (ae <= t)
            out[f"angle_acc{int(t)}_all" + suffix] = ok.sum() / n if n else float("nan")
            out[f"angle_acc{int(t)}_det" + suffix] = ok.sum() / nd if nd else float("nan")
        c = s["correct"][sel]
        out["pck02_all" + suffix] = c.mean() if n else float("nan")
        out["pck02_det" + suffix] = c[det].mean() if nd else float("nan")
        for j, nm in enumerate(["prox", "vertex", "dist"]):
            out[f"pck02_all_{nm}" + suffix] = c[:, j].mean() if n else float("nan")
            out[f"px_err_det_{nm}" + suffix] = _mean(s["px_err"][sel][:, j])
        out["px_err_det" + suffix] = _mean(np.nanmean(s["px_err"][sel], axis=1)) if nd else float("nan")
        out["gt_limb_px_mean" + suffix] = _mean(s["limb"][sel])
    out["n_oob_frames"] = int(oob_any[idx].sum())
    return out


METRIC_KEYS = ["detection_rate", "angle_mae_det", "angle_median_det", "angle_acc10_all", "angle_acc10_det", "angle_acc15_all",
               "angle_acc15_det", "pck02_all", "pck02_det", "pck02_all_prox", "pck02_all_vertex", "pck02_all_dist",
               "px_err_det", "px_err_det_prox", "px_err_det_vertex", "px_err_det_dist"]


def hierarchical(clip_rows, keys=METRIC_KEYS, suffix=""):
    """clip -> subject (equal weight per clip within subject) -> mean over subjects (equal weight)."""
    by_subj = {}
    for r in clip_rows:
        by_subj.setdefault(r["subject_id"], []).append(r)
    res = {"n_clips": len(clip_rows), "n_subjects": len(by_subj)}
    for k in keys:
        kk = k + suffix
        subj_means = [_mean([r[kk] for r in rs]) for rs in by_subj.values()]
        res[k] = _mean(subj_means)
        res[k + "_subject_sd"] = float(np.nanstd(subj_means, ddof=1)) if len(subj_means) > 1 else float("nan")
    return res


def micro(clip_rows, s, idx_by_clip):
    idx = np.concatenate([idx_by_clip[r["tag"]] for r in clip_rows]) if clip_rows else np.zeros(0, dtype=int)
    n = len(idx)
    det = s["detected"][idx]
    return {"n_frames": int(n), "detection_rate": float(det.mean()) if n else float("nan"),
            "angle_mae_det": _mean(s["angle_err"][idx]), "pck02_all": float(s["correct"][idx].mean()) if n else float("nan"),
            "px_err_det": _mean(np.nanmean(s["px_err"][idx], axis=1)) if det.any() else float("nan")}


def build_tables(rows, gt, pred26, detected, full_strata=True):
    s = score_frames(rows, gt, pred26, detected)
    oob_any = gt["gt_oob"].any(axis=1)
    idx_by_clip = {}
    info = {}
    for i, r in enumerate(rows):
        tag = f"{r['subject_id']}_{int(r['exercise_id']):02d}_cam{r['camera_id']}"
        idx_by_clip.setdefault(tag, []).append(i)
        info[tag] = r
    clip_rows = []
    for tag, idx in idx_by_clip.items():
        idx_by_clip[tag] = idx = np.asarray(idx)
        r = info[tag]
        row = {"tag": tag, "subject_id": r["subject_id"], "exercise_id": r["exercise_id"], "camera_id": r["camera_id"],
               "body": r["body"], "side": r["side"], "joint_order": r["joint_order"], "rehab26_joints": "/".join(JOINT_MAP_NAMES[r["body"] + "/" + r["side"]])}
        row.update(clip_metrics(idx, s, oob_any))
        clip_rows.append(row)
    strata = {"overall": clip_rows}
    for cam in range(5):
        strata[f"camera_{cam}"] = [r for r in clip_rows if r["camera_id"] == cam]
    for body in ("lower", "upper"):
        strata[f"body_{body}"] = [r for r in clip_rows if r["body"] == body]
    if full_strata:
        for body in ("lower", "upper"):
            for cam in range(5):
                strata[f"body_{body}_camera_{cam}"] = [r for r in clip_rows if r["body"] == body and r["camera_id"] == cam]
        for ex in sorted({r["exercise_id"] for r in clip_rows}):
            strata[f"exercise_{int(ex):02d}"] = [r for r in clip_rows if r["exercise_id"] == ex]
    metrics = {"subject_macro": {k: hierarchical(v) for k, v in strata.items()},
               "subject_macro_excluding_oob_frames": {k: hierarchical(v, suffix="_xoob") for k, v in strata.items() if k in ("overall", "body_lower", "body_upper")},
               "frame_micro": {k: micro(v, s, idx_by_clip) for k, v in strata.items()},
               "oob": {"frames_with_any_oob_gt": int(oob_any.sum()), "clips_with_oob": int(sum(1 for r in clip_rows if r["n_oob_frames"] > 0))}}
    return s, clip_rows, metrics


class Loader:
    """Prefetches PNG frames (verifying SHA256) in background threads."""

    def __init__(self, rows, threads):
        self.rows = rows
        self.pool = ThreadPoolExecutor(max_workers=threads)

    def _load(self, r):
        import cv2
        b = open(r["image_path"], "rb").read()
        if sha256_bytes(b) != r["image_sha256"]:
            raise RuntimeError("frame cache checksum mismatch: " + r["frame_key"])
        img = cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.shape[:2] != (720, 1280):
            raise RuntimeError("frame decode failed: " + r["frame_key"])
        return img

    def submit(self, start, end):
        return [self.pool.submit(self._load, self.rows[i]) for i in range(start, end)]


def predict_chunk(model, imgs, device, batch, shape):
    n = len(imgs)
    pred = np.full((n, shape[0], shape[1]), np.nan, dtype=np.float32)
    det = np.zeros(n, dtype=bool)
    bconf = np.full(n, np.nan, dtype=np.float32)
    ninst = np.zeros(n, dtype=np.int16)
    ninst025 = np.zeros(n, dtype=np.int16)
    for s in range(0, n, batch):
        cur = imgs[s:s + batch]
        res = model.predict(source=cur, device=device, verbose=False, stream=False,
                            **{k: v for k, v in INFERENCE_CONFIG.items() if k in API_KEYS})
        if len(res) != len(cur):
            raise RuntimeError("predictor dropped a frame")
        for j, rr in enumerate(res):
            if rr.boxes is None or rr.keypoints is None:
                continue
            boxes = rr.boxes.conf.detach().cpu().numpy()
            kp = rr.keypoints.data.detach().cpu().numpy()
            ninst[s + j] = len(boxes)
            ninst025[s + j] = int((boxes >= PRIMARY_T).sum())
            ch = choose_instance(boxes, kp)
            if ch is None:
                continue
            _, inst, score = ch
            if inst.shape != (shape[0], shape[1]):
                raise RuntimeError("prediction head shape differs from checkpoint: %s" % (inst.shape,))
            pred[s + j] = inst
            det[s + j] = True
            bconf[s + j] = score
    return pred, det, bconf, ninst, ninst025


def infer_model_shape(model):
    head = model.model.model[-1]
    shape = getattr(head, "kpt_shape", None)
    if shape is None:
        shape = getattr(model.model, "yaml", {}).get("kpt_shape")
    if shape is None:
        raise ValueError("cannot verify pose head shape")
    return [int(v) for v in shape]


def write_csv(path, rows):
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_model(entry, meta, rows, gt, out_root, protocol_hash, code_hash, device, batch, chunk_size, threads, versions):
    from ultralytics import YOLO
    folder = Path(out_root) / entry["model_id"]
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "metrics.json").exists():
        log(model=entry["model_id"], status="already_complete")
        return
    ckpt = Path(entry["checkpoint"])
    ck_hash = sha256_file(ckpt)
    if entry.get("checkpoint_sha256", ck_hash) != ck_hash:
        raise RuntimeError("checkpoint SHA256 mismatch: " + entry["model_id"])
    identity = {"schema": 2, "model_entry": entry, "checkpoint_sha256": ck_hash, "protocol_sha256": protocol_hash,
                "manifest_sha256": meta["manifest_sha256"], "gt_npz_sha256": meta["gt_npz_sha256"], "stride": meta["stride"],
                "offset": meta["offset"], "evaluator_sha256": code_hash, "inference": INFERENCE_CONFIG, "joint_map": JOINT_MAP,
                "pck_alpha": PCK_ALPHA}
    fp = hashlib.sha256(canonical(identity).encode()).hexdigest()
    idp = folder / "identity.json"
    if idp.exists():
        old = json.load(open(idp, encoding="utf-8"))
        if old["fingerprint"] != fp:
            raise RuntimeError("existing results have a different fingerprint; refusing to overwrite: " + entry["model_id"])
    else:
        json.dump({"fingerprint": fp, "identity": identity, "created_at_utc": now()}, open(idp, "w", encoding="utf-8"), indent=2)
    lock = folder / "prediction.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        raise RuntimeError("lock exists (another process?) : " + str(lock))
    try:
        model = YOLO(str(ckpt))
        shape = infer_model_shape(model)
        if shape != EXPECTED_SHAPE or shape != list(entry["expected_kpt_shape"]):
            raise RuntimeError("kpt_shape mismatch %s" % shape)
        n = len(rows)
        chunks_dir = folder / "prediction_chunks"
        chunks_dir.mkdir(exist_ok=True)
        starts = list(range(0, n, chunk_size))
        loader = Loader(rows, threads)
        pending = None
        t0 = time.time()
        done_frames = 0
        for ci, start in enumerate(starts):
            end = min(n, start + chunk_size)
            cp = chunks_dir / f"chunk_{ci:05d}.npz"
            if cp.exists():
                z = np.load(cp, allow_pickle=False)
                if str(z["fingerprint"]) != fp or int(z["start"]) != start or int(z["end"]) != end:
                    raise RuntimeError("chunk identity mismatch: " + str(cp))
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
            pred, det, bconf, ninst, ninst025 = predict_chunk(model, imgs, device, batch, shape)
            tmp = cp.with_suffix(".tmp.npz")
            np.savez(tmp, fingerprint=np.asarray(fp), start=np.asarray(start), end=np.asarray(end), pred26=pred,
                     detected_any=det, box_conf=bconf, n_instances=ninst, n_instances_ge025=ninst025,
                     frame_keys=np.asarray([r["frame_key"] for r in rows[start:end]]))
            os.replace(tmp, cp)
            done_frames += end - start
            el = time.time() - t0
            log(model=entry["model_id"], chunk=ci, of=len(starts), frames=end, fps=round(done_frames / el, 1),
                det_rate_chunk_t025=round(float((det & (bconf >= PRIMARY_T)).mean()), 3))
        # assemble
        pred26 = np.full((n, 26, 3), np.nan, dtype=np.float32)
        det_any = np.zeros(n, dtype=bool)
        bconf = np.full(n, np.nan, dtype=np.float32)
        ninst = np.zeros(n, dtype=np.int16)
        ninst025 = np.zeros(n, dtype=np.int16)
        for ci, start in enumerate(starts):
            end = min(n, start + chunk_size)
            z = np.load(chunks_dir / f"chunk_{ci:05d}.npz", allow_pickle=False)
            if str(z["fingerprint"]) != fp or list(z["frame_keys"]) != [r["frame_key"] for r in rows[start:end]]:
                raise RuntimeError("chunk/fingerprint mismatch on assembly")
            pred26[start:end] = z["pred26"]; det_any[start:end] = z["detected_any"]; bconf[start:end] = z["box_conf"]
            ninst[start:end] = z["n_instances"]; ninst025[start:end] = z["n_instances_ge025"]
        bc = np.where(np.isfinite(bconf), bconf, -1.0)
        detected = det_any & (bc >= PRIMARY_T)
        s, clip_rows, metrics = build_tables(rows, gt, pred26, detected)
        sens = {}
        sens_clip = {}
        for t in SENS_T:
            d_t = det_any & (bc >= t)
            s_t, cr_t, m_t = build_tables(rows, gt, pred26, d_t, full_strata=False)
            sens[str(t)] = {"subject_macro": m_t["subject_macro"], "frame_micro": m_t["frame_micro"]}
            sens_clip[str(t)] = {r["tag"]: r for r in cr_t}
        pf = folder / "predictions.npz"
        np.savez_compressed(pf, frame_keys=np.asarray([r["frame_key"] for r in rows]), pred26=pred26, detected_any_t001=det_any,
                            box_conf=bconf, n_instances_t001=ninst, n_instances_t025=ninst025, detected_primary_t025=detected,
                            pred3=s["pred3"].astype(np.float32), gt3=gt["gt_xy"].astype(np.float32),
                            gt_oob=gt["gt_oob"], gt_angle=s["gt_angle"].astype(np.float32), pred_angle=s["pred_angle"].astype(np.float32),
                            angle_err=s["angle_err"].astype(np.float32), px_err=s["px_err"].astype(np.float32), pck02_correct=s["correct"],
                            meta=gt["meta"], meta_columns=gt["meta_columns"], fingerprint=np.asarray(fp), checkpoint_sha256=np.asarray(ck_hash))
        for r in clip_rows:
            r.update({"model_id": entry["model_id"], "method": entry["method"], "seed": entry["seed"], "budget": entry["budget"]})
            for t in SENS_T:
                st = sens_clip[str(t)][r["tag"]]
                tt = str(t).replace("0.", "t0")
                for k in ("detection_rate", "angle_mae_det", "angle_acc10_all", "pck02_all", "px_err_det"):
                    r[f"{k}_{tt}"] = st[k]
        write_csv(folder / "per_clip.csv", clip_rows)
        out = {"model_id": entry["model_id"], "method": entry["method"], "seed": entry["seed"], "budget": entry["budget"],
               "backbone": entry["backbone"], "checkpoint": str(ckpt), "checkpoint_sha256": ck_hash, "fingerprint": fp,
               "protocol_sha256": protocol_hash, "manifest_sha256": meta["manifest_sha256"], "gt_npz_sha256": meta["gt_npz_sha256"],
               "evaluator_sha256": code_hash, "actual_kpt_shape": shape, "joint_map": JOINT_MAP, "inference_config": INFERENCE_CONFIG,
               "device": device, "batch": batch, "frames": n, "clips": len(clip_rows), "stride": meta["stride"], "offset": meta["offset"],
               "primary_box_conf_threshold": PRIMARY_T, "metrics": metrics, "sensitivity_box_conf": sens,
               "predictions_sha256": sha256_file(pf), "per_clip_sha256": sha256_file(folder / "per_clip.csv"),
               "package_versions": versions, "completed_at_utc": now(), "elapsed_s": round(time.time() - t0, 1)}
        tmp = folder / "metrics.tmp.json"
        json.dump(out, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        os.replace(tmp, folder / "metrics.json")
        ov = metrics["subject_macro"]["overall"]
        log(model=entry["model_id"], status="complete", detection_rate=round(ov["detection_rate"], 4), angle_mae_det=round(ov["angle_mae_det"], 2),
            pck02_all=round(ov["pck02_all"], 4), px_err_det=round(ov["px_err_det"], 2), elapsed_s=round(time.time() - t0, 1))
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


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
    if a.device != "cpu":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise SystemExit("expected exactly one visible CUDA device under CUDA_VISIBLE_DEVICES mask")
    versions = {"python": sys.version.split()[0], "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                "numpy": np.__version__, "opencv": cv2.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    protocol_hash = sha256_file(a.protocol)
    code_hash = sha256_file(Path(__file__))
    meta, rows, gt = load_manifest(a.prepared)
    reg = json.load(open(a.registry, encoding="utf-8-sig"))["models"]
    if a.models:
        want = a.models.split(",")
        reg = [e for e in reg if e["model_id"] in want]
        reg.sort(key=lambda e: want.index(e["model_id"]))
    log(stage="start", models=[e["model_id"] for e in reg], frames=len(rows), device=a.device, versions=versions)
    for e in reg:
        run_model(e, meta, rows, gt, a.out, protocol_hash, code_hash, a.device, a.batch, a.chunk_size, a.loader_threads, versions)
    log(stage="all_done")


if __name__ == "__main__":
    main()
