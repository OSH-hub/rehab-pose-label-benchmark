r"""KERAAL group1A (low-back-pain patients, Poppy-coached) frozen-model transfer check.

Stages (run in order; each is resumable):
  --extract : decode every 5th frame (0-based index 0,5,10,...) of all group1A videos into a
              uint8 memmap + index CSV, and parse the distributed OpenPose 2D reference into arrays.
  --infer   : run the 18 budget-0.40 original checkpoints (A/B/ref x 6 seeds) at imgsz 640,
              conf 0.25, iou 0.70, max_det 300, half False, augment False, person = highest box conf.
  --report  : per-model / per-method tables (detection rate, mean keypoint confidence, agreement
              with OpenPose reference for 12 mappable joints) + qualitative overlays.

The OpenPose coordinates shipped with KERAAL are an automatic estimator's output, NOT ground truth.
Agreement with them is a consistency measure between two estimators, not an accuracy claim.
Fixed conventions (decided before seeing any result):
  * OpenPose dict key for 0-based video frame i is f"{i+1}.0" (keys start at '1.0').
  * OpenPose normalized (x,y) are multiplied by the frame width/height (480x360) to get pixels.
  * OpenPose joint missing if key absent, non-finite, or both coordinates == 0.
  * Agreement scale s = max side of the bbox spanned by the available OpenPose joints of that
    frame (all 14 joints incl. head/mShoulder); frames with < 4 OpenPose joints or s <= 0 are
    excluded from agreement (counted). Agreement@t: ||pred-ref||/s <= t, t in {0.2, 0.1}.
    Missed detections count as disagreement for every available reference joint.
  * Mapping OpenPose -> REHAB26 index: rShoulder->12 lShoulder->7 rElbow->13 lElbow->8
    rWrist->14 lWrist->9 rHip->21 lHip->16 rKnee->22 lKnee->17 rAnkle->23 lAnkle->18.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

STEP = 5
OP_NAMES = ["Head", "mShoulder", "rShoulder", "rElbow", "rWrist", "lShoulder", "lElbow", "lWrist",
            "rHip", "rKnee", "rAnkle", "lHip", "lKnee", "lAnkle"]
OP_TO_REHAB = {"rShoulder": 12, "lShoulder": 7, "rElbow": 13, "lElbow": 8, "rWrist": 14, "lWrist": 9,
               "rHip": 21, "lHip": 16, "rKnee": 22, "lKnee": 17, "rAnkle": 23, "lAnkle": 18}
MAP_OP_IDX = [OP_NAMES.index(n) for n in OP_TO_REHAB]        # 12 openpose indices
MAP_REHAB_IDX = [OP_TO_REHAB[n] for n in OP_TO_REHAB]        # 12 rehab26 indices
MAP_NAMES = list(OP_TO_REHAB)
INFER = {"imgsz": 640, "rect": False, "conf": 0.25, "iou": 0.70, "max_det": 300, "augment": False,
         "half": False, "classes": [0]}
THR = {"020": 0.2, "010": 0.1}
METHODS = ("A", "B", "ref")
# Fixed before any result: (exercise, subject) combos for qualitative overlays, one frame each.
OVERLAY_COMBOS = [("CTK", "R1-Brest"), ("ELK", "R3-Brest"), ("RTK", "R5-Roscoff"), ("CTK", "R6-Roscoff")]
EDGES = [("lShoulder", "rShoulder"), ("lShoulder", "lElbow"), ("lElbow", "lWrist"), ("rShoulder", "rElbow"),
         ("rElbow", "rWrist"), ("lShoulder", "lHip"), ("rShoulder", "rHip"), ("lHip", "rHip"),
         ("lHip", "lKnee"), ("lKnee", "lAnkle"), ("rHip", "rKnee"), ("rKnee", "rAnkle")]


def log(**kw):
    print(json.dumps(kw, ensure_ascii=True), flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def parse_name(stem):
    # G1A-Anon-<Exercise>-<Subject>-<Site>-<Rec>
    parts = stem.split("-")
    return {"exercise": parts[2], "subject": parts[3] + "-" + parts[4], "site": parts[4], "rec": parts[5]}


# ----------------------------------------------------------------------------- extract
def extract(root, out):
    import cv2
    out.mkdir(parents=True, exist_ok=True)
    done = out / "extract_complete.json"
    if done.exists():
        log(stage="extract", status="already_complete")
        return
    videos = sorted((root / "videos").glob("*.mp4"))
    rows = []
    shape = None
    raw = open(out / "frames_bgr.u8", "wb")
    for vi, v in enumerate(videos):
        cap = cv2.VideoCapture(str(v))
        i = 0
        n_sampled = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % STEP == 0:
                if shape is None:
                    shape = frame.shape
                if frame.shape != shape:
                    raise RuntimeError("Frame shape differs: %s %s vs %s" % (v.name, frame.shape, shape))
                raw.write(np.ascontiguousarray(frame).tobytes())
                rows.append({"sample_index": len(rows), "video": v.stem, "frame_index": i, **parse_name(v.stem)})
                n_sampled += 1
            i += 1
        cap.release()
        log(stage="extract", video=v.name, decoded_frames=i, sampled=n_sampled)
    raw.close()
    with open(out / "frames_shape.json", "w", encoding="utf-8") as f:
        json.dump({"n": len(rows), "frame_shape": list(shape), "dtype": "uint8", "layout": "N,H,W,3 BGR"}, f)
    with open(out / "frame_index.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # OpenPose reference for sampled frames
    H, W = shape[0], shape[1]
    ref = np.full((len(rows), len(OP_NAMES), 2), np.nan, dtype=np.float32)
    have_key = np.zeros(len(rows), dtype=bool)
    op_counts = {}
    cache = {}
    for r in rows:
        stem = r["video"].replace("G1A-Anon-", "")
        if stem not in cache:
            p = root / "openpose" / ("G1A-OP-" + stem + ".json")
            with open(p, "r", encoding="utf-8") as f:
                cache[stem] = json.load(f)["positions"]
            op_counts[stem] = len(cache[stem])
        pos = cache[stem]
        key = "%d.0" % (r["frame_index"] + 1)
        if key in pos:
            have_key[r["sample_index"]] = True
            for j, name in enumerate(OP_NAMES):
                xy = pos[key].get(name)
                if xy is None or len(xy) < 2:
                    continue
                x, y = float(xy[0]), float(xy[1])
                if not (np.isfinite(x) and np.isfinite(y)) or (x == 0 and y == 0):
                    continue
                ref[r["sample_index"], j] = (x * W, y * H)
    np.savez_compressed(out / "openpose_reference.npz", ref_xy=ref, have_key=have_key,
                        op_names=np.asarray(OP_NAMES), width=W, height=H)
    meta = {"status": "complete", "videos": len(videos), "sampled_frames": len(rows), "step": STEP,
            "frame_shape": list(shape), "openpose_key_rule": "key = str(frame_index+1)+'.0'",
            "openpose_scale_rule": "x*width, y*height",
            "openpose_frames_available": int(have_key.sum()),
            "openpose_entry_counts": op_counts,
            "frames_sha256": sha256_file(out / "frames_bgr.u8")}
    with open(done, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log(stage="extract", status="complete", sampled=len(rows), videos=len(videos))


# ----------------------------------------------------------------------------- infer
def load_frames(out):
    with open(out / "frames_shape.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return np.memmap(out / "frames_bgr.u8", dtype=np.uint8, mode="r", shape=tuple([meta["n"]] + meta["frame_shape"]))


def select_models(registry):
    ms = [m for m in registry["models"] if m["method"] in METHODS and abs(m["budget"] - 0.4) < 1e-9
          and m.get("provenance") == "original_reused"]
    if len(ms) != 18:
        raise RuntimeError("Expected 18 original b040 models, got %d" % len(ms))
    return ms


def infer(out, registry_path, device, batch):
    from ultralytics import YOLO
    import torch
    import ultralytics
    # Pin CUDA to the masked GPU before ultralytics.select_device overwrites CUDA_VISIBLE_DEVICES.
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit("CUDA_VISIBLE_DEVICES must be '1' (GPU 1 only)")
    torch.cuda.init()
    if torch.cuda.device_count() != 1:
        raise SystemExit("expected one visible GPU")
    log(stage="infer", cuda_pinned=True, gpu=torch.cuda.get_device_name(0))
    with open(registry_path, "r", encoding="utf-8-sig") as f:
        registry = json.load(f)
    models = select_models(registry)
    frames = load_frames(out)
    n = frames.shape[0]
    pred_dir = out / "predictions"
    pred_dir.mkdir(exist_ok=True)
    versions = {"python": sys.version, "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                "numpy": np.__version__}
    for m in models:
        target = pred_dir / (m["model_id"] + ".npz")
        if target.exists():
            log(model_id=m["model_id"], status="already_complete")
            continue
        ck = Path(m["checkpoint"])
        digest = sha256_file(ck)
        if digest != m["checkpoint_sha256"]:
            raise RuntimeError("checkpoint sha mismatch " + m["model_id"])
        model = YOLO(str(ck))
        shape = model.model.model[-1].kpt_shape
        if list(shape) != [26, 3]:
            raise RuntimeError("kpt_shape mismatch " + str(shape))
        kp = np.full((n, 26, 3), np.nan, dtype=np.float32)
        det = np.zeros(n, dtype=bool)
        bconf = np.full(n, np.nan, dtype=np.float32)
        ndet = np.zeros(n, dtype=np.int16)
        t0 = time.time()
        for s in range(0, n, batch):
            imgs = [np.ascontiguousarray(frames[k]) for k in range(s, min(s + batch, n))]
            res = model.predict(source=imgs, device=device, verbose=False, stream=False, **INFER)
            if len(res) != len(imgs):
                raise RuntimeError("predictor dropped frames")
            for j, r in enumerate(res):
                if r.boxes is None or r.keypoints is None or len(r.boxes) == 0:
                    continue
                conf = r.boxes.conf.detach().cpu().numpy().reshape(-1)
                k = r.keypoints.data.detach().cpu().numpy()
                idx = int(np.argmax(conf))
                kp[s + j] = k[idx]
                det[s + j] = True
                bconf[s + j] = float(conf[idx])
                ndet[s + j] = len(conf)
            if (s // batch) % 50 == 0:
                log(stage="infer", model_id=m["model_id"], done=min(s + batch, n), total=n)
        np.savez_compressed(target, keypoints=kp, detected=det, box_conf=bconf, n_detections=ndet,
                            model_id=m["model_id"], checkpoint_sha256=digest, inference=json.dumps(INFER),
                            versions=json.dumps(versions))
        log(stage="infer", model_id=m["model_id"], status="complete", seconds=round(time.time() - t0, 1),
            detection_rate=float(det.mean()))
        del model
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- report
def load_index(out):
    with open(out / "frame_index.csv", "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def agreement(kp, det, ref):
    """Return per-frame per-joint normalized distance to OpenPose (inf for missing pred)."""
    n = kp.shape[0]
    avail = np.isfinite(ref).all(axis=2)                      # [n,14]
    n_avail = avail.sum(axis=1)
    mins = np.nanmin(np.where(avail[:, :, None], ref, np.nan), axis=1)
    maxs = np.nanmax(np.where(avail[:, :, None], ref, np.nan), axis=1)
    with np.errstate(invalid="ignore"):
        scale = np.max(maxs - mins, axis=1)
    frame_ok = (n_avail >= 4) & np.isfinite(scale) & (scale > 0)
    ref12 = ref[:, MAP_OP_IDX]                                 # [n,12,2]
    pred12 = kp[:, MAP_REHAB_IDX, :2]
    mask = frame_ok[:, None] & avail[:, MAP_OP_IDX]           # evaluable reference joints
    finite = np.isfinite(pred12).all(axis=2) & det[:, None]
    dist = np.full(mask.shape, np.inf)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.linalg.norm(pred12 - ref12, axis=2) / scale[:, None]
    dist[mask & finite] = d[mask & finite]
    return {"mask": mask, "dist": dist, "frame_ok": frame_ok}


def summarize_model(entry, kp, det, bconf, ref, rows):
    ag = agreement(kp, det, ref)
    mask, dist = ag["mask"], ag["dist"]
    out = {"model_id": entry["model_id"], "method": entry["method"], "seed": entry["seed"],
           "frames": len(det), "detection_rate": float(det.mean()),
           "mean_box_conf_detected": float(np.nanmean(bconf)) if det.any() else None,
           "mean_kpt_conf_detected_26": float(np.nanmean(kp[det][:, :, 2])) if det.any() else None,
           "mean_kpt_conf_detected_12mapped": float(np.nanmean(kp[det][:, MAP_REHAB_IDX, 2])) if det.any() else None,
           "reference_evaluable_frames": int(ag["frame_ok"].sum()),
           "reference_evaluable_joints": int(mask.sum())}
    for tag, t in THR.items():
        out["agree_micro_" + tag] = float((mask & (dist <= t)).sum() / mask.sum()) if mask.sum() else None
    fin = np.isfinite(dist) & mask
    out["mean_norm_dist_detected_only"] = float(dist[fin].mean()) if fin.any() else None
    out["median_norm_dist_detected_only"] = float(np.median(dist[fin])) if fin.any() else None
    # per-video macro
    by_video = defaultdict(list)
    for i, r in enumerate(rows):
        by_video[r["video"]].append(i)
    per_video = []
    for v, idx in by_video.items():
        idx = np.asarray(idx)
        den = int(mask[idx].sum())
        rec = {"model_id": entry["model_id"], "method": entry["method"], "seed": entry["seed"], "video": v,
               **{k: rows[idx[0]][k] for k in ("exercise", "subject", "site")},
               "n_frames": len(idx), "n_detected": int(det[idx].sum()), "detection_rate": float(det[idx].mean()),
               "mean_kpt_conf_12mapped": float(np.nanmean(kp[idx][det[idx]][:, MAP_REHAB_IDX, 2])) if det[idx].any() else None,
               "ref_joints": den}
        for tag, t in THR.items():
            rec["agree_" + tag] = float((mask[idx] & (dist[idx] <= t)).sum() / den) if den else None
        per_video.append(rec)
    for tag in THR:
        vals = [r["agree_" + tag] for r in per_video if r["agree_" + tag] is not None]
        out["agree_video_macro_" + tag] = float(np.mean(vals)) if vals else None
    out["detection_rate_video_macro"] = float(np.mean([r["detection_rate"] for r in per_video]))
    # per joint micro agreement @0.2
    out["per_joint_agree_020"] = {MAP_NAMES[j]: (float((mask[:, j] & (dist[:, j] <= 0.2)).sum() / mask[:, j].sum())
                                                if mask[:, j].sum() else None) for j in range(12)}
    return out, per_video, ag


def draw_overlay(img, kp, ref, text):
    import cv2
    img = img.copy()
    # OpenPose reference (all 14) in yellow with thin edges
    name_to_xy = {}
    for j, name in enumerate(OP_NAMES):
        if np.isfinite(ref[j]).all():
            name_to_xy[name] = (int(round(ref[j, 0])), int(round(ref[j, 1])))
    for a, b in EDGES:
        if a in name_to_xy and b in name_to_xy:
            cv2.line(img, name_to_xy[a], name_to_xy[b], (0, 220, 255), 1, cv2.LINE_AA)
    for name, p in name_to_xy.items():
        cv2.circle(img, p, 3, (0, 220, 255), 1, cv2.LINE_AA)
    # Model prediction: 26 keypoints as small dots (green), mapped 12 with edges (red)
    if kp is not None and np.isfinite(kp[:, :2]).all():
        for j in range(26):
            cv2.circle(img, (int(round(kp[j, 0])), int(round(kp[j, 1]))), 2, (0, 200, 0), -1, cv2.LINE_AA)
        mapped = {name: (int(round(kp[idx, 0])), int(round(kp[idx, 1]))) for name, idx in OP_TO_REHAB.items()}
        for a, b in EDGES:
            if a in mapped and b in mapped:
                cv2.line(img, mapped[a], mapped[b], (0, 0, 255), 2, cv2.LINE_AA)
        for name, p in mapped.items():
            cv2.circle(img, p, 4, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "red=model(12 mapped) green=model(26) yellow=OpenPose ref (not GT)", (6, img.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "red=model(12 mapped) green=model(26) yellow=OpenPose ref (not GT)", (6, img.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def report(out, registry_path, overlay_dir):
    import cv2
    with open(registry_path, "r", encoding="utf-8-sig") as f:
        registry = json.load(f)
    models = select_models(registry)
    rows = load_index(out)
    with np.load(out / "openpose_reference.npz") as z:
        ref = z["ref_xy"]
    frames = load_frames(out)
    model_rows, video_rows = [], []
    preds = {}
    for m in models:
        with np.load(out / "predictions" / (m["model_id"] + ".npz")) as z:
            kp, det, bconf = z["keypoints"], z["detected"], z["box_conf"]
        s, pv, ag = summarize_model(m, kp, det, bconf, ref, rows)
        model_rows.append(s)
        video_rows.extend(pv)
        preds[m["model_id"]] = (kp, det, bconf)
    rep = out / "report"
    rep.mkdir(exist_ok=True)
    with open(rep / "per_model.json", "w", encoding="utf-8") as f:
        json.dump(model_rows, f, indent=2)
    with open(rep / "per_video.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(video_rows[0]))
        w.writeheader()
        w.writerows(video_rows)
    # per-method mean +- std over seeds
    keys = ["detection_rate", "detection_rate_video_macro", "mean_box_conf_detected", "mean_kpt_conf_detected_12mapped",
            "mean_kpt_conf_detected_26", "agree_micro_020", "agree_micro_010", "agree_video_macro_020",
            "agree_video_macro_010", "mean_norm_dist_detected_only", "median_norm_dist_detected_only"]
    method_rows = []
    for meth in METHODS:
        ms = [r for r in model_rows if r["method"] == meth]
        rec = {"method": meth, "n_seeds": len(ms)}
        for k in keys:
            vals = [r[k] for r in ms if r[k] is not None]
            rec[k + "_mean"] = float(np.mean(vals)) if vals else None
            rec[k + "_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        method_rows.append(rec)
    with open(rep / "per_method.json", "w", encoding="utf-8") as f:
        json.dump(method_rows, f, indent=2)
    # by exercise and by subject (detection rate + agree_020, mean over seeds of per-video macro)
    strata = {}
    for key in ("exercise", "subject"):
        for meth in METHODS:
            for level in sorted({r[key] for r in video_rows}):
                per_seed = []
                for seed in range(6):
                    sel = [r for r in video_rows if r["method"] == meth and r["seed"] == seed and r[key] == level]
                    d = np.mean([r["detection_rate"] for r in sel])
                    a = [r["agree_020"] for r in sel if r["agree_020"] is not None]
                    per_seed.append((d, np.mean(a) if a else np.nan))
                per_seed = np.asarray(per_seed)
                strata[f"{key}|{meth}|{level}"] = {"detection_rate_mean": float(per_seed[:, 0].mean()),
                                                   "detection_rate_std": float(per_seed[:, 0].std(ddof=1)),
                                                   "agree_020_mean": float(np.nanmean(per_seed[:, 1])),
                                                   "agree_020_std": float(np.nanstd(per_seed[:, 1], ddof=1)),
                                                   "n_videos": int(len(sel))}
    with open(rep / "by_stratum.json", "w", encoding="utf-8") as f:
        json.dump(strata, f, indent=2)
    # Diagnostic (not used for any selection): OpenPose frame-offset sensitivity for seed-0 models.
    # Shift the reference by -1/0/+1 sampled rows within the same video is NOT the same as frame offset,
    # so we skip; instead report per-joint agreement.
    # Overlays: seed-0 model per method, median mean-kpt-confidence detected frame within each fixed combo.
    overlay_dir.mkdir(parents=True, exist_ok=True)
    chosen = []
    for meth in METHODS:
        mid = f"{meth}_n_s0_b040"
        kp, det, bconf = preds[mid]
        for ex, subj in OVERLAY_COMBOS:
            idx = [i for i, r in enumerate(rows) if r["exercise"] == ex and r["subject"] == subj and det[i]]
            if not idx:
                chosen.append({"model_id": mid, "exercise": ex, "subject": subj, "frame": None, "note": "no detection"})
                continue
            confs = np.array([np.nanmean(kp[i, MAP_REHAB_IDX, 2]) for i in idx])
            order = np.argsort(confs)
            pick = idx[int(order[len(order) // 2])]
            r = rows[pick]
            text = f"{mid} {r['video']} f{r['frame_index']} meanconf={np.nanmean(kp[pick, MAP_REHAB_IDX, 2]):.2f}"
            img = draw_overlay(np.ascontiguousarray(frames[pick]), kp[pick], ref[pick], text)
            name = f"{meth}_s0_{r['video']}_f{int(r['frame_index']):05d}.png"
            cv2.imwrite(str(overlay_dir / name), img)
            chosen.append({"model_id": mid, "exercise": ex, "subject": subj, "video": r["video"],
                           "frame_index": int(r["frame_index"]), "file": name,
                           "mean_kpt_conf_12mapped": float(np.nanmean(kp[pick, MAP_REHAB_IDX, 2])),
                           "rule": "median of mean 12-joint keypoint confidence among detected frames of the combo"})
    with open(rep / "overlay_selection.json", "w", encoding="utf-8") as f:
        json.dump(chosen, f, indent=2)
    log(stage="report", status="complete", models=len(model_rows), videos=len({r['video'] for r in rows}))
    for r in method_rows:
        log(method=r["method"], det=r["detection_rate_mean"], agree020=r["agree_micro_020_mean"],
            agree010=r["agree_micro_010_mean"], kconf=r["mean_kpt_conf_detected_12mapped_mean"])


def smoke(out, registry_path, limit=64):
    """CPU-only end-to-end check on the first model and first `limit` frames; writes nothing persistent."""
    import cv2
    from ultralytics import YOLO
    with open(registry_path, "r", encoding="utf-8-sig") as f:
        registry = json.load(f)
    m = select_models(registry)[0]
    frames = load_frames(out)
    rows = load_index(out)[:limit]
    with np.load(out / "openpose_reference.npz") as z:
        ref = z["ref_xy"][:limit]
    model = YOLO(m["checkpoint"])
    kp = np.full((limit, 26, 3), np.nan, dtype=np.float32); det = np.zeros(limit, dtype=bool); bconf = np.full(limit, np.nan, dtype=np.float32)
    imgs = [np.ascontiguousarray(frames[k]) for k in range(limit)]
    for s in range(0, limit, 16):
        res = model.predict(source=imgs[s:s + 16], device="cpu", verbose=False, **INFER)
        for j, r in enumerate(res):
            if r.boxes is None or r.keypoints is None or len(r.boxes) == 0:
                continue
            conf = r.boxes.conf.cpu().numpy().reshape(-1); k = r.keypoints.data.cpu().numpy(); idx = int(np.argmax(conf))
            kp[s + j] = k[idx]; det[s + j] = True; bconf[s + j] = float(conf[idx])
    summ, pv, ag = summarize_model(m, kp, det, bconf, ref, rows)
    print(json.dumps({k: v for k, v in summ.items() if k != "per_joint_agree_020"}, indent=1))
    print("per_joint", summ["per_joint_agree_020"])
    print("ref available joints per frame (first 5):", np.isfinite(ref).all(axis=2).sum(axis=1)[:5].tolist())
    print("pred mapped xy frame0:", kp[0, MAP_REHAB_IDX, :2].round(1).tolist() if det[0] else None)
    print("ref mapped xy frame0:", ref[0, MAP_OP_IDX].round(1).tolist())
    i = int(np.argmax(det)) if det.any() else 0
    img = draw_overlay(np.ascontiguousarray(frames[i]), kp[i] if det[i] else None, ref[i], "smoke " + rows[i]["video"])
    (out / "smoke").mkdir(exist_ok=True)
    cv2.imwrite(str(out / "smoke" / "smoke_overlay.png"), img)
    print("overlay written", str(out / "smoke" / "smoke_overlay.png"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=r"rehabal_work\penn\keraal\extracted")
    p.add_argument("--out", default=r"rehabal_work\penn\keraal\work")
    p.add_argument("--registry", default=r"rehabal_work\penn\registry_server.json")
    p.add_argument("--overlays", default=r"rehabal_work\penn\keraal\overlays")
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=32)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--extract", action="store_true")
    g.add_argument("--infer", action="store_true")
    g.add_argument("--report", action="store_true")
    g.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    root, out = Path(a.root), Path(a.out)
    if a.extract:
        extract(root, out)
    elif a.infer:
        infer(out, a.registry, a.device, a.batch)
    elif a.smoke:
        smoke(out, a.registry)
    else:
        report(out, a.registry, Path(a.overlays))


if __name__ == "__main__":
    main()
