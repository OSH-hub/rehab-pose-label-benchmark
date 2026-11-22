r"""REHAB26-ViewAngles (Zenodo 23018128, CC BY-NC 4.0): build the scored-frame manifest and PNG cache.

Never predicts. Steps: (1) extract videos.zip (once) into <work>/data; (2) for each of the 45
recordings x 3 cameras, take every 20th video frame (frame % 20 == 0) that lies inside a labelled
repetition of Segmentation.csv; attach the repetition's person orientation o and the camera's
viewing angle (cam1: -o, cam2: 45-o, cam3: 90-o) and its |angle| bin (0, 10-30, 40-60, 70-90);
(3) decode frames sequentially with OpenCV and store lossless PNG with SHA256; (4) attach the
MediaPipe 2D reference (jointSequences-mediapipe, <video_id>_cam<k>_2d.npy, [frames,33,4] =
x,y,z,visibility). MediaPipe is a reference estimator, NOT ground truth.

Usage: python va_prepare.py --dataset <zenodo dir> --work <dir> [--stride 20] [--workers 8]
"""
import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def view_angle(cam, o):
    return {1: -o, 2: 45 - o, 3: 90 - o}[cam]


def angle_bin(v):
    a = abs(v)
    if a == 0:
        return "0"
    if a <= 30:
        return "10-30"
    if a <= 60:
        return "40-60"
    return "70-90"


def process_recording(args):
    video_id, ex, cam, video_path, mp_path, reps, stride, out_dir = args
    tag = f"{video_id}_cam{cam}"
    rec_path = Path(out_dir) / "clips" / (tag + ".json")
    if rec_path.exists():
        return json.load(open(rec_path, encoding="utf-8"))
    mp = np.load(mp_path, allow_pickle=False)
    if mp.ndim != 3 or mp.shape[1] != 33 or mp.shape[2] != 4:
        raise RuntimeError(f"{tag}: unexpected mediapipe shape {mp.shape}")
    # frame -> repetition info
    want = {}
    for r in reps:
        for f in range(int(r["first_frame"]), int(r["last_frame"]) + 1):
            if f % stride == 0:
                if f in want:
                    raise RuntimeError(f"{tag}: overlapping repetitions at frame {f}")
                want[f] = r
    frame_dir = Path(out_dir) / "frames" / f"ex{ex}" / tag
    frame_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"{tag}: cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    decoded = 0
    got = {}
    width = height = None
    while True:
        if not cap.grab():
            break
        if decoded in want:
            ok, img = cap.retrieve()
            if not ok or img is None:
                raise RuntimeError(f"{tag}: retrieve failed at {decoded}")
            height, width = img.shape[:2]
            p = frame_dir / f"f{decoded:06d}.png"
            if not cv2.imwrite(str(p), img, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise RuntimeError(f"{tag}: imwrite failed")
            got[decoded] = {"image_path": str(p), "image_sha256": sha256_file(p)}
        decoded += 1
    cap.release()
    frames = []
    for f in sorted(want):
        if f not in got or f >= mp.shape[0]:
            continue
        r = want[f]
        o = int(r["person_orientation"])
        va = view_angle(cam, o)
        frames.append({"frame_key": f"{tag}/f{f:06d}", "frame": f, "repetition_number": int(r["repetition_number"]),
                       "person_orientation": o, "view_angle": va, "angle_bin": angle_bin(va),
                       "correctness_score": int(r["correctness_score"]), "error_description": r["error_description"],
                       "mp_xy": mp[f, :, :2].tolist(), "mp_vis": mp[f, :, 3].tolist(), **got[f]})
    rec = {"tag": tag, "video_id": video_id, "exercise_id": ex, "camera": cam, "person": reps[0]["person"],
           "video": str(video_path), "video_sha256": sha256_file(video_path), "mediapipe_npy": str(mp_path),
           "mediapipe_sha256": sha256_file(mp_path), "mediapipe_frames": int(mp.shape[0]), "decoded_frames": decoded,
           "fps": fps, "width": width, "height": height, "repetitions": len(reps), "wanted": len(want), "scored_frames": len(frames),
           "missing_frames": sorted(f for f in want if f not in got), "frames_beyond_mediapipe": sorted(f for f in want if f in got and f >= mp.shape[0]),
           "mp_xy_max": [float(np.nanmax(mp[:, :, 0])), float(np.nanmax(mp[:, :, 1]))], "frames": frames}
    tmp = rec_path.with_suffix(".tmp")
    json.dump(rec, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(tmp, rec_path)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    ds = Path(a.dataset)
    work = Path(a.work)
    data = work / "data"
    out = work / "prepared"
    (out / "clips").mkdir(parents=True, exist_ok=True)
    man = json.load(open(ds / "download_manifest.json", encoding="utf-8"))
    if not all(v["match"] for v in man["files"].values()):
        raise RuntimeError("download manifest reports an MD5 mismatch")
    t0 = time.time()
    for zname, marker in (("videos.zip", "videos.extracted"), ("jointSequences-mediapipe.zip", "mediapipe.extracted")):
        if not (data / marker).exists():
            data.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(ds / zname) as z:
                for n in z.namelist():
                    if n.startswith("/") or ".." in n.split("/"):
                        raise RuntimeError("unsafe zip member " + n)
                z.extractall(str(data))
            (data / marker).write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
            print("extracted", zname, round(time.time() - t0, 1), flush=True)
    with open(ds / "Segmentation.csv", encoding="utf-8-sig") as f:
        seg = list(csv.DictReader(f, delimiter=";"))
    by_video = {}
    for r in seg:
        by_video.setdefault(r["video_id"], []).append(r)
    jobs = []
    for vid, reps in sorted(by_video.items()):
        ex = int(reps[0]["exercise_id"])
        for cam in (1, 2, 3):
            vp = data / "videos" / f"ex{ex}" / f"{vid}_cam{cam}.mp4"
            mpp = data / "mediapipe" / f"ex{ex}" / f"{vid}_cam{cam}_2d.npy"
            if not vp.exists():
                cands = list((data / "videos").rglob(f"{vid}_cam{cam}.mp4"))
                if not cands:
                    raise FileNotFoundError(str(vp))
                vp = cands[0]
            if not mpp.exists():
                cands = list((data / "mediapipe").rglob(f"{vid}_cam{cam}_2d.npy"))
                if not cands:
                    raise FileNotFoundError(str(mpp))
                mpp = cands[0]
            jobs.append((vid, ex, cam, vp, mpp, reps, a.stride, str(out)))
    recs = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex_:
        futs = [ex_.submit(process_recording, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            recs.append(f.result())
            print(json.dumps({"done": i + 1, "total": len(futs), "elapsed_s": round(time.time() - t0, 1)}), flush=True)
    recs.sort(key=lambda r: (r["exercise_id"], r["person"], r["camera"]))
    keys, mp_xy, mp_vis, meta_rows = [], [], [], []
    persons = sorted({r["person"] for r in recs})
    with open(out / "manifest.jsonl", "w", encoding="utf-8", newline="\n") as mf:
        for r in recs:
            for fr in r["frames"]:
                row = {"frame_key": fr["frame_key"], "video_id": r["video_id"], "exercise_id": r["exercise_id"], "camera": r["camera"],
                       "person": r["person"], "frame": fr["frame"], "repetition_number": fr["repetition_number"],
                       "person_orientation": fr["person_orientation"], "view_angle": fr["view_angle"], "angle_bin": fr["angle_bin"],
                       "correctness_score": fr["correctness_score"], "width": r["width"], "height": r["height"],
                       "image_path": fr["image_path"], "image_sha256": fr["image_sha256"], "mp_xy": fr["mp_xy"], "mp_vis": fr["mp_vis"]}
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")
                keys.append(fr["frame_key"]); mp_xy.append(fr["mp_xy"]); mp_vis.append(fr["mp_vis"])
                meta_rows.append([r["exercise_id"], persons.index(r["person"]), r["camera"], fr["frame"], fr["repetition_number"],
                                  fr["person_orientation"], fr["view_angle"], fr["correctness_score"], r["width"], r["height"]])
    np.savez_compressed(out / "reference.npz", frame_keys=np.asarray(keys), mp_xy=np.asarray(mp_xy, dtype=np.float64),
                        mp_vis=np.asarray(mp_vis, dtype=np.float64), meta=np.asarray(meta_rows, dtype=np.int64),
                        meta_columns=np.asarray(["exercise_id", "person_index", "camera", "frame", "repetition_number", "person_orientation",
                                                 "view_angle", "correctness_score", "width", "height"]), persons=np.asarray(persons))
    bins = {}
    for r in meta_rows:
        b = angle_bin(r[6]); bins[b] = bins.get(b, 0) + 1
    meta = {"status": "complete", "stride": a.stride, "recordings": len(recs), "frames_scored": len(keys), "persons": persons,
            "angle_bin_counts": bins, "resolutions": sorted({f"{r['width']}x{r['height']}" for r in recs}), "fps": sorted({round(r["fps"], 3) for r in recs}),
            "mediapipe_xy_max_overall": [max(r["mp_xy_max"][0] for r in recs), max(r["mp_xy_max"][1] for r in recs)],
            "recordings_with_missing_frames": [(r["tag"], len(r["missing_frames"])) for r in recs if r["missing_frames"]],
            "recordings_with_frames_beyond_mediapipe": [(r["tag"], len(r["frames_beyond_mediapipe"])) for r in recs if r["frames_beyond_mediapipe"]],
            "decoded_vs_mediapipe_frames": [(r["tag"], r["decoded_frames"], r["mediapipe_frames"]) for r in recs if r["decoded_frames"] != r["mediapipe_frames"]],
            "segmentation_sha256": sha256_file(ds / "Segmentation.csv"), "download_manifest": man,
            "manifest_sha256": sha256_file(out / "manifest.jsonl"), "reference_npz_sha256": sha256_file(out / "reference.npz"),
            "prepare_code_sha256": sha256_file(Path(__file__)), "opencv_version": cv2.__version__, "elapsed_s": round(time.time() - t0, 1),
            "reference_is_ground_truth": False, "license": "CC BY-NC 4.0", "prediction_performed": False}
    json.dump(meta, open(out / "prepare_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("download_manifest",)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
