import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

W, H = 1280, 720
LOWER = ("hip", "knee", "ankle")
UPPER = ("shoulder", "elbow", "wrist")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_txt(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            v = [float(x) for x in s.replace(",", " ").split()]
            if len(v) != 6:
                raise ValueError(f"{path}: expected 6 columns, got {len(v)}")
            rows.append(v)
    arr = np.asarray(rows, dtype=np.float64).reshape(-1, 3, 2)
    return arr


def angle_deg(p0, p1, p2):
    a = p0 - p1
    b = p2 - p1
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (a * b).sum(-1) / (na * nb)
    c = np.clip(c, -1.0, 1.0)
    out = np.degrees(np.arccos(c))
    out = np.where((na > 0) & (nb > 0), out, np.nan)
    return out


def process_clip(args):
    clip, data_dir, out_dir, stride, offset = args
    s, e, k = int(clip["subject_id"]), int(clip["exercise_id"]), int(clip["camera_id"])
    tag = f"{s}_{e:02d}_cam{k}"
    rec_path = Path(out_dir) / "clips" / (tag + ".json")
    if rec_path.exists():
        return json.load(open(rec_path, encoding="utf-8"))
    video = Path(data_dir) / "data" / clip["video"]
    txt = Path(data_dir) / "data" / clip["p2d_path"]
    gt = load_txt(txt)
    if gt.shape[0] != int(clip["gt_rows"]):
        raise RuntimeError(f"{tag}: TXT rows {gt.shape[0]} != validation gt_rows {clip['gt_rows']}")
    rows = list(range(0, gt.shape[0], stride))
    need = {r + offset: r for r in rows}
    frame_dir = Path(out_dir) / "frames" / str(s) / f"{e:02d}" / f"cam{k}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"{tag}: cannot open video")
    decoded = 0
    got = {}
    while True:
        ok = cap.grab()
        if not ok:
            break
        if decoded in need:
            ok2, img = cap.retrieve()
            if not ok2 or img is None:
                raise RuntimeError(f"{tag}: retrieve failed at frame {decoded}")
            if img.shape[0] != H or img.shape[1] != W:
                raise RuntimeError(f"{tag}: unexpected frame size {img.shape}")
            p = frame_dir / f"f{decoded:06d}.png"
            if not cv2.imwrite(str(p), img, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise RuntimeError(f"{tag}: imwrite failed {p}")
            got[decoded] = {"image_path": str(p), "image_sha256": sha256_file(p), "image_bytes": p.stat().st_size}
        decoded += 1
    cap.release()
    frames = []
    for r in rows:
        f = r + offset
        if f not in got:
            continue
        xy = gt[r]
        oob = [bool(x < 0 or x >= W or y < 0 or y >= H) for x, y in xy]
        frames.append({"frame_key": f"{tag}/f{f:06d}", "txt_row": r, "rgb_frame": f,
                       "gt_xy": xy.tolist(), "gt_oob": oob, "gt_angle": float(angle_deg(xy[0], xy[1], xy[2])),
                       **got[f]})
    missing = [r + offset for r in rows if (r + offset) not in got]
    rec = {"tag": tag, "subject_id": s, "exercise_id": e, "camera_id": k, "body": clip["body"], "side": clip["side"],
           "joint_order": clip["joint_order"], "p3d_present": bool(clip["p3d_present"]),
           "video": clip["video"], "p2d_path": clip["p2d_path"], "video_sha256": sha256_file(video),
           "p2d_sha256": sha256_file(txt), "gt_rows": int(gt.shape[0]), "ffprobe_frames": int(clip["video_frames"]),
           "decoded_frames": decoded, "decoded_minus_gt_rows": decoded - int(gt.shape[0]),
           "sampled_rows": len(rows), "scored_frames": len(frames), "missing_rgb_frames": missing,
           "out_of_image_rows_total": int(clip["out_of_image_rows"]), "frames": frames}
    tmp = rec_path.with_suffix(".tmp")
    json.dump(rec, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(tmp, rec_path)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--offset", type=int, default=5)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--clips", default="", help="comma list subj/ex/cam to restrict (pilot)")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    val_path = Path(a.data) / "meta" / "dataset_validation.json"
    val = json.load(open(val_path, encoding="utf-8-sig"))
    clips = [c for c in val["clips"] if c["p2d_present"]]
    no_p2d = [c for c in val["clips"] if not c["p2d_present"]]
    if a.clips:
        want = set(a.clips.split(","))
        clips = [c for c in clips if f"{c['subject_id']}/{int(c['exercise_id']):02d}/{c['camera_id']}" in want]
        if len(clips) != len(want):
            raise RuntimeError("some requested pilot clips not found or lack p2d")
    for c in clips:
        jo = c["joint_order"].split(",")
        exp = LOWER if c["body"] == "lower" else UPPER
        if tuple(jo) != exp:
            raise RuntimeError(f"unexpected joint order {jo} for {c['video']}")
    t0 = time.time()
    recs = []
    jobs = [(c, a.data, str(out), a.stride, a.offset) for c in clips]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(process_clip, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            recs.append(f.result())
            if (i + 1) % 50 == 0 or i + 1 == len(futs):
                print(json.dumps({"done": i + 1, "total": len(futs), "elapsed_s": round(time.time() - t0, 1)}), flush=True)
    recs.sort(key=lambda r: (r["subject_id"], r["exercise_id"], r["camera_id"]))
    keys, gt_xy, oob, meta_rows = [], [], [], []
    with open(out / "manifest.jsonl", "w", encoding="utf-8", newline="\n") as mf:
        for r in recs:
            for fr in r["frames"]:
                row = {"frame_key": fr["frame_key"], "subject_id": r["subject_id"], "exercise_id": r["exercise_id"],
                       "camera_id": r["camera_id"], "body": r["body"], "side": r["side"], "joint_order": r["joint_order"],
                       "txt_row": fr["txt_row"], "rgb_frame": fr["rgb_frame"], "image_path": fr["image_path"],
                       "image_sha256": fr["image_sha256"], "gt_xy": fr["gt_xy"], "gt_oob": fr["gt_oob"],
                       "gt_angle": fr["gt_angle"]}
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")
                keys.append(fr["frame_key"]); gt_xy.append(fr["gt_xy"]); oob.append(fr["gt_oob"])
                meta_rows.append([r["subject_id"], r["exercise_id"], r["camera_id"], 0 if r["body"] == "lower" else 1,
                                  0 if r["side"] == "left" else 1, fr["txt_row"], fr["rgb_frame"]])
    gt_xy = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 3, 2)
    gt_angle = angle_deg(gt_xy[:, 0], gt_xy[:, 1], gt_xy[:, 2])
    np.savez_compressed(out / "gt.npz", frame_keys=np.asarray(keys), gt_xy=gt_xy, gt_oob=np.asarray(oob, dtype=bool),
                        gt_angle=gt_angle, meta=np.asarray(meta_rows, dtype=np.int64),
                        meta_columns=np.asarray(["subject_id", "exercise_id", "camera_id", "body(0=lower,1=upper)",
                                                 "side(0=left,1=right)", "txt_row", "rgb_frame"]))
    decode_mismatch = [(r["tag"], r["ffprobe_frames"], r["decoded_frames"]) for r in recs if r["decoded_frames"] != r["ffprobe_frames"]]
    missing = [(r["tag"], r["missing_rgb_frames"]) for r in recs if r["missing_rgb_frames"]]
    diffs = {}
    for r in recs:
        diffs[str(r["decoded_minus_gt_rows"])] = diffs.get(str(r["decoded_minus_gt_rows"]), 0) + 1
    meta = {"status": "complete", "stride": a.stride, "offset": a.offset, "clips_scored": len(recs),
            "clips_without_p2d_excluded": len(no_p2d) if not a.clips else None,
            "sequences_without_any_p2d": len({(c["subject_id"], c["exercise_id"]) for c in no_p2d}
                                              - {(c["subject_id"], c["exercise_id"]) for c in val["clips"] if c["p2d_present"]}) if not a.clips else None,
            "frames_scored": len(keys), "subjects": sorted({r["subject_id"] for r in recs}),
            "decoded_minus_gt_rows_histogram": diffs, "decode_count_mismatch_vs_ffprobe": decode_mismatch,
            "clips_with_missing_rgb_frames": missing, "gt_oob_frames": int(np.asarray(oob).any(axis=1).sum()),
            "gt_oob_joints": int(np.asarray(oob).sum()), "validation_json_sha256": sha256_file(val_path),
            "manifest_sha256": sha256_file(out / "manifest.jsonl"), "gt_npz_sha256": sha256_file(out / "gt.npz"),
            "prepare_code_sha256": sha256_file(Path(__file__)), "opencv_version": cv2.__version__,
            "frame_format": "PNG lossless 1280x720 BGR, OpenCV sequential grab/retrieve", "elapsed_s": round(time.time() - t0, 1),
            "prediction_performed": False}
    json.dump(meta, open(out / "prepare_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("decode_count_mismatch_vs_ffprobe", "clips_with_missing_rgb_frames")}, ensure_ascii=False))
    print("decode mismatches:", len(decode_mismatch), "missing-frame clips:", len(missing))


if __name__ == "__main__":
    main()
