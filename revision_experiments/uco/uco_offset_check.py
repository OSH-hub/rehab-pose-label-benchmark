import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uco_prepare import load_txt  # noqa: E402

MAP = {("lower", "left"): [16, 17, 18], ("lower", "right"): [21, 22, 23],
       ("upper", "left"): [7, 8, 9], ("upper", "right"): [12, 13, 14]}
OFFSETS = list(range(0, 11))


def read_frames(video, wanted):
    cap = cv2.VideoCapture(str(video))
    out = {}
    i = 0
    mx = max(wanted)
    while i <= mx:
        if not cap.grab():
            break
        if i in wanted:
            ok, img = cap.retrieve()
            if ok:
                out[i] = img
        i += 1
    cap.release()
    return out


def draw(img, xy, color, r=6):
    pts = [tuple(int(round(v)) for v in p) for p in xy]
    for p in pts:
        cv2.circle(img, p, r, color, 2)
    cv2.line(img, pts[0], pts[1], color, 2)
    cv2.line(img, pts[1], pts[2], color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--window", type=int, default=40, help="frames around the peak-motion row for the model curve")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    val = json.load(open(Path(a.data) / "meta" / "dataset_validation.json", encoding="utf-8-sig"))
    index = {f"{c['subject_id']}/{int(c['exercise_id']):02d}/{c['camera_id']}": c for c in val["clips"] if c["p2d_present"]}
    model = None
    if a.model:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        from ultralytics import YOLO
        model = YOLO(a.model)
    curve = {}
    for key in a.clips.split(","):
        c = index[key]
        tag = f"{c['subject_id']}_{int(c['exercise_id']):02d}_cam{c['camera_id']}"
        gt = load_txt(Path(a.data) / "data" / c["p2d_path"])
        n = gt.shape[0]
        vel = np.linalg.norm(np.diff(gt, axis=0), axis=2).sum(1)
        sm = np.convolve(vel, np.ones(5) / 5, mode="same")
        lo, hi = 15, n - 16
        r = int(lo + np.argmax(sm[lo:hi]))
        wanted = set(r + o for o in OFFSETS)
        if model is not None:
            w0, w1 = max(0, r - a.window), min(n - 1, r + a.window)
            wanted |= set(range(w0, w1 + 1 + OFFSETS[-1]))
        frames = read_frames(Path(a.data) / "data" / c["video"], wanted)
        xy = gt[r]
        cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
        half = int(max(160, np.abs(xy - [cx, cy]).max() * 1.6))
        tiles = []
        for o in OFFSETS:
            f = r + o
            img = frames[f].copy()
            draw(img, xy, (0, 0, 255))
            x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
            x1, y1 = int(min(1280, cx + half)), int(min(720, cy + half))
            crop = img[y0:y1, x0:x1]
            crop = cv2.resize(crop, (320, int(320 * crop.shape[0] / crop.shape[1])))
            cv2.putText(crop, f"off={o} rgb={f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            tiles.append(crop)
        hmax = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
        row1 = np.concatenate(tiles[:6], axis=1)
        row2 = np.concatenate(tiles[6:] + [np.zeros_like(tiles[0])], axis=1)
        grid = np.concatenate([row1, row2], axis=0)
        cv2.putText(grid, f"{tag} {c['body']}/{c['side']} GT row {r} (peak motion)", (5, grid.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imwrite(str(out / f"offset_grid_{tag}.png"), grid)
        if model is not None:
            jm = MAP[(c["body"], c["side"])]
            w0, w1 = max(0, r - a.window), min(n - 1, r + a.window)
            fids = list(range(w0, w1 + 1 + OFFSETS[-1]))
            pred = {}
            for s in range(0, len(fids), 8):
                batch = fids[s:s + 8]
                res = model.predict(source=[frames[f] for f in batch], imgsz=640, conf=0.05, iou=0.70, max_det=300,
                                    classes=[0], augment=False, half=False, verbose=False, device="cpu")
                for f, rr in zip(batch, res):
                    if rr.boxes is None or rr.keypoints is None or len(rr.boxes) == 0:
                        continue
                    bi = int(np.argmax(rr.boxes.conf.cpu().numpy()))
                    kp = rr.keypoints.data.cpu().numpy()[bi]
                    pred[f] = kp[jm, :2]
            errs = {}
            for o in OFFSETS:
                e = []
                for rr_ in range(w0, w1 + 1):
                    f = rr_ + o
                    if f in pred:
                        e.append(float(np.linalg.norm(pred[f] - gt[rr_], axis=1).mean()))
                errs[o] = (float(np.mean(e)) if e else None, len(e))
            cands = [(v[0], o) for o, v in errs.items() if v[0] is not None]
            best = min(cands)[1] if cands else None
            curve[tag] = {"body": c["body"], "side": c["side"], "peak_row": r, "window": [w0, w1],
                          "frames_predicted": len(pred), "frames_in_window": len(fids),
                          "mean_px_err_by_offset": {str(o): errs[o][0] for o in OFFSETS},
                          "n_by_offset": {str(o): errs[o][1] for o in OFFSETS}, "best_offset": best}
            print(tag, "detected", len(pred), "/", len(fids), "best offset", best,
                  {o: round(errs[o][0], 1) if errs[o][0] else None for o in OFFSETS}, flush=True)
        else:
            print(tag, "grid written, peak row", r, flush=True)
    if curve:
        json.dump(curve, open(out / "offset_curve.json", "w", encoding="utf-8"), indent=2)
        votes = {}
        for v in curve.values():
            votes[str(v["best_offset"])] = votes.get(str(v["best_offset"]), 0) + 1
        print("best-offset votes:", votes)


if __name__ == "__main__":
    main()
