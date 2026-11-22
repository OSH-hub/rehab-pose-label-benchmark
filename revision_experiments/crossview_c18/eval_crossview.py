r"""Cross-view evaluation: every main-grid checkpoint (A, B sigma=0, ref; 6 seeds;
rounds 0-3) is validated on the Camera17 val split (sanity / reproduction) and on
the Camera18 val split (unseen view). Inference only; resume-safe.

Usage (from repo root):
  python rehabal_work\eval_crossview.py --device 1 --imgsz 416
"""
import argparse, csv, os, re, time
from pathlib import Path

BUDGET = {0: 0.05, 1: 0.10, 2: 0.20, 3: 0.40}
ap = argparse.ArgumentParser()
ap.add_argument("--exp", default="outputs/experiments/clinician_i4_uniform30")
ap.add_argument("--sessions", default=r"^(A_s\d|B_s\d_sig0|ref_s\d)$")
ap.add_argument("--c17", default="outputs/rehab24_6_yolo")
ap.add_argument("--c18", default="outputs/rehab24_6_yolo_c18")
ap.add_argument("--out", default=r"rehabal_work\crossview\crossview_11n.csv")
ap.add_argument("--device", default="1")
ap.add_argument("--imgsz", type=int, default=416)
ap.add_argument("--batch", type=int, default=32)
args = ap.parse_args()

from ultralytics import YOLO  # noqa: E402

out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

FLIP = "[0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 16, 17, 18, 19, 20]"


def mk_yaml(root: str, name: str) -> str:
    """Write a val-only dataset yaml with an absolute path (the repo's c17 data.yaml
    still points at an old machine path)."""
    r = Path(root).resolve().as_posix()
    y = out.parent / f"{name}.yaml"
    lines = [f"path: {r}", "train: images/val", "val: images/val", "",
             "kpt_shape: [26, 3]", f"flip_idx: {FLIP}", "", "names:", "  0: person", ""]
    y.write_text("\n".join(lines), encoding="utf-8")
    return str(y)


cams = {"c17": mk_yaml(args.c17, "c17"), "c18": mk_yaml(args.c18, "c18")}
print("yamls", cams, flush=True)

done = set()
if out.exists():
    with open(out, newline="") as f:
        for r in csv.DictReader(f):
            done.add((r["session"], int(r["round"]), r["cam"]))
fields = ["session", "condition", "seed", "round", "budget", "cam", "map50", "map50_95", "weights", "imgsz", "sec"]
new = not out.exists()
fh = open(out, "a", newline="")
w = csv.DictWriter(fh, fieldnames=fields)
if new:
    w.writeheader()

sess_re = re.compile(args.sessions)
sessions = sorted(d for d in os.listdir(args.exp) if sess_re.match(d))
print("sessions", len(sessions), sessions[:4], "...", flush=True)
for s in sessions:
    cond = s.split("_s")[0]
    seed = int(re.search(r"_s(\d)", s).group(1))
    for rnd in range(4):
        wpath = Path(args.exp) / s / f"round_{rnd}" / "run" / "train" / "weights" / "best.pt"
        if not wpath.exists():
            print("MISSING", wpath, flush=True)
            continue
        for cam, yaml in cams.items():
            if (s, rnd, cam) in done:
                continue
            t0 = time.time()
            model = YOLO(str(wpath))
            res = model.val(data=yaml, imgsz=args.imgsz, batch=args.batch, device=args.device,
                            plots=False, verbose=False, workers=0,
                            project=r"rehabal_work\crossview\runs",
                            name=f"{s}_r{rnd}_{cam}", exist_ok=True)
            pose = res.pose
            row = dict(session=s, condition=cond, seed=seed, round=rnd, budget=BUDGET[rnd], cam=cam,
                       map50=float(pose.map50), map50_95=float(pose.map), weights=str(wpath),
                       imgsz=args.imgsz, sec=round(time.time() - t0, 1))
            w.writerow(row)
            fh.flush()
            print(f"{s} r{rnd} {cam}: mAP50-95={row['map50_95']:.4f} mAP50={row['map50']:.4f} ({row['sec']}s)", flush=True)
fh.close()
print("EVAL_DONE", out)
