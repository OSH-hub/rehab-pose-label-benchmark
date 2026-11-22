r"""E1: same-frame label-source factorial on REHAB24-6 (YOLO11n).

For each seed s and round r in {1,2,3}, the frame list is FIXED to the frames the
original ref_s{s} arm actually trained on at that round (round_r/train.txt). Only
the label source changes; training is the unchanged cold-start 30-epoch recipe of
run_clinician_experiment.train_eval_fn. No acquisition loop, no rejection, no
top-up, so arm differences are pure label-source effects on identical frames.

Arms (label source for every frame in the fixed list):
  pseudo_A    : predictions of the original A_s{s}/round_{r-1} model (A's own teacher)
  pseudo_init : predictions of the shared round-0 model (A_s{s}/round_0, 5% GT init)
  gt_sigma20  : released GT keypoints + isotropic N(0, 20px) noise, clipped (vis kept)
  (gt         : identical to the original ref arm; not re-run, read from the grid)

Each trained model is validated on the Camera17 and Camera18 val splits at 416px.
Resume-safe. Run from the repo root:
  python rehabal_work\run_sameframe_e1.py --seeds 0 1 2 --arms pseudo_A pseudo_init gt_sigma20
"""
import argparse, csv, os, sys, time
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--source-root", type=Path, default=Path(r"RehabAL-pose"))
ap.add_argument("--grid", type=Path, default=Path("outputs/experiments/clinician_i4_uniform30"))
ap.add_argument("--out", type=Path, default=Path(r"rehabal_work\sameframe"))
ap.add_argument("--yaml-dir", type=Path, default=Path(r"rehabal_work\crossview"))
ap.add_argument("--arms", nargs="+", default=["pseudo_A", "pseudo_init", "gt_sigma20"])
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
ap.add_argument("--rounds", nargs="+", type=int, default=[1, 2, 3])
ap.add_argument("--sigma", type=float, default=20.0)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--imgsz", type=int, default=416)
ap.add_argument("--batch", type=int, default=16)
args = ap.parse_args()

sys.path.insert(0, str(args.source_root / "src"))
sys.path.insert(0, str(args.source_root / "scripts"))
os.chdir(args.source_root)
import run_clinician_experiment as runner  # noqa: E402
from ultralytics import YOLO  # noqa: E402

BUDGET = {0: 0.05, 1: 0.10, 2: 0.20, 3: 0.40}
runner.DATASET = (args.source_root / "outputs/rehab24_6_yolo").resolve()
runner.MANIFEST = (args.source_root / "outputs/rehab24_6_real/manifest.parquet").resolve()
runner.IMAGES_TRAIN = runner.DATASET / "images" / "train"
runner.SEG_CSV = (args.source_root / "data/rehab24_6_raw/Segmentation.csv").resolve()
pool_images = [str(p) for p in sorted(runner.IMAGES_TRAIN.glob("*.jpg"))]
metadata = runner._recording_meta(runner.ManifestBuilder.load(runner.MANIFEST))
infer_fn, _score_fn, train_eval_fn, gt_lookup = runner._make_callables(
    pool_images, metadata, "temporal", args.epochs, args.imgsz, args.batch, args.epochs, "yolo11n-pose.pt")

args.out.mkdir(parents=True, exist_ok=True)
csv_path = args.out / "sameframe_e1.csv"
fields = ["arm", "seed", "round", "budget", "n_frames", "cam", "map50", "map50_95", "weights", "train_sec"]
done = set()
if csv_path.exists():
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            done.add((r["arm"], int(r["seed"]), int(r["round"]), r["cam"]))
fh = open(csv_path, "a", newline=""); w = csv.DictWriter(fh, fieldnames=fields)
if fh.tell() == 0:
    w.writeheader()
yamls = {c: str(args.yaml_dir / f"{c}.yaml") for c in ("c17", "c18")}


def fixed_frames(seed, rnd):
    tl = args.grid / f"ref_s{seed}" / f"round_{rnd}" / "train.txt"
    stems = [Path(line.strip()).stem for line in tl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not stems:
        raise RuntimeError(f"empty frame list {tl}")
    return stems


def labels_for(arm, seed, rnd, stems):
    if arm == "pseudo_A":
        wpath = args.grid / f"A_s{seed}" / f"round_{rnd - 1}" / "run" / "train" / "weights" / "best.pt"
        return infer_fn(str(wpath), stems), str(wpath)
    if arm == "pseudo_init":
        wpath = args.grid / f"A_s{seed}" / "round_0" / "run" / "train" / "weights" / "best.pt"
        return infer_fn(str(wpath), stems), str(wpath)
    if arm.startswith("gt_sigma"):
        rng = np.random.default_rng(1000 * seed + rnd)
        out = {}
        for s in stems:
            kp = np.asarray(gt_lookup(s), dtype=np.float64)
            if kp.shape != (26, 3):
                continue
            wimg, himg = runner._get_image_size(runner.IMAGES_TRAIN / f"{s}.jpg")
            noisy = kp.copy()
            noisy[:, 0] = np.clip(kp[:, 0] + rng.normal(0, args.sigma, 26), 0, wimg - 1)
            noisy[:, 1] = np.clip(kp[:, 1] + rng.normal(0, args.sigma, 26), 0, himg - 1)
            out[s] = noisy.tolist()
        return out, f"gt+N(0,{args.sigma}px)"
    raise ValueError(arm)


for seed in args.seeds:
    for rnd in args.rounds:
        stems = fixed_frames(seed, rnd)
        for arm in args.arms:
            sess = args.out / f"{arm}_s{seed}" / f"round_{rnd}"
            best = sess / "run" / "train" / "weights" / "best.pt"
            t0 = time.time()
            if not best.exists():
                labels, src = labels_for(arm, seed, rnd, stems)
                missing = [s for s in stems if s not in labels]
                # A's own pipeline accepts an all-zero fallback for missed detections; mirror it.
                for s in missing:
                    labels[s] = np.zeros((26, 3)).tolist()
                sess.mkdir(parents=True, exist_ok=True)
                (sess / "label_source.txt").write_text(
                    f"arm={arm}\nseed={seed}\nround={rnd}\nn_frames={len(stems)}\nsource={src}\n"
                    f"missing_pred={len(missing)}\n", encoding="utf-8")
                print(f"[train] {arm} s{seed} r{rnd}: {len(stems)} frames, source={src}, missing={len(missing)}", flush=True)
                train_eval_fn(stems, labels, str(sess), seed)
                runner._free_cuda()
            train_sec = round(time.time() - t0, 1)
            if not best.exists():
                print(f"[FAIL] no best.pt for {arm} s{seed} r{rnd}", flush=True)
                continue
            for cam, y in yamls.items():
                if (arm, seed, rnd, cam) in done:
                    continue
                res = YOLO(str(best)).val(data=y, imgsz=args.imgsz, batch=32, device="0",
                                           plots=False, verbose=False, workers=0,
                                           project=str(args.out / "val_runs"), name=f"{arm}_s{seed}_r{rnd}_{cam}", exist_ok=True)
                w.writerow(dict(arm=arm, seed=seed, round=rnd, budget=BUDGET[rnd], n_frames=len(stems), cam=cam,
                                map50=float(res.pose.map50), map50_95=float(res.pose.map), weights=str(best), train_sec=train_sec))
                fh.flush()
                print(f"[eval ] {arm} s{seed} r{rnd} {cam}: mAP50-95={float(res.pose.map):.4f}", flush=True)
                runner._free_cuda()
            print(f"E1_SESSION_COMPLETE {arm} s{seed} r{rnd}", flush=True)
fh.close()
print("E1_DONE", csv_path)
