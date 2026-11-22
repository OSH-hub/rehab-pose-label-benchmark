r"""E3: evaluate best.pt AND last.pt of every round of one session on the fold's
val split (val-only yaml with absolute path). Resume-safe; appends to a per-session
rows CSV (merged later by e3_aggregate.py into e3_eval.csv).

Usage (repo root, the only CUDA process on its GPU):
  python rehabal_work\e3_eval.py --fold 1 --session-dir <dir> --condition A --seed 0 --out <rows.csv>
"""
import argparse, csv, os, time
from pathlib import Path

# Pin the CUDA context under the worker's CUDA_VISIBLE_DEVICES mask BEFORE
# ultralytics runs: select_device("0") rewrites CUDA_VISIBLE_DEVICES="0", which
# would re-target physical GPU 0 if the driver had not been initialised yet.
import torch  # noqa: E402
_MASK = os.environ.get("CUDA_VISIBLE_DEVICES")
torch.cuda.init()
_ = torch.zeros(1, device="cuda:0")
print(f"[e3_eval] CUDA_VISIBLE_DEVICES={_MASK} -> cuda:0 = {torch.cuda.get_device_name(0)} "
      f"(visible count {torch.cuda.device_count()})", flush=True)

WORK = Path(r"rehabal_work\e3_folds")
BUDGET = {0: 0.05, 1: 0.10, 2: 0.20, 3: 0.40}
FIELDS = ["fold", "condition", "seed", "round", "budget", "ckpt", "map50", "map50_95",
          "weights", "session", "imgsz", "n_val_images", "sec"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    yaml = WORK / "yamls" / f"fold{args.fold}_val.yaml"
    if not yaml.is_file():
        raise SystemExit(f"missing yaml {yaml}")
    sess = Path(args.session_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.is_file():
        with out.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((int(r["round"]), r["ckpt"]))
    todo = [(r, c) for r in range(4) for c in ("best", "last") if (r, c) not in done]
    if not todo:
        print("E3_EVAL_DONE (nothing to do)", out, flush=True)
        return 0

    from ultralytics import YOLO
    n_val = len(list((Path(__import__("yaml").safe_load(yaml.read_text())["path"]) / "images" / "val").glob("*.jpg")))
    new = not out.is_file()
    with out.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for rnd, ck in todo:
            wpath = sess / f"round_{rnd}" / "run" / "train" / "weights" / f"{ck}.pt"
            if not wpath.is_file():
                print("MISSING", wpath, flush=True)
                continue
            t0 = time.time()
            model = YOLO(str(wpath))
            res = model.val(data=str(yaml), imgsz=args.imgsz, batch=args.batch, device="0",
                            plots=False, verbose=False, workers=0,
                            project=str(WORK / "val_runs"),
                            name=f"fold{args.fold}_{sess.name}_r{rnd}_{ck}", exist_ok=True)
            row = dict(fold=args.fold, condition=args.condition, seed=args.seed, round=rnd,
                       budget=BUDGET[rnd], ckpt=ck, map50=float(res.pose.map50),
                       map50_95=float(res.pose.map), weights=str(wpath), session=sess.name,
                       imgsz=args.imgsz, n_val_images=n_val, sec=round(time.time() - t0, 1))
            w.writerow(row)
            fh.flush()
            print(f"fold{args.fold} {sess.name} r{rnd} {ck}: mAP50-95={row['map50_95']:.4f} "
                  f"mAP50={row['map50']:.4f} ({row['sec']}s)", flush=True)
            del model, res
            torch.cuda.empty_cache()
    print("E3_EVAL_DONE", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
