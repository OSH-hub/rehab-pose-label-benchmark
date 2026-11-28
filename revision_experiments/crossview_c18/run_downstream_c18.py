import argparse, os, subprocess, sys
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--root", default="outputs/experiments/clinician_i4_uniform30")
ap.add_argument("--outdir", default="outputs/experiments/downstream_11n_u30_c18")
ap.add_argument("--val", default="outputs/rehab24_6_yolo_c18/images/val")
ap.add_argument("--lab", default="outputs/rehab24_6_yolo_c18/labels/val")
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
ap.add_argument("--gpu", default="1")
a = ap.parse_args()
env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES=a.gpu)
Path(a.outdir).mkdir(parents=True, exist_ok=True)
suffix = {"A": "A_s{}", "B": "B_s{}_sig0", "ref": "ref_s{}"}
for seed in a.seeds:
    for cond in ("A", "B", "ref"):
        sess = Path(a.root) / suffix[cond].format(seed)
        out = Path(a.outdir) / f"{cond}_s{seed}_downstream.json"
        if out.exists():
            print("[skip]", out); continue
        wl = [sess / f"round_{r}" / "run" / "train" / "weights" / "best.pt" for r in range(4)]
        if not all(w.exists() for w in wl):
            print("[warn] missing best.pt in", sess); continue
        cmd = [sys.executable, "scripts/run_downstream_eval.py", "--weights-list", *map(str, wl),
               "--budgets", "0.05", "0.1", "0.2", "0.4", "--val-dir", a.val, "--labels-dir", a.lab, "--out", str(out)]
        print("[run ]", cond, seed, flush=True)
        rc = subprocess.call(cmd, env=env)
        print("[ok  ]" if rc == 0 else "[FAIL]", out, flush=True)
print("DOWNSTREAM_C18_DONE")
