"""Usage: python scripts/run_clinician_isolated.py"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import queue
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
RUNNER = _HERE / "run_clinician_experiment.py"
DEFAULT_OUT = "outputs/experiments/clinician"
CONDITIONS = ["A", "B", "ref"]


def _run(extra, out, gpu=None):
    cmd = [sys.executable, str(RUNNER), "--out", out] + extra
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    tag = f"[gpu{gpu}] " if gpu is not None else ""
    print(f"\n=== RUN {tag}{' '.join(extra) or '(full aggregate)'} ===", flush=True)
    return subprocess.run(cmd, env=env).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2],
                    help="seeds to (re)run; completed sessions are skipped")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.0],
                    help="clinician correction-noise sigmas (px) for condition B "
                         "only; each sigma is its own isolated session. A/ref "
                         "ignore this (sigma-independent).")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--gpus", nargs="+", type=int, default=None,
                    help="physical GPU indices to parallelize across (e.g. "
                         "--gpus 0 1). One in-flight session PER GPU at a time. "
                         "Omit for the original single-GPU serial behaviour.")
    ap.add_argument("--weights", default=None,
                    help="backbone checkpoint forwarded to the base runner "
                         "(e.g. yolo11s-pose.pt). Omit to keep the default.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="epochs for rounds 1..N, forwarded to the base runner. "
                         "Omit to keep the base default (uniform 30).")
    ap.add_argument("--epochs-round0", type=int, default=None,
                    help="epochs for round 0, forwarded to the base runner. Set "
                         "equal to --epochs for a uniform-epoch schedule. Omit to "
                         "keep the base default (uniform 30).")
    ap.add_argument("--conf-threshold", type=float, default=None,
                    help="I3: confidence gate for condition A_conf, forwarded to "
                         "the base runner. Omit for the base default.")
    ap.add_argument("--verify-oks-threshold", type=float, default=None,
                    help="I1: OKS threshold for selective correction in condition "
                         "B, forwarded to the base runner. Predictions meeting the "
                         "threshold are kept; lower-OKS predictions are corrected.")
    ap.add_argument("--strategy", default=None,
                    help="I7: AL selector forwarded to the base runner "
                         "(random/uncertainty/coreset/temporal/rehabal). Omit to "
                         "keep the base default (temporal).")
    args = ap.parse_args(argv)

    jobs = []
    for seed in args.seeds:
        for cond in args.conditions:
            if cond == "B":
                jobs.extend((seed, cond, sigma) for sigma in args.sigmas)
            else:
                jobs.append((seed, cond, None))
    results: dict = {}
    weights_extra = ["--weights", args.weights] if args.weights else []
    epoch_extra = []
    if args.epochs is not None:
        epoch_extra += ["--epochs", str(args.epochs)]
    if args.epochs_round0 is not None:
        epoch_extra += ["--epochs-round0", str(args.epochs_round0)]
    if args.conf_threshold is not None:
        epoch_extra += ["--conf-threshold", str(args.conf_threshold)]
    if args.verify_oks_threshold is not None:
        epoch_extra += ["--verify-oks-threshold", str(args.verify_oks_threshold)]
    if args.strategy is not None:
        epoch_extra += ["--strategy", args.strategy]

    def _session(seed, cond, sigma, gpu):
        extra = ["--seeds", str(seed), "--conditions", cond]
        if sigma is not None:
            extra += ["--sigmas", f"{sigma:g}"]
        extra += weights_extra + epoch_extra
        rc = _run(extra, args.out, gpu=gpu)
        sig_tag = f"_sig{sigma:g}" if sigma is not None else ""
        g = f" (gpu{gpu})" if gpu is not None else ""
        print(f"  -> session s{seed} {cond}{sig_tag}{g} exit={rc}", flush=True)
        return f"s{seed}_{cond}{sig_tag}", rc

    if args.gpus and len(args.gpus) > 1:
        gpus = list(args.gpus)
        gpu_pool: "queue.Queue[int]" = queue.Queue()
        for g in gpus:
            gpu_pool.put(g)

        def _session_pooled(seed, cond, sigma):
            gpu = gpu_pool.get()
            try:
                return _session(seed, cond, sigma, gpu)
            finally:
                gpu_pool.put(gpu)

        with cf.ThreadPoolExecutor(max_workers=len(gpus)) as ex:
            futs = {ex.submit(_session_pooled, seed, cond, sigma): (seed, cond, sigma)
                    for seed, cond, sigma in jobs}
            for fut in cf.as_completed(futs):
                key, rc = fut.result()
                results[key] = rc
    else:
        gpu = args.gpus[0] if args.gpus else None
        for seed, cond, sigma in jobs:
            key, rc = _session(seed, cond, sigma, gpu)
            results[key] = rc

    print("\n=== AGGREGATE (full grid, all skipped, rebuilds master CSV) ===",
          flush=True)
    agg_extra = (["--seeds", *[str(s) for s in args.seeds]]
                 + ["--conditions", *args.conditions]
                 + ["--sigmas", *[f"{s:g}" for s in args.sigmas]])
    _run(agg_extra, args.out)

    print("\nISOLATED_RUN_DONE", results, flush=True)
    bad = [k for k, v in results.items() if v != 0]
    if bad:
        print("STILL_FAILED:", bad, "(re-run this script to retry)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
