"""Run clinician sessions each in a FRESH subprocess (CUDA-OOM isolation).

A single long-lived process accumulates CUDA fragmentation; once one training
OOMs, the poisoned context makes every later training in that process OOM too
(observed 2026-06-19: all 6 seed-1/2 sessions failed within seconds after the
seed-0 sessions completed). Running each ``(seed, condition)`` session as its
own subprocess gives each a clean CUDA context and reclaims all VRAM on exit,
so a transient OOM kills only that one retriable session.

After the per-session passes, a final FULL-grid pass runs the base runner with
every session already complete (so all are skipped — no GPU work) purely to
rebuild the aggregated master ``clinician_curve.csv`` (per-session filtered runs
overwrite the master with only their own rows; the per-session ``curve.csv``
files are the durable source of truth).

Resume-safe: completed sessions are skipped by the base runner. Re-run this
script to mop up any sessions that still failed (e.g. transient VRAM contention).

Usage::
    python scripts/run_clinician_isolated.py                 # seeds 1 2 (default)
    python scripts/run_clinician_isolated.py --seeds 1 2 3 4 # extend
"""

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
    """Run one base-runner subprocess, optionally pinned to a single GPU.

    Each session is its own process: termination destroys the CUDA context and
    reclaims all VRAM, so a transient OOM kills only that one session. When
    ``gpu`` is given, ``CUDA_VISIBLE_DEVICES`` pins it to that physical GPU
    (the runner resolves device "auto"->"0", i.e. the only visible device), so
    two parallel streams never land on the same GPU. The box's env rules
    (KMP/IO encoding) are always set in case the child doesn't inherit them.
    """
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

    # The per-process session unit is (seed, condition, sigma). Sigma only
    # multiplies condition B (the simulated clinician's correction noise); A/ref
    # are sigma-independent, so they get a single sigma=None session that does
    # NOT pass --sigmas. e.g. `--conditions B --sigmas 5 10 20 40 --seeds 0..5`
    # expands to 24 B sessions (one per seed x sigma).
    jobs = []
    for seed in args.seeds:
        for cond in args.conditions:
            if cond == "B":
                jobs.extend((seed, cond, sigma) for sigma in args.sigmas)
            else:
                jobs.append((seed, cond, None))
    results: dict = {}
    # Forward the backbone / epoch schedule to each child base-runner only when set.
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
            # match the base runner's B_s{seed}_sig{sigma:g} tag formatting
            extra += ["--sigmas", f"{sigma:g}"]
        extra += weights_extra + epoch_extra
        rc = _run(extra, args.out, gpu=gpu)
        sig_tag = f"_sig{sigma:g}" if sigma is not None else ""
        g = f" (gpu{gpu})" if gpu is not None else ""
        print(f"  -> session s{seed} {cond}{sig_tag}{g} exit={rc}", flush=True)
        return f"s{seed}_{cond}{sig_tag}", rc

    if args.gpus and len(args.gpus) > 1:
        # Two (or more) GPUs. A static round-robin assignment (gpus[i % n]) is
        # WRONG: ThreadPoolExecutor only bounds *total* concurrency, not which
        # GPU a freed worker picks up next, so uneven session durations (e.g.
        # round-0 30ep vs 10ep, OOM retries) let a worker that just freed gpu0
        # start a gpu1-tagged job while gpu1 is still busy -> two sessions on
        # one GPU, the other idle. Instead, hold the GPU ids in a queue and have
        # each session *claim* a free GPU on start and *release* it on finish.
        # With max_workers == #GPUs and one id per GPU, this guarantees exactly
        # one in-flight session per GPU regardless of completion order.
        gpus = list(args.gpus)
        gpu_pool: "queue.Queue[int]" = queue.Queue()
        for g in gpus:
            gpu_pool.put(g)

        def _session_pooled(seed, cond, sigma):
            gpu = gpu_pool.get()  # block until a GPU is free, then claim it
            try:
                return _session(seed, cond, sigma, gpu)
            finally:
                gpu_pool.put(gpu)  # release for the next queued session

        with cf.ThreadPoolExecutor(max_workers=len(gpus)) as ex:
            futs = {ex.submit(_session_pooled, seed, cond, sigma): (seed, cond, sigma)
                    for seed, cond, sigma in jobs}
            for fut in cf.as_completed(futs):
                key, rc = fut.result()
                results[key] = rc
    else:
        # Single-GPU (or unspecified) serial path — original behaviour. A lone
        # --gpus N still pins the device; None leaves device selection to auto.
        gpu = args.gpus[0] if args.gpus else None
        for seed, cond, sigma in jobs:
            key, rc = _session(seed, cond, sigma, gpu)
            results[key] = rc

    # Final aggregation: rebuild the master CSV over THIS run's grid. Must pass
    # the same --seeds/--conditions, otherwise the base runner falls back to its
    # default seeds (0,1,2) and the master CSV silently drops every other seed
    # (e.g. a --seeds 0..5 run would aggregate only 0,1,2). All sessions are
    # already complete -> skipped -> no GPU work, purely a CSV rebuild.
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
