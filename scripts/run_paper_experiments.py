"""Canonical multi-seed iterative AL experiment runner for the paper.

The runner characterizes strategy performance across the complete label-budget
curve without presupposing a particular ranking.

Protocol (per strategy, per seed):
  - Round 0: shared random init set (same across all strategies for fair start)
             = budgets[0] fraction of the pool.
  - Round r: train on labeled set -> record (budget, mAP);
             score the pool with that round's model and acquire the next
             increment via the strategy (excluding already-labeled).
             Labeled set grows to budgets[r+1].

Canonical settings:
  budgets  = [0.05, 0.10, 0.20, 0.40, 1.00]
  seeds    = [0, 1, 2]
  strategies = [random, uncertainty, coreset, temporal, rehabal]
  epochs   = 10, imgsz = 416
  out      = outputs/experiments/paper_main

Resumable: a (strategy, budget, seed) cell is skipped if its output dir and
recorded metric already exist in curve.csv. curve.csv / summary.json are
rewritten from the union of cached + new rows after every completed cell.

AUC is computed PER-SEED then mean+/-std, NOT AUC of the mean curve.

Usage::
    python scripts/run_paper_experiments.py
    python scripts/run_paper_experiments.py --smoke
    python scripts/run_paper_experiments.py --out outputs/experiments/paper_main
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Lazy sys.path bootstrap: scripts/ dir may not be on PYTHONPATH.             #
# Import helpers from al_budget_curve (same directory).                       #
# --------------------------------------------------------------------------- #

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))

from al_budget_curve import (  # noqa: E402
    DATASET,
    MANIFEST,
    _build_pool,
    _free_cuda,
    _recording_meta,
    _train_eval as _budget_curve_train_eval,
)
from rehabal_pose.al import get_strategy
from rehabal_pose.data import ManifestBuilder
from rehabal_pose.metrics.learning_curve import area_under_learning_curve
from rehabal_pose.utils.io import ensure_dir, write_json
from rehabal_pose.utils.logging import get_logger
from rehabal_pose.utils.seed import set_seed

_LOG = get_logger("paper_exp")

# --------------------------------------------------------------------------- #
# Default canonical settings                                                   #
# --------------------------------------------------------------------------- #

DEFAULT_BUDGETS: list[float] = [0.05, 0.10, 0.20, 0.40, 1.00]
DEFAULT_SEEDS: list[int] = [0, 1, 2]
DEFAULT_STRATEGIES: list[str] = ["random", "uncertainty", "coreset", "temporal", "rehabal"]
DEFAULT_EPOCHS: int = 10
DEFAULT_IMGSZ: int = 416
DEFAULT_BATCH: int = 16
DEFAULT_OUT: str = "outputs/experiments/paper_main"


# --------------------------------------------------------------------------- #
# Injectable training hook (wraps the real _train_eval from al_budget_curve)  #
# --------------------------------------------------------------------------- #

def _default_train_eval(
    selected_imgs: list[str],
    out_dir: Path,
    epochs: int,
    imgsz: int,
    seed: int,
    batch: int,
) -> dict:
    """Default training hook: delegates to al_budget_curve._train_eval.

    Tests can inject a replacement via ``main(argv, train_eval=fake_fn)``.
    """
    return _budget_curve_train_eval(selected_imgs, out_dir, epochs, imgsz, seed, batch)


# --------------------------------------------------------------------------- #
# Resume helpers                                                                #
# --------------------------------------------------------------------------- #

def _load_existing_rows(csv_path: Path) -> pd.DataFrame:
    """Load existing curve.csv rows if the file exists, else return empty df."""
    if csv_path.is_file():
        try:
            return pd.read_csv(csv_path)
        except Exception:  # noqa: BLE001
            _LOG.warning("Could not read existing %s; starting fresh.", csv_path)
    return pd.DataFrame(
        columns=["strategy", "budget", "seed", "n_labeled", "pose_map50", "pose_map50_95"]
    )


def _cell_done(df: pd.DataFrame, strategy: str, budget: float, seed: int) -> bool:
    """Return True if this (strategy, budget, seed) cell already has a metric row."""
    if df.empty:
        return False
    mask = (
        (df["strategy"] == strategy)
        & (np.isclose(df["budget"].astype(float), budget))
        & (df["seed"].astype(int) == seed)
    )
    sub = df[mask]
    if sub.empty:
        return False
    # Consider done only if the metric value is present (not NaN).
    val = sub["pose_map50_95"].iloc[-1]
    try:
        return float(val) == float(val)  # False if NaN
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# Pool scoring with the round's model (real version — GPU)                    #
# --------------------------------------------------------------------------- #

def _score_pool_with_weights(pool_imgs: list[str], weights: str, rec_meta: dict) -> object:
    """Build Pool by scoring with a trained model (real; GPU path).

    This is only called from the real run, not during tests (tests inject a
    fake strategy that never needs the pool).
    """
    from rehabal_pose.models.yolo_pose import YoloPoseModel  # guarded import

    model = YoloPoseModel().load(weights)
    pool = _build_pool(pool_imgs, model, rec_meta)
    del model
    _free_cuda()
    return pool


# --------------------------------------------------------------------------- #
# Core experiment loop                                                          #
# --------------------------------------------------------------------------- #

def _run_experiments(
    pool_imgs: list[str],
    rec_meta: dict,
    strategies: list[str],
    budgets: list[float],
    seeds: list[int],
    epochs: int,
    imgsz: int,
    batch: int,
    out_root: Path,
    train_eval: Callable,
    strategy_factory: Optional[Callable[[str], object]],
    existing_rows: pd.DataFrame,
    pool_factory: Optional[Callable] = None,
) -> pd.DataFrame:
    """Execute (or skip) all (strategy, budget, seed) cells.

    Parameters
    ----------
    pool_imgs:
        Sorted list of training image paths.
    rec_meta:
        Recording metadata from the manifest.
    strategies, budgets, seeds:
        Experiment grid.
    epochs, imgsz, batch:
        Training hyper-parameters.
    out_root:
        Root output directory.
    train_eval:
        Injected training function with signature
        ``(selected_imgs, out_dir, epochs, imgsz, seed, batch) -> dict``.
    strategy_factory:
        Optional callable ``(name: str) -> AcquisitionStrategy``.
        Defaults to ``get_strategy`` (real strategies).
    existing_rows:
        Already-recorded rows loaded from a previous run's curve.csv.
    pool_factory:
        Optional callable ``(pool_imgs, weights, rec_meta) -> Pool``.
        Defaults to ``_score_pool_with_weights`` (real GPU path).
        Tests inject a stub so no model loading occurs.

    Returns
    -------
    pd.DataFrame
        All rows (existing + newly computed).
    """
    _get_strat = strategy_factory if strategy_factory is not None else get_strategy
    _build_scored_pool = (
        pool_factory if pool_factory is not None else _score_pool_with_weights
    )

    N = len(pool_imgs)
    budgets = sorted(budgets)
    targets = [N if b >= 1.0 else max(1, int(round(b * N))) for b in budgets]
    _LOG.info("Pool %d imgs; budgets %s -> targets %s", N, budgets, targets)

    rows = list(existing_rows.to_dict("records"))

    for seed in seeds:
        # Shared random init set (identical across strategies for a fair start).
        init_rng = np.random.default_rng(1000 + seed)
        init_idx = list(init_rng.choice(N, size=targets[0], replace=False))

        for strategy in strategies:
            labeled = list(init_idx)
            weights = None  # previous round's trained weights (scorer for next round)

            for r, (budget, target) in enumerate(zip(budgets, targets)):
                cell_tag = f"{strategy}_b{budget:g}_s{seed}"

                # --- resume check -------------------------------------------
                # Rebuild the DataFrame from the running rows list each iteration
                # so the resume check reflects newly completed cells too.
                current_df = pd.DataFrame(rows) if rows else pd.DataFrame(
                    columns=["strategy", "budget", "seed", "n_labeled",
                             "pose_map50", "pose_map50_95"]
                )
                if _cell_done(current_df, strategy, budget, seed):
                    _LOG.info("[%s] SKIP (already recorded)", cell_tag)
                    cell_dir = out_root / cell_tag
                    # Restore weights path for the next round's pool scorer.
                    maybe_w = cell_dir / "run" / "train" / "weights" / "best.pt"
                    if maybe_w.is_file():
                        weights = str(maybe_w)
                    # Restore labeled indices from the persisted snapshot so that
                    # subsequent rounds acquire from the exact same prefix.
                    idx_file = cell_dir / "labeled_idx.json"
                    if idx_file.is_file():
                        try:
                            labeled = json.load(idx_file.open(encoding="utf-8"))
                            _LOG.info("[%s] restored %d labeled indices from %s",
                                      cell_tag, len(labeled), idx_file)
                        except Exception as exc:  # noqa: BLE001
                            _LOG.warning("[%s] could not load %s: %s; "
                                         "falling back to random-fill", cell_tag, idx_file, exc)
                            if len(labeled) < target:
                                rng = np.random.default_rng(seed * 31 + r)
                                rem = [i for i in range(N) if i not in set(labeled)]
                                need = min(target - len(labeled), len(rem))
                                labeled += list(rng.choice(rem, size=need, replace=False))
                    else:
                        # Backward-compat: old cell without labeled_idx.json.
                        _LOG.warning("[%s] labeled_idx.json not found in %s; "
                                     "using random-fill (labeled prefix may differ from "
                                     "a fresh run — subsequent AUC may be inconsistent)",
                                     cell_tag, cell_dir)
                        if len(labeled) < target:
                            rng = np.random.default_rng(seed * 31 + r)
                            rem = [i for i in range(N) if i not in set(labeled)]
                            need = min(target - len(labeled), len(rem))
                            labeled += list(rng.choice(rem, size=need, replace=False))
                    continue

                # --- acquire up to target ------------------------------------
                if len(labeled) < target:
                    if weights is None:
                        # cold start: random fill (no model yet for scoring)
                        rng = np.random.default_rng(seed * 97 + r)
                        rem = [i for i in range(N) if i not in set(labeled)]
                        labeled += list(
                            rng.choice(rem, size=target - len(labeled), replace=False)
                        )
                    else:
                        # score pool with current round's model and acquire
                        pool = _build_scored_pool(pool_imgs, weights, rec_meta)
                        cand = [i for i in range(N) if i not in set(labeled)]
                        set_seed(seed * 97 + r)
                        strat_obj = _get_strat(strategy)
                        sel = strat_obj.select(pool, target - len(labeled), candidates=cand)
                        labeled += [i for i in sel if i not in set(labeled)]
                        # top up if strategy returned fewer than requested
                        if len(labeled) < target:
                            rng = np.random.default_rng(seed * 13 + r)
                            rem = [i for i in range(N) if i not in set(labeled)]
                            labeled += list(
                                rng.choice(
                                    rem,
                                    size=min(target - len(labeled), len(rem)),
                                    replace=False,
                                )
                            )

                # --- train + eval --------------------------------------------
                sel_imgs = [pool_imgs[i] for i in labeled]
                cell_dir = out_root / cell_tag
                _LOG.info("[%s] round %d: train on %d imgs", cell_tag, r, len(sel_imgs))
                res = train_eval(sel_imgs, cell_dir, epochs, imgsz, seed, batch)
                weights = res.get("weights")

                # Persist labeled indices so resume can reconstruct the exact prefix.
                # Cast to plain int to avoid np.int64 JSON serialisation errors.
                ensure_dir(cell_dir)
                idx_file = cell_dir / "labeled_idx.json"
                write_json(idx_file, [int(i) for i in sorted(labeled)])

                _nn = lambda v: float("nan") if v is None else float(v)
                _LOG.info(
                    "[%s] pose mAP50=%.4f mAP50-95=%.4f",
                    cell_tag,
                    _nn(res.get("map50")),
                    _nn(res.get("map50_95")),
                )

                rows.append({
                    "strategy": strategy,
                    "budget": budget,
                    "seed": seed,
                    "n_labeled": len(sel_imgs),
                    "pose_map50": res.get("map50"),
                    "pose_map50_95": res.get("map50_95"),
                })
                # Incremental write: crash-safe so any restart can resume from
                # exactly this point (rows already contains existing+new union).
                pd.DataFrame(rows).to_csv(out_root / "curve.csv", index=False)

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["strategy", "budget", "seed", "n_labeled", "pose_map50", "pose_map50_95"]
    )


# --------------------------------------------------------------------------- #
# Summary: mean±std per (strategy, budget) and per-strategy AUC mean±std      #
# --------------------------------------------------------------------------- #

def _build_summary(df: pd.DataFrame, strategies: list[str], budgets: list[float]) -> dict:
    """Build summary dict: per strategy per budget mean±std, plus per-strategy AUC.

    AUC is computed PER-SEED first, then mean±std across seeds — NOT AUC of
    the mean curve (as required by the paper protocol).
    """
    summary: dict = {}

    for strategy in strategies:
        strat_df = df[df["strategy"] == strategy]
        if strat_df.empty:
            summary[strategy] = {"budgets": {}, "auc_map50_95": {"mean": None, "std": None},
                                 "auc_map50": {"mean": None, "std": None}}
            continue

        # Per-budget mean±std across seeds.
        budget_stats: dict = {}
        for b in sorted(budgets):
            sub = strat_df[np.isclose(strat_df["budget"].astype(float), b)]
            if sub.empty:
                budget_stats[float(b)] = {
                    "map50_mean": None, "map50_std": None,
                    "map50_95_mean": None, "map50_95_std": None,
                    "n_seeds": 0,
                }
                continue
            vals_50 = sub["pose_map50"].astype(float).values
            vals_95 = sub["pose_map50_95"].astype(float).values
            budget_stats[float(b)] = {
                "map50_mean": float(np.nanmean(vals_50)),
                "map50_std": float(np.nanstd(vals_50, ddof=0)),
                "map50_95_mean": float(np.nanmean(vals_95)),
                "map50_95_std": float(np.nanstd(vals_95, ddof=0)),
                "n_seeds": int(len(sub)),
            }

        # AUC per seed, then mean±std.
        auc_95_per_seed: list[float] = []
        auc_50_per_seed: list[float] = []
        for seed_val in strat_df["seed"].unique():
            seed_sub = strat_df[strat_df["seed"].astype(int) == int(seed_val)].sort_values("budget")
            bs = seed_sub["budget"].astype(float).tolist()
            vs_95 = seed_sub["pose_map50_95"].astype(float).tolist()
            vs_50 = seed_sub["pose_map50"].astype(float).tolist()
            if len(bs) >= 2:
                auc_95_per_seed.append(
                    float(area_under_learning_curve(bs, vs_95, normalize=True))
                )
                auc_50_per_seed.append(
                    float(area_under_learning_curve(bs, vs_50, normalize=True))
                )
            else:
                auc_95_per_seed.append(float("nan"))
                auc_50_per_seed.append(float("nan"))

        def _mean_std(vals: list[float]) -> tuple[float, float]:
            arr = np.array([v for v in vals if v == v], dtype=float)  # drop NaN
            if arr.size == 0:
                return float("nan"), float("nan")
            return float(np.mean(arr)), float(np.std(arr, ddof=0))

        m95, s95 = _mean_std(auc_95_per_seed)
        m50, s50 = _mean_std(auc_50_per_seed)

        summary[strategy] = {
            "budgets": budget_stats,
            "auc_map50_95": {"mean": m95, "std": s95},
            "auc_map50": {"mean": m50, "std": s50},
        }

    return summary


# --------------------------------------------------------------------------- #
# main()                                                                        #
# --------------------------------------------------------------------------- #

def main(
    argv: Optional[list[str]] = None,
    train_eval: Optional[Callable] = None,
    strategy_factory: Optional[Callable[[str], object]] = None,
    pool_factory: Optional[Callable] = None,
) -> int:
    """Run the canonical paper experiments.

    Parameters
    ----------
    argv:
        Argument list (``None`` = sys.argv[1:]`).
    train_eval:
        Injectable training hook for tests. When ``None`` the real
        :func:`_default_train_eval` is used.
    strategy_factory:
        Injectable strategy constructor for tests. When ``None`` the real
        :func:`rehabal_pose.al.get_strategy` is used.
    pool_factory:
        Injectable pool-building hook for tests. When ``None`` the real
        :func:`_score_pool_with_weights` (GPU path) is used.

    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--strategies",
        nargs="+",
        default=DEFAULT_STRATEGIES,
        help="AL strategies to benchmark (default: all 5 canonical strategies)",
    )
    ap.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=DEFAULT_BUDGETS,
        help="Label-budget fractions (default: 0.05 0.10 0.20 0.40 1.00)",
    )
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds (default: 0 1 2)",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Training epochs per cell (default: 10)",
    )
    ap.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help="Image size for YOLO training (default: 416)",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help="Training batch size (default: 16)",
    )
    ap.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"Output root directory (default: {DEFAULT_OUT})",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: 2 strategies, 2 budgets, 1 epoch, 1 seed",
    )
    args = ap.parse_args(argv)

    # Smoke-test overrides.
    if args.smoke:
        args.strategies = ["random", "rehabal"]
        args.budgets = [0.05, 0.10]
        args.epochs = 1
        args.seeds = [0]
        _LOG.info("Smoke mode: strategies=%s budgets=%s epochs=%d seeds=%s",
                  args.strategies, args.budgets, args.epochs, args.seeds)

    out_root = ensure_dir(Path(args.out))
    csv_path = out_root / "curve.csv"
    json_path = out_root / "summary.json"

    # Resolve train_eval hook.
    _train_eval_fn = train_eval if train_eval is not None else _default_train_eval

    # Load pool images.
    pool_imgs = sorted(glob.glob(str(DATASET / "images" / "train" / "*.jpg")))
    if not pool_imgs:
        _LOG.error("No images under %s/images/train/. Build the dataset first.", DATASET)
        return 1

    # Load manifest and recording metadata.
    manifest_df = ManifestBuilder.load(MANIFEST)
    rec_meta = _recording_meta(manifest_df)

    # Load existing rows for resume.
    existing_rows = _load_existing_rows(csv_path)
    n_existing = len(existing_rows)
    if n_existing:
        _LOG.info("Resuming: loaded %d existing rows from %s", n_existing, csv_path)

    # Run the experiment grid.
    df = _run_experiments(
        pool_imgs=pool_imgs,
        rec_meta=rec_meta,
        strategies=args.strategies,
        budgets=args.budgets,
        seeds=args.seeds,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        out_root=out_root,
        train_eval=_train_eval_fn,
        strategy_factory=strategy_factory,
        existing_rows=existing_rows,
        pool_factory=pool_factory,
    )

    # Save curve.csv (always rewrite from union of cached+new rows).
    df.to_csv(csv_path, index=False)
    _LOG.info("Saved curve.csv (%d rows) -> %s", len(df), csv_path)

    # Build and save summary.json.
    summary = _build_summary(df, args.strategies, args.budgets)
    write_json(json_path, summary)
    _LOG.info("Saved summary.json -> %s", json_path)

    # Log per-strategy AUC.
    for strat, info in summary.items():
        auc_m = info["auc_map50_95"]["mean"]
        auc_s = info["auc_map50_95"]["std"]
        if auc_m is not None:
            _LOG.info("AUC mAP50-95 [%s]: %.4f ± %.4f", strat, auc_m, auc_s)

    # Optional plot (matplotlib guarded).
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        plt.figure(figsize=(8, 5))
        for s in args.strategies:
            pts = summary[s]["budgets"]
            xs = sorted(pts)
            ys = [pts[b]["map50_95_mean"] for b in xs if pts[b]["map50_95_mean"] is not None]
            xs_valid = [b for b in xs if pts[b]["map50_95_mean"] is not None]
            auc_m = summary[s]["auc_map50_95"]["mean"]
            label = f"{s} (AUC={auc_m:.3f})" if auc_m is not None else s
            plt.plot(xs_valid, ys, marker="o",
                     linewidth=3 if s == "rehabal" else 1.5,
                     label=label)
        plt.xlabel("label budget")
        plt.ylabel("pose mAP50-95")
        plt.title("RehabAL-Pose: Paper Main Results (REHAB24-6)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_root / "learning_curve.png", dpi=120)
        _LOG.info("Saved learning_curve.png")
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("plot skipped: %s", exc)

    print("PAPER_EXP_DONE", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
